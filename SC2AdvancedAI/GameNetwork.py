import numpy as np
from sc2.bot_ai import BotAI
from macro import MacroBot
from micro import MicroBot
import time
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps
from sc2.data import Race, Difficulty
from collections import deque
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

class SwapperBot(BotAI):
    def __init__(self):
        super().__init__()
        self.macro_bot = MacroBot(self)
        self.micro_bot = MicroBot(self)
        self.current_bot = self.macro_bot
        self.strategy_model = self.create_strategy_model(input_shape=6, num_actions=2)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_frequency = 10
        self.last_update_time = time.time()
        self.update_interval = 30  # Update interval in seconds

    def create_strategy_model(self, input_shape, num_actions):
        model = Sequential()
        model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_actions, activation='softmax'))  # Softmax for strategy probabilities
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    async def on_step(self, iteration: int):
        state = await self.get_strategy_state()
        action = self.choose_strategy(state)
        
        if action == 0:
            self.current_bot = self.macro_bot
        else:
            self.current_bot = self.micro_bot

        await self.current_bot.on_step(iteration)

        # Perform learning updates for strategy model
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            reward = await self.calculate_strategy_reward()
            next_state = await self.get_strategy_state()
            self.store_strategy_experience(state, action, reward, next_state)
            self.replay_strategy()
            self.last_update_time = current_time

    def choose_strategy(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(2)
        probabilities = self.strategy_model.predict(state)
        return np.argmax(probabilities[0])

    def store_strategy_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay_strategy(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*minibatch)

        states = np.array([s.flatten() for s in states])
        next_states = np.array([s.flatten() for s in next_states])
        targets = self.strategy_model.predict(states)

        for i in range(self.batch_size):
            target = rewards[i]  # Adjust this if you need more complex reward calculation
            targets[i][actions[i]] = target

        self.strategy_model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    async def get_strategy_state(self):
        state = np.array([
            self.macro_bot.bot_ai.minerals,
            self.macro_bot.bot_ai.supply_used,
            self.macro_bot.bot_ai.supply_cap,
            len(self.macro_bot.bot_ai.units(UnitTypeId.DRONE)),
            len(self.macro_bot.bot_ai.units(UnitTypeId.QUEEN)),
            len(self.macro_bot.bot_ai.structures(UnitTypeId.HATCHERY))
        ])
        return state.reshape(1, -1)

    async def calculate_strategy_reward(self):
        # Define how you calculate reward for strategies
        return random.random()

    async def run(self):
        run_game(
            maps.get("Abyssal Reef LE"),
            [Bot(Race.Zerg, self), Computer(Race.Terran, Difficulty.Medium)],
            realtime=False
        )

if __name__ == "__main__":
    bot = SwapperBot()
    bot.run()
