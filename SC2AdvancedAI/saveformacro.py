import os
import time
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from macromanager import MacroManager
class MacroBot(BotAI):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.model = self.create_model(input_shape=6, num_actions=6)
        self.target_model = self.create_model(input_shape=6, num_actions=6)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_frequency = 10
        self.previous_drones = 0
        self.total_reward = 0

        # Initialize the MacroManager
        self.macro_manager = MacroManager(self)

        # Initialize the time tracking variables
        self.last_update_time = time.time()
        self.update_interval = 30  # Update interval in seconds

        # Load model weights if available
        self.load_model()

    async def on_step(self, iteration: int):
        state = await self.get_game_state()
        action = self.choose_action(state)
        await self.execute_action(action)

        # Check if it's time to perform learning updates
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            reward = await self.calculate_reward()
            next_state = await self.get_game_state()
            self.store_experience(state, action, reward, next_state)
            self.replay()
            self.update_target_model(iteration)
            self.last_update_time = current_time  # Update the last update time

        await self.macro_manager.run_periodic_tasks(iteration)

    def create_model(self, input_shape, num_actions):
        model = Sequential()
        model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(6)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        self.total_reward += reward

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*minibatch)

        states = np.array([s.flatten() for s in states])
        next_states = np.array([s.flatten() for s in next_states])
        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        
        for i in range(self.batch_size):
            target = rewards[i] + self.gamma * np.amax(next_q_values[i])
            targets[i][actions[i]] = target
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self, iteration):
        if iteration % self.update_target_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())

    async def get_game_state(self):
        minerals = self.minerals
        supply_used = self.supply_used
        supply_cap = self.supply_cap
        num_drones = len(self.units(UnitTypeId.DRONE))
        num_queens = len(self.units(UnitTypeId.QUEEN))
        num_hatcheries = len(self.structures(UnitTypeId.HATCHERY))
        
        state = np.array([
            minerals,
            supply_used,
            supply_cap,
            num_drones,
            num_queens,
            num_hatcheries
        ])
        return state.reshape(1, -1)

    async def execute_action(self, action):
        if action == 0:
            if self.can_afford(UnitTypeId.DRONE):
                self.train(UnitTypeId.DRONE)
        elif action == 1:
            if self.can_afford(UnitTypeId.OVERLORD):
                self.train(UnitTypeId.OVERLORD)
        elif action == 2:
            if self.can_afford(UnitTypeId.QUEEN) and self.structures(UnitTypeId.SPAWNINGPOOL).exists:
                self.train(UnitTypeId.QUEEN)
        elif action == 3:
            await self.build_spawning_pool()
        elif action == 4:
            if self.can_afford(UnitTypeId.HATCHERY):
                await self.expand_now()
        elif action == 5:
            await self.build_extractor_and_assign_drones()

    async def build_spawning_pool(self):
        hatchery = self.townhalls.random
        if not self.structures(UnitTypeId.SPAWNINGPOOL) and self.already_pending(UnitTypeId.SPAWNINGPOOL) == 0:
            if self.can_afford(UnitTypeId.SPAWNINGPOOL):
                await self.build(UnitTypeId.SPAWNINGPOOL, near=hatchery)

    async def expand(self):
        if self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now()

    async def calculate_reward(self):
        current_drones = len(self.units(UnitTypeId.DRONE))
        idle_drones = len(self.units(UnitTypeId.DRONE).idle)
        
        drone_count_reward = current_drones - self.previous_drones
        
        idle_penalty = -0.1 * idle_drones
        
        reward = drone_count_reward + idle_penalty
        
        self.previous_drones = current_drones
        
        return reward

    async def build_extractor_and_assign_drones(self):
        hatchery = self.townhalls.random
        vespene_geysers = self.vespene_geyser.closer_than(20, hatchery.position)
        
        if not vespene_geysers:
            return

        for geyser in vespene_geysers:
            if not self.structures(UnitTypeId.EXTRACTOR).closer_than(1, geyser.position).exists:
                if self.can_afford(UnitTypeId.EXTRACTOR):
                    built_extractor = await self.build(UnitTypeId.EXTRACTOR, near=geyser)
                    break

    def save_model(self):
        self.model.save("model.h5")
        print("Model saved!")

    def load_model(self):
        if os.path.exists("model.h5"):
            self.model = tf.keras.models.load_model("model.h5")
            self.target_model = self.create_model(input_shape=6, num_actions=6)
            self.target_model.set_weights(self.model.get_weights())
            print("Model loaded!")
        else:
            print("No pre-trained model found, starting from scratch.")

def run_multiple_games(num_games):
    for _ in range(num_games):
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Zerg, MacroBot()),
            Computer(Race.Terran, Difficulty.Easy)],  # PassiveComputer replaced with Computer
            realtime=True)  # Set to False for non-real-time mode

if __name__ == "__main__":
    run_multiple_games(3)
