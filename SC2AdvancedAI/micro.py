import os
import time
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
from sc2.bot_ai import BotAI
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps
from sc2.data import Race, Difficulty

class MicroBot(BotAI):
    def __init__(self):
        super().__init__()
        self.model = self.create_micro_model(input_shape=60, num_actions=4)
        self.target_model = self.create_micro_model(input_shape=60, num_actions=4)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_frequency = 10
        self.start_time = time.time()
        self.max_duration = 5 * 60  # 5 minutes
        self.defend_location = None
        self.attack_location = None
        self.last_damage_time = 0
        self.damage_threshold = 5  # Time in seconds to consider if the base is under attack
        self.upgrade_tasks = [
            ("ZERGLINGMOVEMENTSPEED", UnitTypeId.SPAWNINGPOOL, UpgradeId.ZERGLINGMOVEMENTSPEED),
            ("BANELING_NEST", UnitTypeId.BANELINGNEST, None),
            ("EVO_CHAMBER", UnitTypeId.EVOLUTIONCHAMBER, None),
            ("MELEE_UPGRADE_L1", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGMELEEWEAPONSLEVEL1),
            ("RANGED_UPGRADE_L1", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGMISSILEWEAPONSLEVEL1),
            ("LAIR", UnitTypeId.LAIR, None),
            ("HYDRALISK_DEN", UnitTypeId.HYDRALISKDEN, None),
            ("HYDRALISK_UPGRADE1", UnitTypeId.HYDRALISKDEN, UpgradeId.EVOLVEGROOVEDSPINES),
            ("EVO_CHAMBER_UPGRADE_L2", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGMELEEWEAPONSLEVEL2),
            ("EVO_CHAMBER_UPGRADE_R2", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGMISSILEWEAPONSLEVEL2),
            ("HEALTH_UPGRADE_L1", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGGROUNDARMORSLEVEL1),
            ("HEALTH_UPGRADE_L2", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGGROUNDARMORSLEVEL2),
            ("HIVE", UnitTypeId.HIVE, None),
            ("ADRENALINE_GLANDS", UnitTypeId.SPAWNINGPOOL, UpgradeId.ZERGLINGATTACKSPEED),
            ("EVO_CHAMBER_UPGRADE_MELEE_L3", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGMELEEWEAPONSLEVEL3),
            ("EVO_CHAMBER_UPGRADE_RANGED_L3", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGMISSILEWEAPONSLEVEL3),
            ("HYDRALISK_UPGRADE2", UnitTypeId.HYDRALISKDEN, UpgradeId.EVOLVEMUSCULARAUGMENTS),
            ("HEALTH_UPGRADE_L3", UnitTypeId.EVOLUTIONCHAMBER, UpgradeId.ZERGGROUNDARMORSLEVEL3),
        ]
        self.current_task_index = 0  # To track the current task in the upgrade sequence

        # Load model weights if available
        self.load_model()

    def create_micro_model(self, input_shape, num_actions):
        model = Sequential()
        model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_actions, activation='softmax'))  # Softmax for action probabilities
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    async def get_micro_state(self):
        friendly_units_health = []
        friendly_units_positions = []
        enemy_units_health = []
        enemy_units_positions = []

        TOP_N_UNITS = 10
        for unit in self.units:
            if len(friendly_units_health) < TOP_N_UNITS:
                friendly_units_health.append(unit.health)
                friendly_units_positions.append((unit.position.x, unit.position.y))
            else:
                break
        for unit in self.enemy_units:
            if len(enemy_units_health) < TOP_N_UNITS:
                enemy_units_health.append(unit.health)
                enemy_units_positions.append((unit.position.x, unit.position.y))
            else:
                break

        while len(friendly_units_health) < TOP_N_UNITS:
            friendly_units_health.append(0)
            friendly_units_positions.append((0, 0))
        
        while len(enemy_units_health) < TOP_N_UNITS:
            enemy_units_health.append(0)
            enemy_units_positions.append((0, 0))

        friendly_positions_flat = [coord for pos in friendly_units_positions for coord in pos]
        enemy_positions_flat = [coord for pos in enemy_units_positions for coord in pos]

        state = np.array(
            friendly_units_health +
            friendly_positions_flat +
            enemy_units_health +
            enemy_positions_flat
        )
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)  # Normalize state

        return state.reshape(1, -1)  # Ensure state is of shape (1, 60)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(4)  # Assuming 4 actions
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    async def execute_action(self, action):
        if action == 0:
            await self.train_unit()
            print('train')
        elif action == 1:
            await self.upgrade_units()
            print('upgrade')
        elif action == 2:
            await self.defend()
            print('defend')
        elif action == 3:
            await self.attack()
            print('attack')

    async def train_unit(self):
        # Get counts of current units
        zerglings_count = len(self.units(UnitTypeId.ZERGLING))
        hydralisks_count = len(self.units(UnitTypeId.HYDRALISK))
        banelings_count = len(self.units(UnitTypeId.BANELING))

        # Check for the presence of required buildings
        has_baneling_nest = self.structures(UnitTypeId.BANELINGNEST).exists
        has_hydralisk_den = self.structures(UnitTypeId.HYDRALISKDEN).exists

        if not has_baneling_nest and not has_hydralisk_den:
            # Only train Zerglings if neither building is available
            target_zerglings = 20
            target_hydralisks = 0
            target_banelings = 0
        elif has_baneling_nest and not has_hydralisk_den:
            # Train Zerglings and Banelings if Baneling Nest is available but no Hydralisk Den
            total_units = zerglings_count + banelings_count
            target_zerglings = int(total_units * 0.70) + max(5, 20 - total_units)  # Ensure at least some Zerglings
            target_banelings = int(total_units * 0.30)
            target_hydralisks = 0
        elif has_baneling_nest and has_hydralisk_den:
            # Use the 50/30/20 composition if both buildings are available
            total_units = zerglings_count + hydralisks_count + banelings_count
            if total_units > 0:
                target_zerglings = int(total_units * 0.50)
                target_hydralisks = int(total_units * 0.30)
                target_banelings = int(total_units * 0.20)
            else:
                target_zerglings = 10
                target_hydralisks = 0
                target_banelings = 0
        else:
            # Default case (shouldn't be reached due to previous conditions)
            target_zerglings = 20
            target_hydralisks = 0
            target_banelings = 0

        # Train or morph units to meet the desired composition

        # Train Zerglings if below the target count
        if zerglings_count < target_zerglings:
            if self.can_afford(UnitTypeId.ZERGLING):
                self.train(UnitTypeId.ZERGLING)
        
        # Train Hydralisks if below the target count and Hydralisk Den exists
        if hydralisks_count < target_hydralisks and has_hydralisk_den:
            if self.can_afford(UnitTypeId.HYDRALISK):
                self.train(UnitTypeId.HYDRALISK)
        
        # Morph Zerglings into Banelings if below the target count and Baneling Nest exists
        if banelings_count < target_banelings and has_baneling_nest:
            zerglings = self.units(UnitTypeId.ZERGLING)
            if zerglings.exists:
                for zergling in zerglings:
                    if self.can_afford(AbilityId.MORPHTOBANELING_BANELING):
                        self.do(zergling(AbilityId.MORPHTOBANELING_BANELING))
                        break

    async def upgrade_units(self):
        # Check if there are any more tasks to process
        if self.current_task_index >= len(self.upgrade_tasks):
            print("All upgrade tasks are completed.")
            return

        # Get the current task details
        task_name, unit_type, upgrade_id = self.upgrade_tasks[self.current_task_index]

        # Handle Hatchery to Lair upgrade
        if unit_type == UnitTypeId.LAIR and upgrade_id is None:
            hatcheries = self.structures(UnitTypeId.HATCHERY)
            if hatcheries.exists:
                hatchery = hatcheries.first
                # Check if the bot can afford the upgrade and if it's not already in progress
                if self.can_afford(UnitTypeId.LAIR):
                    # Check if the upgrade is already in progress or completed
                    if not self.structures(UnitTypeId.LAIR).exists and self.already_pending(UnitTypeId.LAIR) == 0:
                        # Perform the upgrade action
                        self.do(hatchery(AbilityId.UPGRADETOLAIR_LAIR))
                        print("Initiating Hatchery to Lair upgrade.")
                    else:
                        print(f"{task_name} is under construction or completed. Skipping build initiation.")
                        self.current_task_index += 1
                        return
                else:
                    print("Not enough resources for Lair upgrade.")
                    return

            # Wait for the Lair to be built
            if self.structures(UnitTypeId.LAIR).exists:
                print(f"Lair has been built. Moving to next task.")
                self.current_task_index += 1
                return

        # Handle Lair to Hive upgrade
        if unit_type == UnitTypeId.HIVE and upgrade_id is None:
            lairs = self.structures(UnitTypeId.LAIR)
            if lairs.exists:
                lair = lairs.first
                # Check if the bot can afford the upgrade and if it's not already in progress
                if self.can_afford(UnitTypeId.HIVE):
                    if not self.structures(UnitTypeId.HIVE).exists and self.already_pending(UnitTypeId.HIVE) == 0:
                        # Perform the upgrade action
                        self.do(lair(AbilityId.UPGRADETOHIVE_HIVE))
                        print("Initiating Lair to Hive upgrade.")
                    else:
                        print(f"{task_name} is under construction or completed. Skipping build initiation.")
                        self.current_task_index += 1
                        return
                else:
                    print("Not enough resources for Hive upgrade.")
                    return

            # Wait for the Hive to be built
            if self.structures(UnitTypeId.HIVE).exists:
                print(f"Hive has been built. Moving to next task.")
                self.current_task_index += 1
                return

        # Handle building tasks (None means it's an upgrade task)
        if upgrade_id is None:
            # Check if the structure is being built or already exists
            if not self.structures(unit_type).exists:
                if self.already_pending(unit_type) == 0:
                    # Attempt to build the structure if it's not present
                    if self.can_afford(unit_type):
                        building_type = UnitTypeId(unit_type)
                        hatchery = self.townhalls.random
                        await self.build(building_type, near=hatchery)
                        print(f"Building {task_name}...")
                    else:
                        print(f"Not enough resources to build {task_name}.")
                        return
                else:
                    print(f"{task_name} is under construction. Skipping build initiation.")
                    return

            # Wait for the building to be completed
            if self.structures(unit_type).exists:
                print(f"Completed building {task_name}. Moving to next task.")
                self.current_task_index += 1
                return

        # Handle upgrade tasks (when upgrade_id is provided)
        if upgrade_id:
            building_type = {
                UpgradeId.ZERGLINGMOVEMENTSPEED: UnitTypeId.SPAWNINGPOOL,
                UpgradeId.EVOLVEGROOVEDSPINES: UnitTypeId.HYDRALISKDEN,
                UpgradeId.ZERGMELEEWEAPONSLEVEL1: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.ZERGMISSILEWEAPONSLEVEL1: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.ZERGGROUNDARMORSLEVEL1: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.ZERGMELEEWEAPONSLEVEL2: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.ZERGMISSILEWEAPONSLEVEL2: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.ZERGGROUNDARMORSLEVEL2: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.ZERGMELEEWEAPONSLEVEL3: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.ZERGMISSILEWEAPONSLEVEL3: UnitTypeId.EVOLUTIONCHAMBER,
                UpgradeId.EVOLVEMUSCULARAUGMENTS: UnitTypeId.HYDRALISKDEN,
                UpgradeId.ZERGLINGATTACKSPEED: UnitTypeId.SPAWNINGPOOL,
            }.get(upgrade_id, None)

            if building_type is None:
                print(f"Building type for {task_name} is not recognized. Waiting for correction.")
                return

            # Check if the required building is present
            if not self.structures(building_type).exists:
                print(f"Building for {task_name} is missing. Waiting for construction.")
                return

            # Get the building where the upgrade should be performed
            building = self.structures(building_type).first

            # Ensure the building is complete before starting the upgrade
            if self.already_pending(building_type) > 0:
                print(f"Building {building_type} is still under construction. Waiting to start upgrade.")
                return

            # Check if the upgrade is already in progress
            if self.already_pending_upgrade(upgrade_id) > 0:
                print(f"Upgrade {task_name} is in progress. Moving to next task.")
                self.current_task_index += 1
                return

            # Initiate the upgrade if resources are sufficient
            if self.can_afford(UpgradeId(upgrade_id)):
                # Perform the upgrade from the building
                self.do(building.research(UpgradeId(upgrade_id)))
                print(f"Started upgrading {task_name}. Moving to next task.")
                self.current_task_index += 1
                return
            else:
                print(f"Not enough resources to upgrade {task_name}.")
                return

        # In case the task index is out of range, reset or log it
        if self.current_task_index >= len(self.upgrade_tasks):
            print("Upgrade task index out of range. Resetting index.")
            self.current_task_index = 0
            self.upgrade_tasks = [
                # Reinitialize the task list if needed
            ]

    async def defend(self):
        if self.defend_location is None:
            self.defend_location = self.start_location

        # Filter out drones and overlords from the list of units to defend
        defend_units = [unit for unit in self.units if unit.type_id not in (UnitTypeId.DRONE, UnitTypeId.OVERLORD)]

        enemy_units_near_defend_location = [unit for unit in self.enemy_units
                                            if unit.position.distance_to(self.defend_location) < 10]

        if enemy_units_near_defend_location:
            for unit in defend_units:
                self.do(unit.attack(self.defend_location))
        else:
            for unit in defend_units:
                self.do(unit.attack(self.start_location))

    async def attack(self):
        if self.attack_location is None:
            self.attack_location = self.enemy_start_locations[0].position

        # Filter out drones and overlords from the list of units to attack
        attack_units = [unit for unit in self.units if unit.type_id not in (UnitTypeId.DRONE, UnitTypeId.OVERLORD)]

        for unit in attack_units:
            self.do(unit.attack(self.attack_location))

    async def calculate_reward(self):
        # Placeholder reward function; adjust according to your micro strategy
        return 0

    async def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    async def replay(self):
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

    def save_model(self):
        self.model.save("micro_model.h5")
        print("Micro model saved!")

    def load_model(self):
        if os.path.exists("micro_model.h5"):
            self.model = tf.keras.models.load_model("micro_model.h5")
            self.target_model = self.create_micro_model(input_shape=60, num_actions=4)
            self.target_model.set_weights(self.model.get_weights())
            print("Micro model loaded!")
        else:
            print("No pre-trained micro model found, starting from scratch.")

    async def on_step(self, iteration: int):
        state = await self.get_micro_state()
        action = self.choose_action(state)
        await self.execute_action(action)

        # Perform learning updates
        reward = await self.calculate_reward()
        next_state = await self.get_micro_state()
        self.store_experience(state, action, reward, next_state)
        self.replay()
        self.update_target_model(iteration)

def run_multiple_games(num_games):
    for _ in range(num_games):
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Zerg, MicroBot()),
            Computer(Race.Terran, Difficulty.Easy)],
            realtime=True)  # Set to False for non-real-time mode

if __name__ == "__main__":
    run_multiple_games(3)
