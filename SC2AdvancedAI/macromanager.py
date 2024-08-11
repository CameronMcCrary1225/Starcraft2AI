# macromanager.py
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

class MacroManager:
    def __init__(self, bot_ai: BotAI):
        self.bot_ai = bot_ai

    async def queen_inject(self):
        queens = self.bot_ai.units(UnitTypeId.QUEEN)
        hatcheries = self.bot_ai.structures(UnitTypeId.HATCHERY)

        for queen in queens:
            if hatcheries:
                target_hatchery = hatcheries.closest_to(queen)
                if queen.energy >= 25:
                    self.bot_ai.do(queen(AbilityId.EFFECT_INJECTLARVA, target_hatchery))

    '''async def redistribute_workers(self):
        for base in self.bot_ai.structures(UnitTypeId.HATCHERY):
            workers_around_base = self.bot_ai.workers.closer_than(20, base.position)
            worker_count = len(workers_around_base)
            
            
            if worker_count > 16:
                excess_workers = workers_around_base[16:]

                potential_new_bases = self.bot_ai.structures(UnitTypeId.HATCHERY).filter(
                    lambda b: len(self.bot_ai.workers.closer_than(20, b.position)) < 16
                )
                
                potential_new_bases = sorted(potential_new_bases, key=lambda b: len(self.bot_ai.workers.closer_than(20, b.position)))
                
                for worker in excess_workers:
                    if potential_new_bases:
                        new_base = potential_new_bases[0]
                        self.bot_ai.do(worker.move(new_base.position))
                        potential_new_bases = self.bot_ai.structures(UnitTypeId.HATCHERY).filter(
                            lambda b: len(self.bot_ai.workers.closer_than(20, b.position)) < 16
                        )
                        potential_new_bases = sorted(potential_new_bases, key=lambda b: len(self.bot_ai.workers.closer_than(20, b.position)))
                    break
                break

        idle_workers = self.bot_ai.workers.idle
        for worker in idle_workers:
            if worker.is_idle:
                potential_new_bases = self.bot_ai.structures(UnitTypeId.HATCHERY).filter(
                    lambda b: len(self.bot_ai.workers.closer_than(20, b.position)) < 16
                )
                
                potential_new_bases = sorted(potential_new_bases, key=lambda b: len(self.bot_ai.workers.closer_than(20, b.position)))
                
                if potential_new_bases:
                    new_base = potential_new_bases[0]
                    self.bot_ai.do(worker.move(new_base.position))
                    potential_new_bases = self.bot_ai.structures(UnitTypeId.HATCHERY).filter(
                        lambda b: len(self.bot_ai.workers.closer_than(20, b.position)) < 16
                    )
                    potential_new_bases = sorted(potential_new_bases, key=lambda b: len(self.bot_ai.workers.closer_than(20, b.position)))

    async def assign_new_workers(self):
        for base in self.bot_ai.structures(UnitTypeId.HATCHERY):
            if len(self.bot_ai.workers.closer_than(16, base.position)) < 16:
                mineral_patches = self.bot_ai.mineral_field.closer_than(20, base.position)
                for patch in mineral_patches:
                    if len(self.bot_ai.workers.closer_than(2, patch.position)) == 0:
                        idle_workers = self.bot_ai.workers.idle
                        for worker in idle_workers:
                            self.bot_ai.do(worker.gather(patch))
                            break'''
    
    async def run_periodic_tasks(self, iteration: int):
        if iteration % 20 == 0:
            await self.bot_ai.distribute_workers()
            await self.queen_inject()

