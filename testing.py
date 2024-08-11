from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.data import Difficulty, Race
class TestBot(BotAI):
    async def on_step(self, iteration: int):
        minerals = self.minerals
        print(f"Minerals: {minerals}")

def run_test():
    # Use TestBot instead of MacroBot to see if minerals is accessible
    from sc2.main import run_game
    from sc2.player import Bot, Computer
    from sc2 import maps
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Zerg, TestBot()),
        Computer(Race.Terran, Difficulty.Easy)],
        realtime=True)

if __name__ == "__main__":
    run_test()