from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps
from sc2.data import Race, Difficulty

def run_game_with_swapper_bot():
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Zerg, SwapperBot()),
        Computer(Race.Terran, Difficulty.Easy)],
        realtime=True)  # Set to False for non-real-time mode

if __name__ == "__main__":
    run_game_with_swapper_bot()