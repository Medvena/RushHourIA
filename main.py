# main.py
import sys

from levels import load_level
from rush_hour_gui import RushHourGUI


def main() -> int:
    level = 1

    while True:
        try:
            vehicles = load_level(level)
        except Exception:
            print("Tous les niveaux sont terminés.")
            return 0

        game = RushHourGUI(vehicles)
        won = game.run()  # must return True if solved, False if user quit

        if not won:
            return 0

        level += 1


if __name__ == "__main__":
    sys.exit(main())
