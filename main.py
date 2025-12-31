# main.py
from vehicle import Vehicle
from config import EXIT_ROW
from rush_hour_gui import RushHourGUI
from levels import load_level
import sys


def load_game_from_file(filename: str) -> list[Vehicle]:
    """Charge l'état initial des véhicules depuis un fichier texte."""
    vehicles = []
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            # Pour l'ID, on utilise X pour la rouge (si en Ligne 2, Colonne 0) et A, B, C... pour les autres.
            current_id_char = ord('A')
            for line in lines:
                line = line.strip()
                if line.startswith('#') or not line: continue  # Ignorer les commentaires et lignes vides

                parts = line.split(', ')
                if len(parts) == 4:
                    orientation, size_str, row_str, col_str = parts
                    size, row, col = int(size_str), int(row_str), int(col_str)

                    if orientation == 'h' and row == EXIT_ROW and col == 0:
                        id = 'X'  # ID de la voiture rouge
                    else:
                        id = chr(current_id_char)
                        current_id_char += 1

                    vehicles.append(Vehicle(id, orientation, size, row, col))
    except FileNotFoundError:
        print(f"Erreur: Fichier de niveau {filename} non trouvé.")
        sys.exit()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de niveau: {e}")
        sys.exit()

    return vehicles


if __name__ == "__main__":
    try:
        vehicles = load_level(1)  # level 1..40
    except Exception as e:
        print(f"Level load error: {e}")
        sys.exit(1)

    game = RushHourGUI(vehicles)
    game.run()