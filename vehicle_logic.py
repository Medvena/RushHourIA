# vehicle_logic.py
import copy
from typing import List, Tuple, Dict, Optional
import math

# --- Constantes du Jeu ---
GRID_SIZE = 6
EXIT_ROW = 2  # La voiture rouge sort par la ligne 2 (indices 0 à 5)
EXIT_COL = 5  # La sortie est à la colonne 5
PER_SQ = 80  # Taille d'une case en pixels


class Vehicle:
    """Représente un seul véhicule sur la grille (état pur)."""

    def __init__(self, id: str, orientation: str, size: int, row: int, col: int):
        self.id = id
        self.orientation = orientation
        self.size = size
        self.row = row  # Ligne de la tête (coordonnée GRILLE)
        self.col = col  # Colonne de la tête (coordonnée GRILLE)

    def __repr__(self):
        return f"V({self.id}, {self.orientation}, L{self.row}C{self.col})"

    def __eq__(self, other):
        if not isinstance(other, Vehicle): return NotImplemented
        return self.id == other.id and self.row == other.row and self.col == other.col

    def __hash__(self):
        return hash((self.id, self.row, self.col))


class BoardState:
    """Représente l'état complet du plateau logique."""

    def __init__(self, vehicles: List[Vehicle]):
        self.vehicles: Dict[str, Vehicle] = {v.id: v for v in vehicles}
        self.grid = self._update_grid_matrix()

    def _update_grid_matrix(self) -> List[List[Optional[str]]]:
        """Construit la matrice 6x6 pour la vérification rapide des collisions."""
        grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        for vehicle in self.vehicles.values():
            # Vérification des limites ajoutée pour plus de robustesse
            if vehicle.orientation == 'h':
                for i in range(vehicle.size):
                    if 0 <= vehicle.row < GRID_SIZE and 0 <= vehicle.col + i < GRID_SIZE:
                        grid[vehicle.row][vehicle.col + i] = vehicle.id
            else:  # orientation == 'v'
                for i in range(vehicle.size):
                    if 0 <= vehicle.row + i < GRID_SIZE and 0 <= vehicle.col < GRID_SIZE:
                        grid[vehicle.row + i][vehicle.col] = vehicle.id
        return grid

    def is_solved(self) -> bool:
        """Vérifie si la voiture rouge ('X') est en position de victoire."""
        red_car = self.vehicles.get('X')
        if red_car is None: return False
        # Le nez de la voiture est à col 4, l'extrémité à col 5 (la sortie)
        return red_car.row == EXIT_ROW and (red_car.col + red_car.size) > EXIT_COL
        # Le critère (red_car.col + red_car.size) == (EXIT_COL + 1) est souvent utilisé
        # pour s'assurer que l'extrémité a dépassé la sortie.

    def is_move_valid(self, v_id: str, delta: int) -> bool:
        """Vérifie si le véhicule peut se déplacer de 'delta' cases (collision et limites)."""
        vehicle = self.vehicles.get(v_id)
        if not vehicle or delta == 0: return False

        temp_row, temp_col = vehicle.row, vehicle.col
        step = int(math.copysign(1, delta))

        for _ in range(abs(delta)):
            if vehicle.orientation == 'h':
                temp_col += step
                target_col = temp_col + vehicle.size - 1 if step > 0 else temp_col
                target_row = temp_row
            else:  # 'v'
                temp_row += step
                target_row = temp_row + vehicle.size - 1 if step > 0 else temp_row
                target_col = temp_col

            # 1. Vérification des limites du plateau
            if not (0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE):
                # La voiture rouge peut dépasser la colonne 5 à sa ligne (EXIT_ROW)
                if v_id == 'X' and target_row == EXIT_ROW and target_col > EXIT_COL:
                    continue  # Continue si elle sort légalement
                return False

            # 2. Vérification de la collision : la nouvelle case doit être vide ou c'est la sortie pour X.
            if 0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE:
                if self.grid[target_row][target_col] is not None:
                    # Si la case n'est pas vide et n'est pas la sortie pour la voiture X
                    if self.grid[target_row][target_col] != v_id:  # N'est pas la voiture elle-même
                        return False

        return True