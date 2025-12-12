# board.py

import copy
import math
from typing import List, Tuple, Dict, Optional

# Imports des modules du projet
from vehicle import Vehicle
from config import GRID_SIZE, EXIT_ROW, EXIT_COL, RED_CAR_ID


class BoardState:
    """Représente l'état complet du plateau logique."""

    def __init__(self, vehicles: List[Vehicle]):
        # Stocke les véhicules par leur ID pour un accès rapide
        self.vehicles: Dict[str, Vehicle] = {v.id: v for v in vehicles}
        # Matrice de la grille, mise à jour à l'initialisation
        self.grid = self._update_grid_matrix()

    def __eq__(self, other):
        """Vérifie si deux états de plateau sont identiques (utile pour l'IA)."""
        if not isinstance(other, BoardState): return NotImplemented
        return self.vehicles == other.vehicles

    def __hash__(self):
        """Calcule un hachage unique pour l'état du plateau (utile pour les ensembles de l'IA)."""
        # Hache le tuple des véhicules pour obtenir un identifiant d'état unique
        return hash(tuple(sorted(self.vehicles.items())))

    def _update_grid_matrix(self) -> List[List[Optional[str]]]:
        """Construit la matrice GRID_SIZE x GRID_SIZE pour la vérification rapide des collisions."""
        grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        
        for vehicle in self.vehicles.values():
            # Remplit les cases occupées par le véhicule
            if vehicle.orientation == 'h':
                for i in range(vehicle.size):
                    r, c = vehicle.row, vehicle.col + i
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        grid[r][c] = vehicle.id
            else: # orientation == 'v'
                for i in range(vehicle.size):
                    r, c = vehicle.row + i, vehicle.col
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        grid[r][c] = vehicle.id
                        
        return grid

    def is_solved(self) -> bool:
        """Vérifie si la voiture rouge (RED_CAR_ID) est en position de victoire."""
        red_car = self.vehicles.get(RED_CAR_ID)
        if red_car is None: return False
        
        # Vérifie si la voiture est sur la ligne de sortie et si son extrémité a dépassé la sortie.
        return red_car.row == EXIT_ROW and (red_car.col + red_car.size) > EXIT_COL

    def get_next_state(self, v_id: str, delta: int) -> Optional['BoardState']:
        """
        Applique un mouvement valide et retourne un nouvel état de plateau.
        Retourne None si le mouvement est invalide.
        """
        if not self.is_move_valid(v_id, delta):
            return None
        
        # 1. Créer une copie profonde de l'état actuel (important pour l'IA)
        new_vehicles = copy.deepcopy(list(self.vehicles.values()))
        new_board = BoardState(new_vehicles)
        
        # 2. Mettre à jour la position du véhicule dans la copie
        vehicle_to_move = new_board.vehicles[v_id]
        
        if vehicle_to_move.orientation == 'h':
            vehicle_to_move.col += delta
        else:
            vehicle_to_move.row += delta

        # 3. Recalculer la matrice de grille pour le nouvel état
        new_board.grid = new_board._update_grid_matrix()
        
        return new_board

    def is_move_valid(self, v_id: str, delta: int) -> bool:
        """Vérifie si le véhicule peut se déplacer de 'delta' cases (collision et limites)."""
        vehicle = self.vehicles.get(v_id)
        if not vehicle or delta == 0: return False

        temp_row, temp_col = vehicle.row, vehicle.col
        step = int(math.copysign(1, delta)) # +1 ou -1
        
        # Itérer sur chaque case du mouvement
        for _ in range(abs(delta)):
            
            # Déterminer la case "à l'avant" (à l'extrémité dans le sens du mouvement)
            if vehicle.orientation == 'h':
                # La case à vérifier est l'extrémité du véhicule, après le pas
                target_col = temp_col + vehicle.size if step > 0 else temp_col - 1
                target_row = temp_row
                temp_col += step
            else: # 'v'
                target_row = temp_row + vehicle.size if step > 0 else temp_row - 1
                target_col = temp_col
                temp_row += step

            # 1. Vérification des limites du plateau (sauf pour la sortie de la voiture X)
            if not (0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE):
                # Cas spécial : la voiture rouge (X) sort par la droite
                if v_id == RED_CAR_ID and target_row == EXIT_ROW and target_col == GRID_SIZE:
                    continue # Le mouvement est permis car c'est la sortie
                return False # Limite dépassée, mouvement invalide

            # 2. Vérification de la collision (si la case est dans la grille)
            if 0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE:
                if self.grid[target_row][target_col] is not None:
                    # La case est occupée par une autre voiture
                    return False
                    
        # Si toutes les cases traversées sont vides (ou la sortie pour X), le mouvement est valide
        return True