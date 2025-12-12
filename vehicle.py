# vehicle.py

from typing import List, Tuple, Dict, Optional

class Vehicle:
    """Représente un seul véhicule sur la grille (état pur)."""

    def __init__(self, id: str, orientation: str, size: int, row: int, col: int):
        self.id = id
        self.orientation = orientation
        self.size = size
        self.row = row # Ligne de la tête (coordonnée GRILLE)
        self.col = col # Colonne de la tête (coordonnée GRILLE)

    def __repr__(self):
        return f"V({self.id}, {self.orientation}, L{self.row}C{self.col})"

    def __eq__(self, other):
        if not isinstance(other, Vehicle): return NotImplemented
        # L'égalité est définie par la position et l'ID
        return self.id == other.id and self.row == other.row and self.col == other.col

    def __hash__(self):
        # Le hash est basé sur l'état (ID et position)
        return hash((self.id, self.row, self.col))