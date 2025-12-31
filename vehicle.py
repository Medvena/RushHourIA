# vehicle.py

class Vehicle:
    """Represents a single vehicle on the grid (pure state)."""

    def __init__(self, id: str, orientation: str, size: int, row: int, col: int):
        self.id = id
        self.orientation = orientation
        self.size = size
        self.row = row  # Head row (grid coordinate)
        self.col = col  # Head column (grid coordinate)

    def __repr__(self):
        return f"V({self.id}, {self.orientation}, R{self.row}C{self.col})"

    def __eq__(self, other):
        if not isinstance(other, Vehicle):
            return NotImplemented
        # Equality is defined by ID and position
        return self.id == other.id and self.row == other.row and self.col == other.col

    def __hash__(self):
        # Hash is based on state (ID and position)
        return hash((self.id, self.row, self.col))
