# board.py

import copy
import math
from typing import Dict, List, Optional

# Project imports
from vehicle import Vehicle
from config import GRID_SIZE, EXIT_ROW, EXIT_COL, RED_CAR_ID


class BoardState:
    """Represents the full logical state of the board."""

    def __init__(self, vehicles: List[Vehicle]):
        # Store vehicles by ID for fast access
        self.vehicles: Dict[str, Vehicle] = {v.id: v for v in vehicles}
        # Grid occupancy matrix, computed at initialization
        self.grid = self._update_grid_matrix()

    def __eq__(self, other):
        """Check whether two board states are identical (useful for AI)."""
        if not isinstance(other, BoardState):
            return NotImplemented
        return self.vehicles == other.vehicles

    def __hash__(self):
        """Compute a unique hash for this board state (useful for AI sets/dicts)."""
        return hash(tuple(sorted(self.vehicles.items())))

    def _update_grid_matrix(self) -> List[List[Optional[str]]]:
        """Build a GRID_SIZE x GRID_SIZE occupancy matrix for fast collision checks."""
        grid: List[List[Optional[str]]] = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]

        for vehicle in self.vehicles.values():
            # Fill the cells occupied by the vehicle
            if vehicle.orientation == "h":
                for i in range(vehicle.size):
                    r, c = vehicle.row, vehicle.col + i
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        grid[r][c] = vehicle.id
            else:  # vehicle.orientation == "v"
                for i in range(vehicle.size):
                    r, c = vehicle.row + i, vehicle.col
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        grid[r][c] = vehicle.id

        return grid

    def is_solved(self) -> bool:
        """Return True if the red car (RED_CAR_ID) has reached the exit."""
        red_car = self.vehicles.get(RED_CAR_ID)
        if red_car is None:
            return False

        # The red car is solved if it is on the exit row and its right end crosses the exit column.
        return red_car.row == EXIT_ROW and (red_car.col + red_car.size) > EXIT_COL

    def get_next_state(self, v_id: str, delta: int) -> Optional["BoardState"]:
        """
        Apply a valid move and return a new board state.
        Returns None if the move is invalid.
        """
        if not self.is_move_valid(v_id, delta):
            return None

        # 1) Deep copy current vehicles (important for AI search)
        new_vehicles = copy.deepcopy(list(self.vehicles.values()))
        new_board = BoardState(new_vehicles)

        # 2) Update the moved vehicle position in the copied state
        vehicle_to_move = new_board.vehicles[v_id]
        if vehicle_to_move.orientation == "h":
            vehicle_to_move.col += delta
        else:
            vehicle_to_move.row += delta

        # 3) Recompute grid occupancy
        new_board.grid = new_board._update_grid_matrix()
        return new_board

    def is_move_valid(self, v_id: str, delta: int) -> bool:
        """Check whether a vehicle can move by 'delta' cells (bounds + collisions)."""
        vehicle = self.vehicles.get(v_id)
        if vehicle is None or delta == 0:
            return False

        temp_row, temp_col = vehicle.row, vehicle.col
        step = int(math.copysign(1, delta))  # +1 or -1

        # Iterate through each cell step of the movement
        for _ in range(abs(delta)):
            # Determine the "front" cell to test (the leading edge in movement direction)
            if vehicle.orientation == "h":
                target_col = temp_col + vehicle.size if step > 0 else temp_col - 1
                target_row = temp_row
                temp_col += step
            else:  # "v"
                target_row = temp_row + vehicle.size if step > 0 else temp_row - 1
                target_col = temp_col
                temp_row += step

            # 1) Board bounds check (special-case: red car exits to the right)
            if not (0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE):
                if v_id == RED_CAR_ID and target_row == EXIT_ROW and target_col == GRID_SIZE:
                    # Allow the red car to move through the exit
                    continue
                return False

            # 2) Collision check (only for in-grid cells)
            if self.grid[target_row][target_col] is not None:
                return False

        return True
