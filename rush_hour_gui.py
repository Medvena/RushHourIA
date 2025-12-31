# rush_hour_gui.py

import pygame
import math
from tkinter import messagebox, Tk
from typing import List

# Project imports
from vehicle import Vehicle
from board import BoardState
from gui_vehicle import GUIVehicle
from config import PER_SQ, GRID_SIZE, EXIT_ROW, EXIT_COL

# Hide the main Tkinter window
Tk().wm_withdraw()

class RushHourGUI:
    """Main controller for the Rush Hour game GUI."""
    
    def __init__(self, initial_vehicles: List[Vehicle]):
        pygame.init()

        self.board_state = BoardState(initial_vehicles)
        self.g_vehicles = self._create_gui_vehicles()

        self.turns = 0
        self.selected_car_id: str | None = None
        self.inGame = True

        surface_size = GRID_SIZE * PER_SQ
        self.surface = pygame.display.set_mode((surface_size, surface_size))
        pygame.display.set_caption("Rush Hour - AI Project")

        # Current pygame event (updated in run loop)
        self.ev = None

    def _create_gui_vehicles(self):
        """Create GUIVehicle objects mapped to logical Vehicle instances."""
        g_vehicles = {}
        for v_id, vehicle_logic in self.board_state.vehicles.items():
            g_vehicles[v_id] = GUIVehicle(vehicle_logic)
        return g_vehicles

    def _click_object(self):
        """Select a vehicle under the mouse cursor and store initial positions."""
        for g_car in self.g_vehicles.values():
            if g_car.rect.collidepoint(self.ev.pos):
                g_car.rectDrag = True
                self.selected_car_id = g_car.id

                mouseX, mouseY = self.ev.pos
                g_car.offsetX = g_car.rect.x - mouseX
                g_car.offsetY = g_car.rect.y - mouseY

                # Store initial logical and pixel positions
                g_car.initial_drag_pos = (g_car.logic.row, g_car.logic.col)
                g_car.initial_pixel_pos = (g_car.rect.x, g_car.rect.y)
                break

    def _find_max_valid_delta(self, v_id: str, requested_delta: int) -> int:
        """
        Compute the maximum number of grid cells a vehicle can move
        in the requested direction based on BoardState logic.
        """
        if requested_delta == 0:
            return 0

        step = int(math.copysign(1, requested_delta))  # +1 or -1
        max_possible_move = 0

        # Simulate movement cell by cell
        for delta in range(1, abs(requested_delta) + 1):
            if self.board_state.is_move_valid(v_id, delta * step):
                max_possible_move = delta * step
            else:
                # First collision encountered, stop
                break

        return max_possible_move

    def _object_mid_air(self):
        """Move the selected vehicle while enforcing axis and collision constraints."""
        if not self.selected_car_id:
            return

        g_car = self.g_vehicles[self.selected_car_id]

        if g_car.rectDrag:
            mouseX, mouseY = self.ev.pos
            logic_car = g_car.logic

            # Unconstrained new pixel position
            new_x_unconstrained = mouseX + g_car.offsetX
            new_y_unconstrained = mouseY + g_car.offsetY

            # Compute requested movement in grid units
            if logic_car.orientation == 'h':
                delta_pixel = new_x_unconstrained - g_car.initial_pixel_pos[0]
                delta_logic = int(round(delta_pixel / PER_SQ))
            else:  # 'v'
                delta_pixel = new_y_unconstrained - g_car.initial_pixel_pos[1]
                delta_logic = int(round(delta_pixel / PER_SQ))

            # Clamp movement to maximum valid delta
            max_valid_delta = self._find_max_valid_delta(g_car.id, delta_logic)

            # Apply axis constraint and collision constraint
            if logic_car.orientation == 'h':
                g_car.rect.x = g_car.initial_pixel_pos[0] + (max_valid_delta * PER_SQ)
                g_car.rect.y = g_car.initial_pixel_pos[1]
            else:  # 'v'
                g_car.rect.x = g_car.initial_pixel_pos[0]
                g_car.rect.y = g_car.initial_pixel_pos[1] + (max_valid_delta * PER_SQ)

    def _unclick_object(self):
        """Release the vehicle and apply or cancel the logical move."""
        if not self.selected_car_id:
            return

        g_car = self.g_vehicles[self.selected_car_id]
        g_car.rectDrag = False

        # Snap pixel position back to grid coordinates
        target_col = int(round(g_car.rect.x / PER_SQ))
        target_row = int(round(g_car.rect.y / PER_SQ))

        old_row, old_col = g_car.initial_drag_pos

        delta_col = target_col - old_col
        delta_row = target_row - old_row

        logic_car = g_car.logic
        delta = 0

        if logic_car.orientation == 'h':
            delta = delta_col
        elif logic_car.orientation == 'v':
            delta = delta_row

        # Valid move if delta != 0
        if delta != 0:
            # Apply logical move
            if logic_car.orientation == 'h':
                logic_car.col = target_col
            else:
                logic_car.row = target_row

            # Update grid state
            self.board_state.grid = self.board_state._update_grid_matrix()
            self.turns += 1

            # Sync graphical position
            g_car.update_position_from_logic()
        else:
            # Cancel move and revert graphical position
            g_car.rect.x = old_col * PER_SQ
            g_car.rect.y = old_row * PER_SQ

        self.selected_car_id = None

    def _draw_board(self):
        """Draw the board grid, exit cell, and all vehicles."""
        self.surface.fill((255, 255, 255))

        # Draw grid lines
        for i in range(GRID_SIZE):
            pygame.draw.line(
                self.surface, (0, 0, 0),
                (0, i * PER_SQ),
                (GRID_SIZE * PER_SQ, i * PER_SQ),
                1
            )
            pygame.draw.line(
                self.surface, (0, 0, 0),
                (i * PER_SQ, 0),
                (i * PER_SQ, GRID_SIZE * PER_SQ),
                1
            )

        # Draw exit cell
        exit_rect = pygame.Rect(
            EXIT_COL * PER_SQ,
            EXIT_ROW * PER_SQ,
            PER_SQ,
            PER_SQ
        )
        pygame.draw.rect(self.surface, (0, 200, 0), exit_rect, 0)
        pygame.draw.rect(self.surface, (0, 0, 0), exit_rect, 2)

        # Draw vehicles
        for g_car in self.g_vehicles.values():
            g_car.draw(self.surface)

    def _check_game_over(self) -> bool:
        """Check victory condition."""
        if self.board_state.is_solved():
            messagebox.showinfo(
                "Congratulations!",
                f"You solved the level in {self.turns} moves!"
            )
            self.inGame = False
            return True
        return False

    def run(self) -> bool:
        """Run the main Pygame loop. Returns True if the level is solved."""
        won = False

        while self.inGame:
            self.ev = pygame.event.poll()

            if self.ev.type == pygame.QUIT:
                self.inGame = False
                break
            elif self.ev.type == pygame.MOUSEBUTTONDOWN:
                self._click_object()
            elif self.ev.type == pygame.MOUSEBUTTONUP:
                self._unclick_object()
            elif self.ev.type == pygame.MOUSEMOTION:
                self._object_mid_air()

            self._draw_board()
            pygame.display.flip()

            if self._check_game_over():
                won = True
                break

        pygame.quit()
        return won
