import pygame
import math
import time
from typing import List

# Project imports
from vehicle import Vehicle
from board import BoardState
from gui_vehicle import GUIVehicle
from config import PER_SQ, GRID_SIZE, EXIT_ROW, EXIT_COL


class RushHourGUI:
    """Main controller for the Rush Hour game GUI."""

    def __init__(self, initial_vehicles: List[Vehicle]):
        # On initialise Pygame si ce n'est pas déjà fait
        pygame.init()

        self.board_state = BoardState(initial_vehicles)
        self.g_vehicles = self._create_gui_vehicles()

        self.turns = 0
        self.selected_car_id: str | None = None
        self.inGame = True

        surface_size = GRID_SIZE * PER_SQ

        # On récupère la fenêtre active (celle du menu) ou on en crée une nouvelle
        if pygame.display.get_surface() is None:
            self.surface = pygame.display.set_mode((surface_size, surface_size))
        else:
            self.surface = pygame.display.get_surface()

        pygame.display.set_caption("Rush Hour - Jeu")

        self.ev = None

    def _create_gui_vehicles(self):
        g_vehicles = {}
        for v_id, vehicle_logic in self.board_state.vehicles.items():
            g_vehicles[v_id] = GUIVehicle(vehicle_logic)
        return g_vehicles

    def _click_object(self):
        for g_car in self.g_vehicles.values():
            if g_car.rect.collidepoint(self.ev.pos):
                g_car.rectDrag = True
                self.selected_car_id = g_car.id

                mouseX, mouseY = self.ev.pos
                g_car.offsetX = g_car.rect.x - mouseX
                g_car.offsetY = g_car.rect.y - mouseY

                g_car.initial_drag_pos = (g_car.logic.row, g_car.logic.col)
                g_car.initial_pixel_pos = (g_car.rect.x, g_car.rect.y)
                break

    def _find_max_valid_delta(self, v_id: str, requested_delta: int) -> int:
        if requested_delta == 0:
            return 0

        step = int(math.copysign(1, requested_delta))
        max_possible_move = 0

        for delta in range(1, abs(requested_delta) + 1):
            if self.board_state.is_move_valid(v_id, delta * step):
                max_possible_move = delta * step
            else:
                break

        return max_possible_move

    def _object_mid_air(self):
        if not self.selected_car_id:
            return

        g_car = self.g_vehicles[self.selected_car_id]

        if g_car.rectDrag:
            mouseX, mouseY = self.ev.pos
            logic_car = g_car.logic

            new_x_unconstrained = mouseX + g_car.offsetX
            new_y_unconstrained = mouseY + g_car.offsetY

            if logic_car.orientation == 'h':
                delta_pixel = new_x_unconstrained - g_car.initial_pixel_pos[0]
                delta_logic = int(round(delta_pixel / PER_SQ))
            else:
                delta_pixel = new_y_unconstrained - g_car.initial_pixel_pos[1]
                delta_logic = int(round(delta_pixel / PER_SQ))

            max_valid_delta = self._find_max_valid_delta(g_car.id, delta_logic)

            if logic_car.orientation == 'h':
                g_car.rect.x = g_car.initial_pixel_pos[0] + (max_valid_delta * PER_SQ)
                g_car.rect.y = g_car.initial_pixel_pos[1]
            else:
                g_car.rect.x = g_car.initial_pixel_pos[0]
                g_car.rect.y = g_car.initial_pixel_pos[1] + (max_valid_delta * PER_SQ)

    def _unclick_object(self):
        if not self.selected_car_id:
            return

        g_car = self.g_vehicles[self.selected_car_id]
        g_car.rectDrag = False

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

        if delta != 0:
            if logic_car.orientation == 'h':
                logic_car.col = target_col
            else:
                logic_car.row = target_row

            self.board_state.grid = self.board_state._update_grid_matrix()
            self.turns += 1
            g_car.update_position_from_logic()
        else:
            g_car.rect.x = old_col * PER_SQ
            g_car.rect.y = old_row * PER_SQ

        self.selected_car_id = None

    def _draw_board(self):
        self.surface.fill((255, 255, 255))
        for i in range(GRID_SIZE):
            pygame.draw.line(self.surface, (0, 0, 0), (0, i * PER_SQ), (GRID_SIZE * PER_SQ, i * PER_SQ), 1)
            pygame.draw.line(self.surface, (0, 0, 0), (i * PER_SQ, 0), (i * PER_SQ, GRID_SIZE * PER_SQ), 1)

        exit_rect = pygame.Rect(EXIT_COL * PER_SQ, EXIT_ROW * PER_SQ, PER_SQ, PER_SQ)
        pygame.draw.rect(self.surface, (0, 200, 0), exit_rect, 0)
        pygame.draw.rect(self.surface, (0, 0, 0), exit_rect, 2)

        for g_car in self.g_vehicles.values():
            g_car.draw(self.surface)

    def _check_game_over(self) -> bool:
        if self.board_state.is_solved():
            # Affiche "WIN" dans la console et attend 2 secondes
            print(f"VICTOIRE ! Niveau résolu en {self.turns} coups !")
            pygame.display.set_caption("VICTOIRE !!!")
            self.inGame = False
            return True
        return False

    def run(self) -> bool:
        won = False
        while self.inGame:
            # Important : poll pour ne pas bloquer
            self.ev = pygame.event.poll()

            if self.ev.type == pygame.QUIT:
                self.inGame = False
                # On ne return pas tout de suite, on laisse la boucle finir proprement
                return False

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
                # On attend un peu pour voir le résultat avant de fermer
                time.sleep(2)
                break

        return won