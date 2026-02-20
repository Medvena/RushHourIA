import pygame
import math
import time
import os
from typing import List

# Project imports
from vehicle import Vehicle
from board import BoardState
from gui_vehicle import GUIVehicle
from config import GRID_SIZE, EXIT_ROW, EXIT_COL

class RushHourGUI:
    """
    Main controller with an enlarged background for better visual padding.
    """

    def __init__(self, initial_vehicles: List[Vehicle]):
        pygame.init()

        self.board_state = BoardState(initial_vehicles)
        
        # --- CONFIGURATION DU DESSIN ---
        self.window_size = 1000
        self.custom_per_sq = 105  
        self.offset_x = 185       
        self.offset_y = 185       

        # On agrandit l'image de 5% (facteur 1.05) pour que les bordures s'écartent
        self.bg_scale_factor = 1.04
        
        try:
            raw_bg = pygame.image.load(os.path.join("images", "parking.png")).convert()
            # On redimensionne l'image pour qu'elle soit un peu plus grande que la fenêtre
            new_bg_size = int(self.window_size * self.bg_scale_factor)
            self.background = pygame.transform.smoothscale(raw_bg, (new_bg_size, new_bg_size))
            
            # On calcule le centrage pour que l'agrandissement se fasse par l'extérieur
            self.bg_pos_offset = (self.window_size - new_bg_size) // 2
        except pygame.error:
            self.background = pygame.Surface((1000, 1000))
            self.background.fill((200, 200, 200))
            self.bg_pos_offset = 0

        self.g_vehicles = self._create_gui_vehicles()
        self.surface = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Rush Hour - Jeu")
        self.ev = None
        self.selected_car_id = None
        self.inGame = True
        self.turns = 0

    def _create_gui_vehicles(self):
        g_vehicles = {}
        for v_id, vehicle_logic in self.board_state.vehicles.items():
            gv = GUIVehicle(vehicle_logic)
            self._sync_visual_with_offset(gv)
            g_vehicles[v_id] = gv
        return g_vehicles

    def _sync_visual_with_offset(self, g_car: GUIVehicle):
        """Syncs the vehicle with the 105px grid and 185px offset."""
        w = self.custom_per_sq * g_car.logic.size if g_car.logic.orientation == "h" else self.custom_per_sq
        h = self.custom_per_sq if g_car.logic.orientation == "h" else self.custom_per_sq * g_car.logic.size
        
        g_car.image = pygame.transform.smoothscale(g_car.image, (w, h))
        g_car.rect = pygame.Rect(
            g_car.logic.col * self.custom_per_sq + self.offset_x,
            g_car.logic.row * self.custom_per_sq + self.offset_y,
            w, h
        )

    def solve_step_with_ai(self, agent):
        """Move one step and re-sync to avoid top-left jumps."""
        from solver_ia import state_to_tensor
        state_tensor = state_to_tensor(self.board_state)
        action_idx = agent.act(state_tensor)
        v_id, delta = agent.decode_action(action_idx)

        if self.board_state.is_move_valid(v_id, delta):
            self.board_state = self.board_state.get_next_state(v_id, delta)
            for vid, g_car in self.g_vehicles.items():
                g_car.logic = self.board_state.vehicles[vid]
                self._sync_visual_with_offset(g_car)
            self.turns += 1
            return True
        return False

    def _draw_board(self):
        # On dessine le fond avec le léger décalage négatif pour le centrer
        self.surface.blit(self.background, (self.bg_pos_offset, self.bg_pos_offset))
        
        for g_car in self.g_vehicles.values():
            g_car.draw(self.surface)

    def _click_object(self):
        for g_car in self.g_vehicles.values():
            if g_car.rect.collidepoint(self.ev.pos):
                g_car.rectDrag = True
                self.selected_car_id = g_car.id
                mouseX, mouseY = self.ev.pos
                g_car.offsetX = g_car.rect.x - mouseX
                g_car.offsetY = g_car.rect.y - mouseY
                g_car.initial_pixel_pos = (g_car.rect.x, g_car.rect.y)
                break

    def _object_mid_air(self):
        if not self.selected_car_id: return
        g_car = self.g_vehicles[self.selected_car_id]
        if g_car.rectDrag:
            mouseX, mouseY = self.ev.pos
            if g_car.logic.orientation == 'h':
                delta_px = (mouseX + g_car.offsetX) - g_car.initial_pixel_pos[0]
                delta_logic = int(round(delta_px / self.custom_per_sq))
                mvd = self._find_max_valid_delta(g_car.id, delta_logic)
                g_car.rect.x = g_car.initial_pixel_pos[0] + (mvd * self.custom_per_sq)
            else:
                delta_px = (mouseY + g_car.offsetY) - g_car.initial_pixel_pos[1]
                delta_logic = int(round(delta_px / self.custom_per_sq))
                mvd = self._find_max_valid_delta(g_car.id, delta_logic)
                g_car.rect.y = g_car.initial_pixel_pos[1] + (mvd * self.custom_per_sq)

    def _unclick_object(self):
        if not self.selected_car_id: return
        g_car = self.g_vehicles[self.selected_car_id]
        g_car.rectDrag = False
        t_col = int(round((g_car.rect.x - self.offset_x) / self.custom_per_sq))
        t_row = int(round((g_car.rect.y - self.offset_y) / self.custom_per_sq))
        
        d = (t_col - g_car.logic.col) if g_car.logic.orientation == 'h' else (t_row - g_car.logic.row)
        if d != 0 and self.board_state.is_move_valid(g_car.id, d):
            if g_car.logic.orientation == 'h': g_car.logic.col = t_col
            else: g_car.logic.row = t_row
            self.board_state.grid = self.board_state._update_grid_matrix()
            self.turns += 1
        self._sync_visual_with_offset(g_car)
        self.selected_car_id = None

    def _find_max_valid_delta(self, v_id, requested_delta):
        if requested_delta == 0: return 0
        step = int(math.copysign(1, requested_delta))
        m_move = 0
        for d in range(1, abs(requested_delta) + 1):
            if self.board_state.is_move_valid(v_id, d * step): m_move = d * step
            else: break
        return m_move

    def run(self) -> bool:
        while self.inGame:
            self.ev = pygame.event.poll()
            if self.ev.type == pygame.QUIT: return False
            elif self.ev.type == pygame.MOUSEBUTTONDOWN: self._click_object()
            elif self.ev.type == pygame.MOUSEBUTTONUP: self._unclick_object()
            elif self.ev.type == pygame.MOUSEMOTION: self._object_mid_air()
            self._draw_board()
            pygame.display.flip()
            if self.board_state.is_solved():
                time.sleep(1)
                break
        return True