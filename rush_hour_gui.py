# rush_hour_gui.py

import pygame
import sys
import math # Nécessaire pour math.copysign dans _find_max_valid_delta
from tkinter import messagebox, Tk
from typing import List

# Import des modules du projet
from vehicle import Vehicle
from board import BoardState
from gui_vehicle import GUIVehicle 
from config import PER_SQ, GRID_SIZE, EXIT_ROW, EXIT_COL


# Masquer la fenêtre Tkinter principale pour les pop-ups
Tk().wm_withdraw()


class RushHourGUI:
    """Contrôleur principal du jeu et de l'interface."""

    def __init__(self, initial_vehicles: List[Vehicle]):

        pygame.init()
        self.board_state = BoardState(initial_vehicles) 
        self.g_vehicles = self._create_gui_vehicles()
        self.turns = 0
        self.selected_car_id: str | None = None
        self.inGame = True

        surfaceSize = GRID_SIZE * PER_SQ
        self.surface = pygame.display.set_mode((surfaceSize, surfaceSize))
        pygame.display.set_caption("Rush Hour - Projet IA")
        self.ev = None # L'événement actuel, mis à jour dans run()

    def _create_gui_vehicles(self):
        """Crée les objets GUIVehicle en mappant l'état logique."""
        g_vehicles = {}
        for v_id, vehicle_logic in self.board_state.vehicles.items():
            g_vehicles[v_id] = GUIVehicle(vehicle_logic)
        return g_vehicles

    def _click_object(self):
        """Sélectionne la voiture à l'endroit du clic et enregistre les positions de départ."""
        for g_car in self.g_vehicles.values():
            if g_car.rect.collidepoint(self.ev.pos):
                g_car.rectDrag = True
                self.selected_car_id = g_car.id

                mouseX, mouseY = self.ev.pos
                g_car.offsetX = g_car.rect.x - mouseX
                g_car.offsetY = g_car.rect.y - mouseY

                # Sauvegarde de la position logique et PIXEL de départ pour la comparaison et l'annulation
                g_car.initial_drag_pos = (g_car.logic.row, g_car.logic.col)
                g_car.initial_pixel_pos = (g_car.rect.x, g_car.rect.y)
                break
    
    def _find_max_valid_delta(self, v_id: str, requested_delta: int) -> int:
        """
        Calcule le nombre maximum de cases qu'une voiture peut parcourir
        dans la direction demandée (basé sur la logique BoardState).
        """
        if requested_delta == 0:
            return 0
            
        step = int(math.copysign(1, requested_delta)) # +1 ou -1
        max_possible_move = 0
        
        # Simuler le mouvement case par case
        for delta in range(1, abs(requested_delta) + 1):
            if self.board_state.is_move_valid(v_id, delta * step):
                max_possible_move = delta * step
            else:
                # La première collision a été trouvée. Arrêter.
                break
                
        return max_possible_move


    def _object_mid_air(self):
        """Déplace l'objet graphique en respectant l'axe et les collisions (LOGIQUE DE CONTRAINTE)."""
        if not self.selected_car_id: return

        g_car = self.g_vehicles[self.selected_car_id]
        
        if g_car.rectDrag:
            mouseX, mouseY = self.ev.pos
            logic_car = g_car.logic
            
            # Nouvelle position pixel SANS contrainte
            new_x_unconstrained = mouseX + g_car.offsetX
            new_y_unconstrained = mouseY + g_car.offsetY
            
            # Calculer le delta (nombre de cases) demandé par l'utilisateur
            if logic_car.orientation == 'h':
                delta_pixel = new_x_unconstrained - g_car.initial_pixel_pos[0]
                delta_logic = int(round(delta_pixel / PER_SQ)) 
            else: # 'v'
                delta_pixel = new_y_unconstrained - g_car.initial_pixel_pos[1]
                delta_logic = int(round(delta_pixel / PER_SQ))

            # Trouver le nombre maximum de cases de déplacement VALIDES
            max_valid_delta = self._find_max_valid_delta(g_car.id, delta_logic)

            # Appliquer la contrainte d'axe et de collision
            if logic_car.orientation == 'h':
                # Limiter la nouvelle position pixel à la position maximale autorisée
                g_car.rect.x = g_car.initial_pixel_pos[0] + (max_valid_delta * PER_SQ)
                # Contrainte d'axe Y: La voiture ne peut pas bouger verticalement
                g_car.rect.y = g_car.initial_pixel_pos[1]
            else: # 'v'
                # Contrainte d'axe X: La voiture ne peut pas bouger horizontalement
                g_car.rect.x = g_car.initial_pixel_pos[0]
                # Limiter la nouvelle position pixel à la position maximale autorisée
                g_car.rect.y = g_car.initial_pixel_pos[1] + (max_valid_delta * PER_SQ)

    def _unclick_object(self):
        """Relâchement : vérifie le mouvement, l'applique à la logique ou annule."""
        if not self.selected_car_id: return
        g_car = self.g_vehicles[self.selected_car_id]
        g_car.rectDrag = False

        # La nouvelle position est déjà alignée sur la grille grâce à _object_mid_air
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

        # Le mouvement est valide si delta != 0 (on a bougé) et qu'il n'y a pas eu de dépassement
        if delta != 0:
            # Mouvement valide : Application des coordonnées à l'objet LOGIQUE
            if logic_car.orientation == 'h':
                logic_car.col = target_col
            else:
                logic_car.row = target_row

            # Mise à jour de la matrice de grille après le mouvement logique
            self.board_state.grid = self.board_state._update_grid_matrix()
            self.turns += 1

            # Synchronisation graphique (ceci est purement esthétique ici, mais bonne pratique)
            g_car.update_position_from_logic()
            
        else:
            # Mouvement invalide ou delta == 0 : Annulation graphique
            g_car.rect.x = old_col * PER_SQ
            g_car.rect.y = old_row * PER_SQ
            if delta != 0: messagebox.showwarning('Erreur', 'Mouvement invalide (Collision détectée au moment du glisser).')
            
        self.selected_car_id = None

    def _draw_board(self):
        """Dessine le plateau, les cases de grille et les véhicules."""
        self.surface.fill((255, 255, 255))

        # Dessiner la grille (lignes)
        for i in range(GRID_SIZE):
            pygame.draw.line(self.surface, (0, 0, 0), (0, i * PER_SQ), (GRID_SIZE * PER_SQ, i * PER_SQ), 1)
            pygame.draw.line(self.surface, (0, 0, 0), (i * PER_SQ, 0), (i * PER_SQ, GRID_SIZE * PER_SQ), 1)

        # Dessiner la sortie (carré vert)
        exit_rect = pygame.Rect(EXIT_COL * PER_SQ, EXIT_ROW * PER_SQ, PER_SQ, PER_SQ)
        pygame.draw.rect(self.surface, (0, 200, 0), exit_rect, 0)
        pygame.draw.rect(self.surface, (0, 0, 0), exit_rect, 2)

        # Dessiner les voitures en utilisant la méthode draw() de GUIVehicle
        for g_car in self.g_vehicles.values():
            g_car.draw(self.surface)


    def _check_game_over(self) -> bool:
        """Vérifie la condition de victoire."""
        if self.board_state.is_solved():
            messagebox.showinfo('Félicitations !', f'Vous avez terminé en {self.turns} mouvements !')
            self.inGame = False
            return True
        return False

    def run(self) -> bool:
        """Lance la boucle principale de Pygame et gère les événements."""
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