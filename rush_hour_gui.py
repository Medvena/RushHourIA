# rush_hour_gui.py

import pygame
import sys
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
        # L'état logique est conservé ici
        self.board_state = BoardState(initial_vehicles) 
        
        # Les voitures graphiques sont créées à partir de la logique
        self.g_vehicles = self._create_gui_vehicles()
        self.turns = 0
        self.selected_car_id: str | None = None
        self.inGame = True

        surfaceSize = GRID_SIZE * PER_SQ
        self.surface = pygame.display.set_mode((surfaceSize, surfaceSize))
        pygame.display.set_caption("Rush Hour - Projet IA")

    def _create_gui_vehicles(self):
        """Crée les objets GUIVehicle en mappant l'état logique."""
        g_vehicles = {}
        for v_id, vehicle_logic in self.board_state.vehicles.items():
            g_vehicles[v_id] = GUIVehicle(vehicle_logic)
        return g_vehicles

    def _click_object(self):
        """Sélectionne la voiture à l'endroit du clic."""
        for g_car in self.g_vehicles.values():
            if g_car.rect.collidepoint(self.ev.pos):
                g_car.rectDrag = True
                self.selected_car_id = g_car.id

                mouseX, mouseY = self.ev.pos
                g_car.offsetX = g_car.rect.x - mouseX
                g_car.offsetY = g_car.rect.y - mouseY

                # Enregistre la position logique de départ du drag (pour l'annulation)
                g_car.initial_drag_pos = (g_car.logic.row, g_car.logic.col)
                break

    def _object_mid_air(self):
        """Déplace l'objet graphique pendant le glisser-déposer."""
        if self.selected_car_id:
            g_car = self.g_vehicles[self.selected_car_id]
            if g_car.rectDrag:
                mouseX, mouseY = self.ev.pos
                g_car.rect.x = mouseX + g_car.offsetX
                g_car.rect.y = mouseY + g_car.offsetY

    def _unclick_object(self):
        """Relâchement : vérifie le mouvement, l'applique à la logique ou annule."""
        if not self.selected_car_id: return
        g_car = self.g_vehicles[self.selected_car_id]
        g_car.rectDrag = False

        # Calcul de la nouvelle position en coordonnées de GRILLE (arrondi au plus proche)
        target_col = int(round(g_car.rect.x / PER_SQ))
        target_row = int(round(g_car.rect.y / PER_SQ))

        old_row, old_col = g_car.initial_drag_pos
        
        delta_col = target_col - old_col
        delta_row = target_row - old_row

        logic_car = g_car.logic # Récupère l'objet Vehicle lié

        delta = 0
        if logic_car.orientation == 'h' and delta_row == 0:
            delta = delta_col
        elif logic_car.orientation == 'v' and delta_col == 0:
            delta = delta_row

        # 1. Validation du mouvement via la logique
        if delta != 0 and self.board_state.is_move_valid(g_car.id, delta):
            # 2. Mouvement valide : Application des coordonnées à l'objet LOGIQUE
            if logic_car.orientation == 'h':
                logic_car.col = target_col
            else:
                logic_car.row = target_row

            # 3. Mise à jour de la matrice de grille après le mouvement logique
            self.board_state.grid = self.board_state._update_grid_matrix()
            self.turns += 1

            # 4. Synchronisation graphique (alignement sur la grille)
            g_car.update_position_from_logic()
            
        else:
            # Mouvement invalide : Annulation graphique (retour à la position de départ)
            g_car.rect.x = old_col * PER_SQ
            g_car.rect.y = old_row * PER_SQ
            if delta != 0: messagebox.showwarning('Erreur', 'Mouvement invalide : Collision ou hors limites.')
            
        self.selected_car_id = None # Désélectionne la voiture

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

    def _check_game_over(self):
        """Vérifie la condition de victoire."""
        if self.board_state.is_solved():
            messagebox.showinfo('Félicitations !', f'Vous avez terminé en {self.turns} mouvements !')
            self.inGame = False
            pygame.quit()
            sys.exit()

    def run(self):
        """Lance la boucle principale."""
        start = True
        while self.inGame:

            self.ev = pygame.event.poll()
            if self.ev.type == pygame.QUIT:
                self.inGame = False
            elif self.ev.type == pygame.MOUSEBUTTONDOWN:
                self._click_object()
            elif self.ev.type == pygame.MOUSEBUTTONUP:
                self._unclick_object()
            elif self.ev.type == pygame.MOUSEMOTION:
                self._object_mid_air()

            self._draw_board()
            pygame.display.flip()

            if start:
                messagebox.showinfo('Bienvenue !', 'Rush Hour')
                start = False

            self._check_game_over()

        pygame.quit()
        sys.exit()