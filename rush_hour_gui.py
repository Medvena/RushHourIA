# rush_hour_gui.py
import pygame
import sys
from tkinter import messagebox, Tk
from vehicle_logic import Vehicle, BoardState, PER_SQ, GRID_SIZE, EXIT_ROW, EXIT_COL
from typing import List

# Masquer la fenêtre Tkinter principale pour les pop-ups
Tk().wm_withdraw()


class GraphicalCar:
    """Représentation graphique d'un véhicule."""

    def __init__(self, vehicle_logic: Vehicle):
        self.id = vehicle_logic.id
        extendX = PER_SQ * vehicle_logic.size if vehicle_logic.orientation == "h" else PER_SQ
        extendY = PER_SQ if vehicle_logic.orientation == "h" else PER_SQ * vehicle_logic.size
        startX = vehicle_logic.col * PER_SQ
        startY = vehicle_logic.row * PER_SQ

        if self.id == 'X':
            self.colour = (204, 0, 0)  # Rouge
        elif vehicle_logic.orientation == 'h':
            self.colour = (0, 255, 0)  # Vert
        else:
            self.colour = (0, 0, 255)  # Bleu

        self.rectDrag = False
        self.offsetX = 0
        self.offsetY = 0
        self.rect = pygame.Rect(startX, startY, extendX, extendY)
        self.current_logic_pos = (vehicle_logic.row, vehicle_logic.col)


class RushHourGUI:
    """Contrôleur principal du jeu et de l'interface."""

    def __init__(self, initial_vehicles: List[Vehicle]):

        pygame.init()
        self.board_state = BoardState(initial_vehicles)
        self.g_cars = self._create_graphical_cars()
        self.turns = 0
        self.selected_car_id = None
        self.inGame = True

        surfaceSize = GRID_SIZE * PER_SQ
        self.surface = pygame.display.set_mode((surfaceSize, surfaceSize))
        pygame.display.set_caption("Rush Hour - Projet IA")

    def _create_graphical_cars(self):
        g_cars = {}
        for v_id, vehicle_logic in self.board_state.vehicles.items():
            g_cars[v_id] = GraphicalCar(vehicle_logic)
        return g_cars

    def _click_object(self):
        """Sélectionne la voiture à l'endroit du clic."""
        for g_car in self.g_cars.values():
            if g_car.rect.collidepoint(self.ev.pos):
                g_car.rectDrag = True
                self.selected_car_id = g_car.id

                mouseX, mouseY = self.ev.pos
                g_car.offsetX = g_car.rect.x - mouseX
                g_car.offsetY = g_car.rect.y - mouseY

                logic_car = self.board_state.vehicles[g_car.id]
                g_car.current_logic_pos = (logic_car.row, logic_car.col)
                break

    def _object_mid_air(self):
        """Déplace l'objet graphique pendant le glisser-déposer."""
        if self.selected_car_id:
            g_car = self.g_cars[self.selected_car_id]
            if g_car.rectDrag:
                mouseX, mouseY = self.ev.pos
                g_car.rect.x = mouseX + g_car.offsetX
                g_car.rect.y = mouseY + g_car.offsetY

    def _unclick_object(self):
        """Relâchement : vérifie le mouvement, l'applique à la logique ou annule."""
        if not self.selected_car_id: return
        g_car = self.g_cars[self.selected_car_id]
        g_car.rectDrag = False

        # Calcul de la nouvelle position en coordonnées de GRILLE (arrondi au plus proche)
        target_col = int(round(g_car.rect.x / PER_SQ))
        target_row = int(round(g_car.rect.y / PER_SQ))

        old_row, old_col = g_car.current_logic_pos
        delta_col = target_col - old_col
        delta_row = target_row - old_row

        logic_car = self.board_state.vehicles[g_car.id]
        delta = 0
        if logic_car.orientation == 'h' and delta_row == 0:
            delta = delta_col
        elif logic_car.orientation == 'v' and delta_col == 0:
            delta = delta_row

        if delta != 0 and self.board_state.is_move_valid(g_car.id, delta):
            # Mouvement valide : Application à la logique
            if logic_car.orientation == 'h':
                logic_car.col = target_col
            else:
                logic_car.row = target_row

            self.board_state.grid = self.board_state._update_grid_matrix()
            self.turns += 1

            # Mise à jour graphique pour l'alignement parfait
            g_car.rect.x = logic_car.col * PER_SQ
            g_car.rect.y = logic_car.row * PER_SQ
            g_car.current_logic_pos = (logic_car.row, logic_car.col)

        else:
            # Mouvement invalide : Annulation graphique
            g_car.rect.x = old_col * PER_SQ
            g_car.rect.y = old_row * PER_SQ
            if delta != 0: messagebox.showwarning('Erreur', 'Mouvement invalide : Collision ou hors limites.')

    def _draw_board(self):
        """Dessine le plateau, les cases de grille et les véhicules."""

        self.surface.fill((255, 255, 255))

        # Dessiner la grille
        for i in range(GRID_SIZE):
            pygame.draw.line(self.surface, (0, 0, 0), (0, i * PER_SQ), (GRID_SIZE * PER_SQ, i * PER_SQ), 1)
            pygame.draw.line(self.surface, (0, 0, 0), (i * PER_SQ, 0), (i * PER_SQ, GRID_SIZE * PER_SQ), 1)

        # Dessiner la sortie (carré vert)
        exit_rect = pygame.Rect(EXIT_COL * PER_SQ, EXIT_ROW * PER_SQ, PER_SQ, PER_SQ)
        pygame.draw.rect(self.surface, (0, 200, 0), exit_rect, 0)
        pygame.draw.rect(self.surface, (0, 0, 0), exit_rect, 2)

        # Dessiner les voitures
        for g_car in self.g_cars.values():
            self.surface.fill(g_car.colour, g_car.rect)
            pygame.draw.rect(self.surface, (0, 0, 0), g_car.rect, 5)

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