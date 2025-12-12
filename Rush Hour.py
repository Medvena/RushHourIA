import pygame
import sys
import math
import copy
from typing import List, Tuple, Dict, Optional
from tkinter import messagebox, Tk  # Utilisé pour les pop-ups seulement

# --- Définitions du Noyau Logique (Copie/Colle du code précédent) ---

GRID_SIZE = 6
EXIT_ROW = 2
EXIT_COL = 5
PER_SQ = 80  # Taille d'une case en pixels (80x80)


class Vehicle:
    """Représentation logique du véhicule."""

    def __init__(self, id: str, orientation: str, size: int, row: int, col: int):
        self.id = id
        self.orientation = orientation
        self.size = size
        self.row = row  # Ligne de la tête (coordonnée GRILLE)
        self.col = col  # Colonne de la tête (coordonnée GRILLE)

    def __repr__(self):
        return f"V({self.id}, {self.orientation}, L{self.row}C{self.col})"

    def __eq__(self, other):
        if not isinstance(other, Vehicle): return NotImplemented
        return (self.id == other.id and self.orientation == other.orientation and
                self.size == other.size and self.row == other.row and self.col == other.col)

    def __hash__(self):
        return hash((self.id, self.orientation, self.size, self.row, self.col))


class BoardState:
    """Représente l'état complet du plateau logique."""

    def __init__(self, vehicles: List[Vehicle]):
        self.vehicles: Dict[str, Vehicle] = {v.id: v for v in vehicles}
        self.grid = self._update_grid_matrix()

    # Méthodes _update_grid_matrix, __repr__, __eq__, __hash__ restent les mêmes
    # ... (les méthodes du code précédent sont ici)

    def _update_grid_matrix(self) -> List[List[Optional[str]]]:
        grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        for vehicle in self.vehicles.values():
            if vehicle.orientation == 'h':
                for i in range(vehicle.size):
                    if 0 <= vehicle.col + i < GRID_SIZE and 0 <= vehicle.row < GRID_SIZE:
                        grid[vehicle.row][vehicle.col + i] = vehicle.id
            else:  # orientation == 'v'
                for i in range(vehicle.size):
                    if 0 <= vehicle.row + i < GRID_SIZE and 0 <= vehicle.col < GRID_SIZE:
                        grid[vehicle.row + i][vehicle.col] = vehicle.id
        return grid

    def is_solved(self) -> bool:
        red_car = self.vehicles.get('X')
        if red_car is None: return False
        return red_car.row == EXIT_ROW and (red_car.col + red_car.size) == (EXIT_COL + 1)

    # Note: Pour le jeu jouable, on simplifie 'apply_move' pour éviter de générer un nouvel état
    # complet, mais on garde la vérification de la validité.

    def is_move_valid(self, v_id: str, delta: int) -> bool:
        """Vérifie si le véhicule peut se déplacer de 'delta' cases."""
        vehicle = self.vehicles.get(v_id)
        if not vehicle: return False

        temp_row, temp_col = vehicle.row, vehicle.col

        # Le déplacement doit être d'une case au minimum
        step = int(math.copysign(1, delta))

        for _ in range(abs(delta)):
            # On vérifie la case cible après un déplacement de 1
            if vehicle.orientation == 'h':
                # Si delta est positif, on regarde la case la plus à droite
                if step > 0:
                    temp_col += 1
                    target_col = temp_col + vehicle.size - 1
                    target_row = temp_row
                # Si delta est négatif, on regarde la case la plus à gauche
                else:
                    target_col = temp_col - 1
                    target_row = temp_row
                    temp_col -= 1

            else:  # orientation == 'v'
                # Si delta est positif, on regarde la case la plus en bas
                if step > 0:
                    temp_row += 1
                    target_row = temp_row + vehicle.size - 1
                    target_col = temp_col
                # Si delta est négatif, on regarde la case la plus en haut
                else:
                    target_row = temp_row - 1
                    target_col = temp_col
                    temp_row -= 1

            # 1. Vérification des limites du plateau
            if not (0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE):
                return False

            # 2. Vérification de la collision (la case doit être vide)
            # Exception pour la voiture rouge à la sortie (ligne 2, colonne 5)
            is_exit_slot = (target_row == EXIT_ROW and target_col == EXIT_COL)

            if self.grid[target_row][target_col] is not None:
                # Si c'est la voiture rouge qui bloque la sortie, c'est OK
                if is_exit_slot and v_id == 'X':
                    pass
                # Sinon, si la case n'est pas vide (et pas la sortie pour la rouge), c'est bloqué
                elif self.grid[target_row][target_col] != v_id:
                    return False

        # Si toutes les étapes intermédiaires sont valides, le mouvement est OK
        return True


# --- Classes Pygame pour l'Interface (GUI) ---

class GraphicalCar:
    """
    Représentation graphique d'un véhicule.
    Contient l'ID pour la liaison avec la logique et les attributs Pygame.
    """

    def __init__(self, vehicle_logic: Vehicle):
        self.id = vehicle_logic.id

        # Dimensions en pixels
        extendX = PER_SQ * vehicle_logic.size if vehicle_logic.orientation == "h" else PER_SQ
        extendY = PER_SQ if vehicle_logic.orientation == "h" else PER_SQ * vehicle_logic.size

        # Coordonnées de départ en pixels (GRILLE * PER_SQ)
        startX = vehicle_logic.col * PER_SQ
        startY = vehicle_logic.row * PER_SQ

        # Définition de la couleur
        if self.id == 'X':
            self.colour = (204, 0, 0)  # Rouge
        elif vehicle_logic.orientation == 'h':
            self.colour = (0, 255, 0)  # Vert
        else:
            self.colour = (0, 0, 255)  # Bleu

        # Attributs pour le glisser-déposer
        self.rectDrag = False
        self.offsetX = 0
        self.offsetY = 0

        # Objet Pygame Rect
        self.rect = pygame.Rect(startX, startY, extendX, extendY)
        self.current_logic_pos = (vehicle_logic.row, vehicle_logic.col)  # Position de la logique avant le drag


class RushHourGUI:  # La classe principale du jeu

    def __init__(self, initial_vehicles: List[Vehicle]):

        Tk().wm_withdraw()  # Cache la fenêtre Tkinter principale (pour les popups)
        pygame.init()

        self.board_state = BoardState(initial_vehicles)  # L'état logique est notre référence !
        self.g_cars = self._create_graphical_cars()  # Création des objets Pygame
        self.turns = 0
        self.selected_car_id = None  # ID de la voiture en cours de déplacement
        self.inGame = True

        # Configuration de la fenêtre
        surfaceSize = GRID_SIZE * PER_SQ
        self.surface = pygame.display.set_mode((surfaceSize, surfaceSize))
        pygame.display.set_caption("Rush Hour - Jouable")

        # Boucle principale
        self._main_loop()

    def _load_game_from_file(self, filename: str) -> List[Vehicle]:
        """Charge l'état initial des véhicules depuis un fichier texte."""
        vehicles = []
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split(', ')
                    if len(parts) == 4:
                        # parts: [orientation, size, row, col]
                        orientation, size_str, row_str, col_str = parts
                        size, row, col = int(size_str), int(row_str), int(col_str)
                        # On génère un ID simple ('X' pour la rouge, puis 'A', 'B', ...)
                        id = 'X' if row == EXIT_ROW and col == 0 and orientation == 'h' else chr(
                            ord('A') + len(vehicles))
                        vehicles.append(Vehicle(id, orientation, size, row, col))
        except FileNotFoundError:
            messagebox.showerror('Erreur', f"Fichier {filename} non trouvé.")
            sys.exit()
        return vehicles

    def _create_graphical_cars(self) -> Dict[str, GraphicalCar]:
        """Convertit l'état logique en objets graphiques Pygame."""
        g_cars = {}
        # Assurez-vous que l'ID 'X' est toujours la voiture rouge
        for v_id, vehicle_logic in self.board_state.vehicles.items():
            g_cars[v_id] = GraphicalCar(vehicle_logic)
        return g_cars

    def _main_loop(self):
        start = True
        while self.inGame:

            # 1. Gestion des événements
            self._handle_events()

            # 2. Dessin
            self._draw_board()

            # 3. Affichage
            pygame.display.flip()

            if start:
                messagebox.showinfo('Bienvenue !',
                                    'Rush Hour\nAmenez la voiture rouge (X) à la sortie (colonne 5, ligne 2).\n Utilisez le glisser-déposer.')
                start = False

            # 4. Vérification de la fin du jeu
            self._check_game_over()

        pygame.quit()
        sys.exit()

    def _handle_events(self):
        """Gère les entrées souris (clic, drag, relâchement)."""

        self.ev = pygame.event.poll()

        if self.ev.type == pygame.QUIT:
            self.inGame = False

        elif self.ev.type == pygame.MOUSEBUTTONDOWN:
            self._click_object()

        elif self.ev.type == pygame.MOUSEBUTTONUP:
            self._unclick_object()

        elif self.ev.type == pygame.MOUSEMOTION:
            self._object_mid_air()

    def _click_object(self):
        """Sélectionne la voiture à l'endroit du clic."""
        for g_car in self.g_cars.values():
            if g_car.rect.collidepoint(self.ev.pos):
                g_car.rectDrag = True
                self.selected_car_id = g_car.id

                # Sauvegarde la position Pygame relative à la souris
                mouseX, mouseY = self.ev.pos
                g_car.offsetX = g_car.rect.x - mouseX
                g_car.offsetY = g_car.rect.y - mouseY

                # Sauvegarde la position logique actuelle (en cas d'annulation)
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
        if not self.selected_car_id:
            return

        g_car = self.g_cars[self.selected_car_id]
        g_car.rectDrag = False

        # Calculer la position d'atterrissage sur la grille (multiples de 80)
        # On utilise math.floor(coord / PER_SQ) si l'on est moins à mi-chemin, et ceil si plus.
        target_col_float = g_car.rect.x / PER_SQ
        target_row_float = g_car.rect.y / PER_SQ

        target_col = int(round(target_col_float))
        target_row = int(round(target_row_float))

        old_row, old_col = g_car.current_logic_pos

        # Calcul du déplacement réel (delta)
        delta_col = target_col - old_col
        delta_row = target_row - old_row

        logic_car = self.board_state.vehicles[g_car.id]

        # Un seul des deux deltas doit être non nul selon l'orientation

        if logic_car.orientation == 'h' and delta_row == 0 and delta_col != 0:
            delta = delta_col
        elif logic_car.orientation == 'v' and delta_col == 0 and delta_row != 0:
            delta = delta_row
        else:
            delta = 0  # Mouvement invalide (diagonale, pas de mouvement, mauvaise direction)

        # --- Cœur de la Vérification ---

        if delta != 0 and self.board_state.is_move_valid(g_car.id, delta):

            # Mouvement valide : Application à la logique
            if logic_car.orientation == 'h':
                logic_car.col = target_col
            else:
                logic_car.row = target_row

            # Mise à jour de la grille interne du BoardState et des compteurs
            self.board_state.grid = self.board_state._update_grid_matrix()
            self.turns += 1

            # Mise à jour de la position Pygame pour l'aligner parfaitement
            g_car.rect.x = logic_car.col * PER_SQ
            g_car.rect.y = logic_car.row * PER_SQ
            g_car.current_logic_pos = (logic_car.row, logic_car.col)

        else:
            # Mouvement invalide : Annulation graphique (repositionnement)
            g_car.rect.x = old_col * PER_SQ
            g_car.rect.y = old_row * PER_SQ

            if delta != 0:  # On affiche un message uniquement si un déplacement était tenté
                messagebox.showwarning('Erreur', 'Mouvement invalide : Collision ou hors limites.')

    def _draw_board(self):
        """Dessine le plateau, les cases de grille et les véhicules."""

        # Fond blanc
        self.surface.fill((255, 255, 255))

        # Dessiner la grille (lignes noires)
        for i in range(GRID_SIZE):
            # Lignes horizontales
            pygame.draw.line(self.surface, (0, 0, 0), (0, i * PER_SQ), (GRID_SIZE * PER_SQ, i * PER_SQ), 1)
            # Lignes verticales
            pygame.draw.line(self.surface, (0, 0, 0), (i * PER_SQ, 0), (i * PER_SQ, GRID_SIZE * PER_SQ), 1)

        # Dessiner la sortie (carré vert)
        exit_rect = pygame.Rect(EXIT_COL * PER_SQ, EXIT_ROW * PER_SQ, PER_SQ, PER_SQ)
        pygame.draw.rect(self.surface, (0, 200, 0), exit_rect, 0)  # Remplissage vert
        pygame.draw.rect(self.surface, (0, 0, 0), exit_rect, 2)  # Bordure noire

        # Dessiner les voitures
        for g_car in self.g_cars.values():
            # Remplissage
            self.surface.fill(g_car.colour, g_car.rect)
            # Bordures noires épaisses
            pygame.draw.rect(self.surface, (0, 0, 0), g_car.rect, 5)

    def _check_game_over(self):
        """Vérifie la condition de victoire."""
        if self.board_state.is_solved():
            messagebox.showinfo('Félicitations !', f'Vous avez terminé en {self.turns} mouvements !')
            self.inGame = False


# --- Initialisation et Lancement ---

if __name__ == '__main__':
    # Initialisation basée sur un jeu standard (comme celui que vous lisiez)
    # Lisez le fichier ou définissez les véhicules ici

    # Pour le test, on va définir un jeu initial simple à la main
    # Vous pouvez le modifier pour lire votre 'game0.txt' si vous préférez.

    # X: H, 2, L2, C0 (Rouge)
    # A: V, 3, L0, C0 (Bloque X)
    # B: H, 2, L0, C4
    # C: V, 2, L4, C5 (Proche de la sortie)

    initial_config = [
        Vehicle('X', 'h', 2, 2, 0),
        Vehicle('A', 'v', 2, 0, 0),
        Vehicle('B', 'h', 2, 0, 4),
        Vehicle('C', 'v', 2, 4, 5),
    ]

    # Début du jeu
    RushHourGUI(initial_config)