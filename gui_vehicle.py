# gui_vehicle.py

import pygame
from vehicle import Vehicle
from config import PER_SQ

class GUIVehicle:
    """
    Associe les propriétés graphiques (Pygame) à un objet Vehicle logique.
    Ceci sépare la représentation (view) du modèle (logic).
    """

    def __init__(self, vehicle_logic: Vehicle):
        self.logic = vehicle_logic # Référence à l'objet Vehicle logique
        self.id = vehicle_logic.id
        
        # --- Propriétés graphiques calculées à partir de la logique ---
        
        # Dimensions en pixels
        extendX = PER_SQ * vehicle_logic.size if vehicle_logic.orientation == "h" else PER_SQ
        extendY = PER_SQ if vehicle_logic.orientation == "h" else PER_SQ * vehicle_logic.size
        
        # Position initiale en pixels
        startX = vehicle_logic.col * PER_SQ
        startY = vehicle_logic.row * PER_SQ
        
        # Définition de la couleur
        if self.id == 'X':
            self.colour = (204, 0, 0)  # Rouge (voiture à sortir)
        elif vehicle_logic.orientation == 'h':
            self.colour = (0, 255, 0)  # Vert (Horizontal standard)
        else:
            self.colour = (0, 0, 255)  # Bleu (Vertical standard)
            
        # Objet Pygame Rect (position actuelle et taille)
        self.rect = pygame.Rect(startX, startY, extendX, extendY)

        # --- Propriétés de glissement (Drag-and-Drop) ---
        self.rectDrag = False
        self.offsetX = 0
        self.offsetY = 0
        self.initial_drag_pos = (vehicle_logic.row, vehicle_logic.col) # Position logique au moment du clic

    def update_position_from_logic(self):
        """Synchronise la position graphique (rect) avec la position logique (Vehicle)."""
        self.rect.x = self.logic.col * PER_SQ
        self.rect.y = self.logic.row * PER_SQ

    def draw(self, surface: pygame.Surface):
        """Dessine la voiture sur la surface Pygame."""
        surface.fill(self.colour, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 5)