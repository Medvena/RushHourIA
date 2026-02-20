import pygame
import os
from vehicle import Vehicle

# On définit les constantes de dessin ici pour qu'elles correspondent au parking.png
# Ces valeurs doivent être identiques à celles utilisées dans RushHourGUI
DRAW_PER_SQ = 105  # La taille d'une case sur ton dessin Krita
OFFSET_X = 185     # La marge gauche sur ton dessin
OFFSET_Y = 185     # La marge haute sur ton dessin

VEHICLE_COLORS = {
    # Cars
    "X": (204, 0, 0),      # red
    "A": (144, 238, 144),  # light green
    "B": (255, 165, 0),    # orange
    "C": (173, 216, 230),  # light blue
    "D": (255, 182, 193),  # pink
    "E": (186, 85, 211),   # purple
    "F": (0, 128, 0),      # dark green
    "G": (90, 90, 90),     # dark gray
    "H": (245, 245, 220),  # beige
    "I": (252, 251, 155),  # light yellow
    "J": (139, 69, 19),    # brown
    "K": (128, 128, 0),    # khaki

    # Trucks
    "O": (255, 215, 0),    # yellow
    "P": (216, 191, 216),  # light violet
    "Q": (52, 119, 247),   # blue
    "R": (64, 224, 208),   # turquoise
}

class GUIVehicle:
    def __init__(self, vehicle_logic: Vehicle):
        self.logic = vehicle_logic
        self.id = vehicle_logic.id
        self.colour = VEHICLE_COLORS.get(self.id, (128, 128, 128))

        # Dimensions basées sur le dessin (105px par case)
        self.width = DRAW_PER_SQ * vehicle_logic.size if vehicle_logic.orientation == "h" else DRAW_PER_SQ
        self.height = DRAW_PER_SQ if vehicle_logic.orientation == "h" else DRAW_PER_SQ * vehicle_logic.size

        # Position initiale incluant l'OFFSET (185px)
        start_x = vehicle_logic.col * DRAW_PER_SQ + OFFSET_X
        start_y = vehicle_logic.row * DRAW_PER_SQ + OFFSET_Y

        self.rect = pygame.Rect(start_x, start_y, self.width, self.height)

        # Chargement de l'image avec les nouvelles dimensions
        self.image = self._init_image(self.width, self.height)

        self.rectDrag = False
        self.offsetX = 0
        self.offsetY = 0
        self.initial_drag_pos = (vehicle_logic.row, vehicle_logic.col)

    def _init_image(self, width, height):
        file_name = "car.png" if self.logic.size == 2 else "truck.png"
        path = os.path.join("images", file_name)

        try:
            img = pygame.image.load(path).convert_alpha()
            
            # On utilise DRAW_PER_SQ pour le redimensionnement de base
            vertical_w = DRAW_PER_SQ
            vertical_h = DRAW_PER_SQ * self.logic.size
            img = pygame.transform.smoothscale(img, (vertical_w, vertical_h))

            if self.logic.orientation == "h":
                img = pygame.transform.rotate(img, -90)
                # S'assurer que l'image rotée fait bien la taille finale width/height
                img = pygame.transform.smoothscale(img, (width, height))

            color_layer = pygame.Surface((width, height), pygame.SRCALPHA)
            color_layer.fill((*self.colour, 255))
            img.blit(color_layer, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
            return img

        except (pygame.error, FileNotFoundError):
            fallback = pygame.Surface((width, height))
            fallback.fill(self.colour)
            return fallback

    def update_position_from_logic(self):
        """Synchronise la position en pixels avec l'offset du dessin."""
        self.rect.x = self.logic.col * DRAW_PER_SQ + OFFSET_X
        self.rect.y = self.logic.row * DRAW_PER_SQ + OFFSET_Y

    def draw(self, surface: pygame.Surface):
        surface.blit(self.image, self.rect)