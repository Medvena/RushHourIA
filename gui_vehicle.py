# gui_vehicle.py

import pygame
from vehicle import Vehicle
from config import PER_SQ

# Vehicle colors mapped by ID (RGB)
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
    "I": (255, 255, 224),  # light yellow
    "J": (139, 69, 19),    # brown
    "K": (128, 128, 0),    # khaki

    # Trucks
    "O": (255, 215, 0),    # yellow
    "P": (216, 191, 216),  # light violet
    "Q": (0, 0, 139),      # dark blue
    "R": (64, 224, 208),   # turquoise
}

class GUIVehicle:
    """
    Binds graphical (Pygame) properties to a logical Vehicle object.
    This cleanly separates the view layer from the game logic.
    """
    
    def __init__(self, vehicle_logic: Vehicle):
        # Reference to the logical Vehicle
        self.logic = vehicle_logic
        self.id = vehicle_logic.id

        # --- Graphical properties derived from logic ---

        # Pixel dimensions
        width = PER_SQ * vehicle_logic.size if vehicle_logic.orientation == "h" else PER_SQ
        height = PER_SQ if vehicle_logic.orientation == "h" else PER_SQ * vehicle_logic.size

        # Initial pixel position
        start_x = vehicle_logic.col * PER_SQ
        start_y = vehicle_logic.row * PER_SQ

        # Vehicle color (fallback to gray if unknown)
        self.colour = VEHICLE_COLORS.get(self.id, (128, 128, 128))

        # Pygame rectangle (current position and size)
        self.rect = pygame.Rect(start_x, start_y, width, height)

        # --- Drag-and-drop state ---
        self.rectDrag = False
        self.offsetX = 0
        self.offsetY = 0

        # Logical position at mouse click time
        self.initial_drag_pos = (vehicle_logic.row, vehicle_logic.col)

    def update_position_from_logic(self):
        """Synchronize the graphical position with the logical Vehicle position."""
        self.rect.x = self.logic.col * PER_SQ
        self.rect.y = self.logic.row * PER_SQ

    def draw(self, surface: pygame.Surface):
        """Draw the vehicle onto the given Pygame surface."""
        surface.fill(self.colour, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 5)
