import pygame
import os
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

        # --- Image Loading and Processing ---
        self.image = self._init_image(width, height)

        # --- Drag-and-drop state ---
        self.rectDrag = False
        self.offsetX = 0
        self.offsetY = 0

        # Logical position at mouse click time
        self.initial_drag_pos = (vehicle_logic.row, vehicle_logic.col)

    def _init_image(self, width, height):
        """Loads the image from /images, rotates, and applies the vehicle color."""
        # Determine file based on size (2 = car, 3 = truck)
        file_name = "car.png" if self.logic.size == 2 else "truck.png"
        path = os.path.join("images", file_name)

        try:
            # Load with alpha channel
            img = pygame.image.load(path).convert_alpha()
            
            # Base scale: Always scale to a vertical version first for consistency
            vertical_w = PER_SQ
            vertical_h = PER_SQ * self.logic.size
            img = pygame.transform.smoothscale(img, (vertical_w, vertical_h))

            # Rotate if orientation is horizontal
            if self.logic.orientation == "h":
                img = pygame.transform.rotate(img, -90)

            # Apply color tint using BLEND_RGBA_MULT
            # This turns white areas into the vehicle color while keeping black lines intact
            color_layer = pygame.Surface((width, height), pygame.SRCALPHA)
            color_layer.fill((*self.colour, 255))
            img.blit(color_layer, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
            return img

        except (pygame.error, FileNotFoundError):
            # Fallback to a solid color rectangle if image is missing
            fallback = pygame.Surface((width, height))
            fallback.fill(self.colour)
            return fallback

    def update_position_from_logic(self):
        """Synchronize the graphical position with the logical Vehicle position."""
        self.rect.x = self.logic.col * PER_SQ
        self.rect.y = self.logic.row * PER_SQ

    def draw(self, surface: pygame.Surface):
        """Draw the vehicle onto the given Pygame surface."""
        # Draw the processed image only
        surface.blit(self.image, self.rect)