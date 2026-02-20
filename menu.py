import pygame
import time
import sys
import os
import math

# Project imports
from levels import load_level, list_levels
from rush_hour_gui import RushHourGUI
from config import GRID_SIZE, RED_CAR_ID
from board import BoardState
from solver_ia import train_cumulative, get_global_agent, state_to_tensor, SolverBFS, solve_astar_stepwise

# --- CONFIGURATION MENU ---
MENU_W = 800
MENU_H = 800
BG_COLOR = (240, 240, 245)
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)

# Couleurs pour teindre les boutons
COLOR_GREEN = (46, 204, 113)
COLOR_ORANGE = (230, 126, 34)
COLOR_RED = (231, 76, 60)
COLOR_BLUE = (52, 152, 219)
COLOR_GRAY = (100, 100, 100)

OFFSET_MENU = 6

# --- FONCTIONS UTILES ---

def get_tinted_image(image, color):
    """Applique une teinte de couleur à une image tout en gardant les traits noirs."""
    tinted = image.copy()
    color_layer = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    color_layer.fill((*color, 255))
    tinted.blit(color_layer, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return tinted

def draw_mini_board(screen, vehicles, x, y, size, assets):
    available_size = size - (OFFSET_MENU * 2)
    cell_size = available_size / 6
    screen.blit(pygame.transform.smoothscale(assets['parking'], (size, size)), (x, y))

    for v in vehicles:
        img_src = assets['red_car'] if v.id == RED_CAR_ID else (assets['truck'] if v.size == 3 else assets['car'])
        w_px = cell_size * v.size if v.orientation == "h" else cell_size
        h_px = cell_size if v.orientation == "h" else cell_size * v.size
        margin = 2 

        if v.orientation == "h":
            temp = pygame.transform.smoothscale(img_src, (int(cell_size - margin*2), int(cell_size * v.size - margin*2)))
            img_final = pygame.transform.rotate(temp, 90)
        else:
            img_final = pygame.transform.smoothscale(img_src, (int(w_px - margin*2), int(h_px - margin*2)))

        vx = x + (v.col * cell_size) + OFFSET_MENU + margin
        vy = y + (v.row * cell_size) + OFFSET_MENU + margin
        screen.blit(img_final, (vx, vy))

def draw_custom_button(screen, text, x, y, w, h, font, mouse_pos, img_base, color):
    rect = pygame.Rect(x, y, w, h)
    is_hover = rect.collidepoint(mouse_pos)
    
    # Création de la version teintée
    tinted_btn = get_tinted_image(img_base, color)
    btn_img = pygame.transform.smoothscale(tinted_btn, (w, h))
    
    # --- EFFET HOVER : ASSOMBRISSEMENT ---
    if is_hover:
        # On multiplie par un gris (ex: 180,180,180) pour assombrir l'image
        darken_layer = pygame.Surface((w, h), pygame.SRCALPHA)
        darken_layer.fill((210, 210, 210, 255)) 
        btn_img.blit(darken_layer, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    screen.blit(btn_img, rect)
    
    # Texte blanc par-dessus
    txt_surf = font.render(text, True, WHITE)
    txt_rect = txt_surf.get_rect(center=rect.center)
    screen.blit(txt_surf, txt_rect)
    return rect

def draw_button_standard(screen, text, x, y, w, h, font, mouse_pos, color, color_hover):
    rect = pygame.Rect(x, y, w, h)
    is_hover = rect.collidepoint(mouse_pos)
    pygame.draw.rect(screen, color_hover if is_hover else color, rect, border_radius=12)
    pygame.draw.rect(screen, WHITE, rect, 2, border_radius=12)
    txt = font.render(text, True, WHITE)
    screen.blit(txt, txt.get_rect(center=rect.center))
    return rect

def watch_ai_play(level_number, font):
    agent = get_global_agent()
    if agent is None: return
    pygame.display.set_mode((1000, 1000))
    game = RushHourGUI(load_level(level_number))
    step_gen = None
    try:
        step_gen = solve_astar_stepwise(game.board_state, agent)
    except: step_gen = None
    
    solving = True
    while solving:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: solving = False
        
        if step_gen:
            try: v_id, delta = next(step_gen)
            except StopIteration: solving = False; continue
        else:
            path = SolverBFS.solve(game.board_state)
            if path: v_id, delta = path[0]
            else: solving = False; continue

        next_board = game.board_state.get_next_state(v_id, delta)
        if next_board:
            game.board_state = next_board
            if v_id in game.g_vehicles:
                game.g_vehicles[v_id].logic = next_board.vehicles[v_id]
                game.g_vehicles[v_id].update_position_from_logic()
            game._draw_board()
            pygame.display.flip()
            time.sleep(0.15)
            if game.board_state.is_solved(): solving = False
    time.sleep(1)

# --- MAIN MENU ---

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((MENU_W, MENU_H))
    pygame.display.set_caption("RUSH HOUR IA")

    assets = {}
    try:
        assets['car'] = pygame.image.load(os.path.join("images", "car_menu.png")).convert_alpha()
        assets['truck'] = pygame.image.load(os.path.join("images", "truck_menu.png")).convert_alpha()
        assets['parking'] = pygame.image.load(os.path.join("images", "parking_menu.png")).convert_alpha()
        assets['red_car'] = get_tinted_image(assets['car'], (220, 40, 40))
        assets['btn_1'] = pygame.image.load(os.path.join("images", "button_1.png")).convert_alpha()
        assets['btn_2'] = pygame.image.load(os.path.join("images", "button_2.png")).convert_alpha()
    except Exception as e:
        print(f"Erreur Assets: {e}")

    f_title = pygame.font.SysFont("Segoe UI", 55, bold=True)
    f_btn = pygame.font.SysFont("Segoe UI", 24, bold=True)
    f_mini = pygame.font.SysFont("Segoe UI", 14)

    current_level = 1
    max_levels = list_levels()
    show_selector = False
    running = True

    while running:
        model_exists = os.path.exists("rush_hour_brain.pth")
        mouse_pos = pygame.mouse.get_pos()
        click = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False
            if ev.type == pygame.MOUSEBUTTONDOWN: click = True

        screen.fill(BG_COLOR)

        if not show_selector:
            # Titre
            title_s = f_title.render("RUSH HOUR IA", True, BLACK)
            screen.blit(title_s, (MENU_W//2 - title_s.get_width()//2, 40))

            # Statut IA
            status_color = (46, 204, 113) if model_exists else (231, 76, 60)
            pygame.draw.rect(screen, status_color, (MENU_W - 150, 30, 120, 34), border_radius=17)
            lbl_status = f_mini.render("IA ONLINE" if model_exists else "IA OFFLINE", True, WHITE)
            screen.blit(lbl_status, lbl_status.get_rect(center=(MENU_W - 90, 47)))

            # Aperçu
            p_size = 350
            px, py = MENU_W//2 - p_size//2, 140
            draw_mini_board(screen, load_level(current_level), px, py, p_size, assets)

            # Flèches (Boutons standard conservés comme demandé)
            btn_prev = draw_button_standard(screen, "<", px - 60, py + 150, 50, 50, f_btn, mouse_pos, COLOR_BLUE, (41, 128, 185))
            btn_next = draw_button_standard(screen, ">", px + p_size + 10, py + 150, 50, 50, f_btn, mouse_pos, COLOR_BLUE, (41, 128, 185))
            
            # Button 2 : Niveau (Etiré en hauteur à 65)
            btn_grid = draw_custom_button(screen, f"NIVEAU {current_level} (CHANGER)", px - 5, py + p_size + 20, 360, 65, f_btn, mouse_pos, assets['btn_2'], COLOR_BLUE)

            # Button 1 : Jouer et Solver
            b_play = draw_custom_button(screen, "JOUER", 140, 580, 240, 65, f_btn, mouse_pos, assets['btn_1'], COLOR_GREEN)
            b_ia = draw_custom_button(screen, "IA SOLVER", 420, 580, 240, 65, f_btn, mouse_pos, assets['btn_1'], COLOR_ORANGE)

            # Académie (Bas Gauche)
            b_acad = draw_custom_button(screen, "APPRENTISSAGE DE L'IA", 30, 700, 320, 65, f_btn, mouse_pos, assets['btn_2'], COLOR_GRAY)

            # Quitter (Bas Droite)
            b_quit = draw_custom_button(screen, "QUITTER", MENU_W - 220, 700, 190, 65, f_btn, mouse_pos, assets['btn_1'], COLOR_RED)

            if click:
                if btn_prev.collidepoint(mouse_pos): current_level = current_level - 1 if current_level > 1 else max_levels
                elif btn_next.collidepoint(mouse_pos): current_level = current_level + 1 if current_level < max_levels else 1
                elif btn_grid.collidepoint(mouse_pos): show_selector = True
                elif b_acad.collidepoint(mouse_pos): train_cumulative(max_level=max_levels, progress_callback=lambda t, p: None)
                elif b_play.collidepoint(mouse_pos):
                    pygame.display.set_mode((1000, 1000))
                    RushHourGUI(load_level(current_level)).run()
                    screen = pygame.display.set_mode((MENU_W, MENU_H))
                elif b_ia.collidepoint(mouse_pos) and model_exists:
                    watch_ai_play(current_level, f_btn)
                    screen = pygame.display.set_mode((MENU_W, MENU_H))
                elif b_quit.collidepoint(mouse_pos): running = False
        else:
            # Sélecteur de niveau
            overlay = pygame.Surface((MENU_W, MENU_H), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 230))
            screen.blit(overlay, (0, 0))
            
            for i in range(1, max_levels + 1):
                r, c = (i - 1) // 8, (i - 1) % 8
                bx, by = 60 + c * 85, 120 + r * 85
                # Button 1 utilisé pour la grille des chiffres
                rect = draw_custom_button(screen, str(i), bx, by, 75, 75, f_btn, mouse_pos, assets['btn_1'], COLOR_BLUE if i != current_level else COLOR_ORANGE)
                if click and rect.collidepoint(mouse_pos):
                    current_level = i
                    show_selector = False

            # Retour (Button 2 étiré)
            if click and draw_custom_button(screen, "RETOUR", MENU_W//2 - 120, 700, 240, 65, f_btn, mouse_pos, assets['btn_2'], COLOR_RED).collidepoint(mouse_pos):
                show_selector = False

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main_menu()