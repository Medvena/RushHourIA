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
# Import des fonctions d'entraînement et du solver
from solver_ia import train_cumulative, get_global_agent, state_to_tensor, SolverBFS

# --- CONFIGURATION MENU ---
MENU_W = 800
MENU_H = 800
BG_COLOR = (240, 240, 245)
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)

# Couleurs
BLUE_BTN, BLUE_HOVER = (52, 152, 219), (41, 128, 185)
ORANGE_BTN, ORANGE_HOVER = (230, 126, 34), (211, 84, 0)
GREEN_BTN, GREEN_HOVER = (46, 204, 113), (39, 174, 96)
RED_BTN, RED_HOVER = (231, 76, 60), (192, 57, 43)
GRAY_LIGHT = (200, 200, 200)
GRAY_DARK = (50, 50, 50)
GRAY_TEXT_LIGHT = (100, 100, 100)

# --- AJUSTEMENT DE L'OFFSET POUR LE MENU ---
OFFSET_MENU = 6

# --- FONCTIONS UTILES ---

def get_tinted_red(image):
    tinted = image.copy()
    color_layer = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    color_layer.fill((220, 40, 40, 255))
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

def draw_button(screen, text, x, y, w, h, font, mouse_pos, color, color_hover, border_radius=12):
    rect = pygame.Rect(x, y, w, h)
    is_hover = rect.collidepoint(mouse_pos)
    actual_color = color_hover if is_hover else color
    pygame.draw.rect(screen, (200, 200, 200), (x + 4, y + 4, w, h), border_radius=border_radius) # Ombre
    pygame.draw.rect(screen, actual_color, rect, border_radius=border_radius)
    pygame.draw.rect(screen, WHITE, rect, 2, border_radius=border_radius)
    txt = font.render(text, True, WHITE)
    screen.blit(txt, txt.get_rect(center=rect.center))
    return rect

def draw_loading_screen(screen, font, title, percent):
    screen.fill(BG_COLOR)
    center_x, center_y = MENU_W // 2, MENU_H // 2
    f_big = pygame.font.SysFont("Segoe UI", 30, bold=True)
    txt_title = f_big.render(title, True, BLACK)
    screen.blit(txt_title, (center_x - txt_title.get_width() // 2, center_y - 80))
    
    bar_w, bar_h = 500, 40
    pygame.draw.rect(screen, GRAY_LIGHT, (center_x - bar_w // 2, center_y, bar_w, bar_h), border_radius=20)
    fill_w = int(bar_w * (percent / 100))
    if fill_w > 0:
        pygame.draw.rect(screen, ORANGE_BTN, (center_x - bar_w // 2, center_y, fill_w, bar_h), border_radius=20)
    
    txt_pct = font.render(f"{percent}%", True, BLACK)
    screen.blit(txt_pct, (center_x - txt_pct.get_width() // 2, center_y + 50))
    pygame.display.flip()
    pygame.event.pump()

# --- MENU PRINCIPAL ---

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((MENU_W, MENU_H))
    pygame.display.set_caption("RUSH HOUR IA")

    assets = {}
    try:
        assets['car'] = pygame.image.load(os.path.join("images", "car_menu.png")).convert_alpha()
        assets['truck'] = pygame.image.load(os.path.join("images", "truck_menu.png")).convert_alpha()
        assets['parking'] = pygame.image.load(os.path.join("images", "parking_menu.png")).convert_alpha()
        assets['red_car'] = get_tinted_red(assets['car'])
    except Exception as e:
        print(f"ERREUR CHARGEMENT : {e}")
        for k in ['car', 'truck', 'parking', 'red_car']:
            assets[k] = pygame.Surface((100, 100))
            assets[k].fill((255, 0, 255))

    f_title = pygame.font.SysFont("Segoe UI", 55, bold=True)
    f_btn = pygame.font.SysFont("Segoe UI", 24, bold=True)
    f_lvl = pygame.font.SysFont("Segoe UI", 20, bold=True)
    f_small = pygame.font.SysFont("Segoe UI", 18)
    f_mini = pygame.font.SysFont("Segoe UI", 14)

    current_level = 1
    max_levels = list_levels()
    show_selector = False
    running = True

    while running:
        # Vérification de l'existence du modèle pour le statut
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

            # INDICATEUR STATUT IA (En haut à droite)
            status_color = GREEN_BTN if model_exists else RED_BTN
            status_text = "IA ONLINE" if model_exists else "IA OFFLINE"
            pygame.draw.rect(screen, status_color, (MENU_W - 160, 30, 130, 34), border_radius=17)
            lbl_status = f_mini.render(status_text, True, WHITE)
            screen.blit(lbl_status, lbl_status.get_rect(center=(MENU_W - 95, 47)))

            # Aperçu
            p_size = 350
            px, py = MENU_W//2 - p_size//2, 140
            try:
                current_v = load_level(current_level)
                draw_mini_board(screen, current_v, px, py, p_size, assets)
            except: pass

            # Navigation
            btn_prev = draw_button(screen, "<", px - 60, py + 150, 50, 50, f_btn, mouse_pos, BLUE_BTN, BLUE_HOVER)
            btn_next = draw_button(screen, ">", px + p_size + 10, py + 150, 50, 50, f_btn, mouse_pos, BLUE_BTN, BLUE_HOVER)
            btn_grid = draw_button(screen, f"NIVEAU {current_level} (CHANGER)", px, py + p_size + 15, p_size, 45, f_btn, mouse_pos, BLUE_BTN, BLUE_HOVER)

            # Actions principales
            b_play = draw_button(screen, "JOUER", 150, 580, 240, 60, f_btn, mouse_pos, GREEN_BTN, GREEN_HOVER)
            
            ia_btn_color = ORANGE_BTN if model_exists else GRAY_LIGHT
            ia_btn_hover = ORANGE_HOVER if model_exists else GRAY_LIGHT
            b_ia = draw_button(screen, "IA SOLVER", 410, 580, 240, 60, f_btn, mouse_pos, ia_btn_color, ia_btn_hover)

            # BOUTON ACADÉMIE (En bas à gauche)
            acad_y = MENU_H - 80
            btn_academy = draw_button(screen, "APPRENTISSAGE DE L'IA", 30, acad_y, 300, 40, f_small, mouse_pos, GRAY_DARK, (80, 80, 80))
            lbl_info = f_mini.render(f"Apprend automatiquement les {max_levels} niveaux.", True, GRAY_TEXT_LIGHT)
            screen.blit(lbl_info, (35, acad_y + 45))

            # Bouton Quitter
            b_quit = draw_button(screen, "QUITTER", MENU_W - 150, MENU_H - 60, 120, 35, f_small, mouse_pos, RED_BTN, RED_HOVER)

            if click:
                if btn_prev.collidepoint(mouse_pos):
                    current_level = current_level - 1 if current_level > 1 else max_levels
                elif btn_next.collidepoint(mouse_pos):
                    current_level = current_level + 1 if current_level < max_levels else 1
                elif btn_grid.collidepoint(mouse_pos):
                    show_selector = True
                elif btn_academy.collidepoint(mouse_pos):
                    train_cumulative(max_level=max_levels, progress_callback=lambda t, p: draw_loading_screen(screen, f_btn, t, p))
                elif b_play.collidepoint(mouse_pos):
                    pygame.display.set_mode((1000, 1000))
                    RushHourGUI(load_level(current_level)).run()
                    screen = pygame.display.set_mode((MENU_W, MENU_H))
                elif b_ia.collidepoint(mouse_pos) and model_exists:
                    agent = get_global_agent()
                    if agent:
                        pygame.display.set_mode((1000, 1000))
                        game = RushHourGUI(load_level(current_level))
                        solving = True
                        while solving:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT: solving = False
                            solving = game.solve_step_with_ai(agent)
                            game._draw_board()
                            pygame.display.flip()
                            time.sleep(0.15)
                            if game.board_state.is_solved(): solving = False
                        time.sleep(1)
                    screen = pygame.display.set_mode((MENU_W, MENU_H))
                elif b_quit.collidepoint(mouse_pos):
                    running = False
        else:
            # Grille de sélection
            overlay = pygame.Surface((MENU_W, MENU_H), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 230))
            screen.blit(overlay, (0, 0))
            title_grid = f_title.render("SÉLECTIONNER UN NIVEAU", True, BLACK)
            screen.blit(title_grid, (MENU_W//2 - title_grid.get_width()//2, 50))

            cols, btn_s, marg = 8, 70, 20
            start_x = (MENU_W - (cols * (btn_s + marg) - marg)) // 2
            start_y = 150
            for i in range(1, max_levels + 1):
                r, c = (i - 1) // cols, (i - 1) % cols
                bx, by = start_x + c * (btn_s + marg), start_y + r * (btn_s + marg)
                rect = draw_button(screen, str(i), bx, by, btn_s, btn_s, f_lvl, mouse_pos, ORANGE_BTN if i == current_level else BLUE_BTN, ORANGE_HOVER if i == current_level else BLUE_HOVER, border_radius=10)
                if click and rect.collidepoint(mouse_pos):
                    current_level = i
                    show_selector = False

            if click and draw_button(screen, "RETOUR", MENU_W//2 - 100, MENU_H - 80, 200, 50, f_btn, mouse_pos, RED_BTN, RED_HOVER).collidepoint(mouse_pos):
                show_selector = False

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_menu()