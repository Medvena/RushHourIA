import pygame
import time
import sys
import os
import math
import copy

# Project imports
from levels import load_level, list_levels
from rush_hour_gui import RushHourGUI
from config import GRID_SIZE, RED_CAR_ID
from board import BoardState

# On importe les fonctions supervisées et de résolution
from solver_ia import Agent, train_cumulative, get_global_agent, state_to_tensor, SolverBFS, solve_astar_complete, plot_learning_curve

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
COLOR_BLUE = (52, 152, 219)
GRAY_LIGHT = (200, 200, 200)
GRAY_DARK = (100, 100, 100)
GRAY_TEXT_LIGHT = (150, 150, 150)

OFFSET_MENU = 6

# --- FONCTIONS UTILES ---

def get_tinted_image(image, color):
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
    tinted_btn = get_tinted_image(img_base, color)
    btn_img = pygame.transform.smoothscale(tinted_btn, (w, h))
    
    if is_hover:
        darken_layer = pygame.Surface((w, h), pygame.SRCALPHA)
        darken_layer.fill((180, 180, 180, 255)) 
        btn_img.blit(darken_layer, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    screen.blit(btn_img, rect)
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

def draw_popup(screen, font, text):
    overlay = pygame.Surface((MENU_W, MENU_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))
    box = pygame.Rect(MENU_W//2 - 250, MENU_H//2 - 100, 500, 200)
    pygame.draw.rect(screen, WHITE, box, border_radius=15)
    pygame.draw.rect(screen, ORANGE_BTN, box, 4, border_radius=15)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        txt = font.render(line, True, BLACK)
        screen.blit(txt, (MENU_W//2 - txt.get_width()//2, box.y + 60 + i*35))
    pygame.display.flip()

# --- LOGIQUE ---

def set_screen_game():
    return pygame.display.set_mode((1000, 1000))

def set_screen_menu():
    return pygame.display.set_mode((MENU_W, MENU_H))


def run_academy(screen, font):
    """
    Lance le cycle complet d'apprentissage :
    1. Résolution des niveaux par BFS pour créer des données.
    2. Entraînement du réseau de neurones.
    3. Génération de la courbe de perte.
    """

    def update_progress(title, percent):
        # On réutilise ta fonction de chargement existante
        draw_loading_screen(screen, font, title, percent)

    # Initialisation de l'agent (le modèle IA)
    agent = Agent()
    dataset = []

    # 1. RÉCUPÉRATION DES NIVEAUX
    try:
        total_levels = list_levels()
    except:
        total_levels = 38  # Sécurité si la fonction n'est pas trouvée

    # 2. GÉNÉRATION DU DATASET
    # L'IA observe comment le BFS résout les niveaux pour apprendre
    for lvl in range(1, total_levels + 1):
        update_progress(f"Analyse du Niveau {lvl}...", int((lvl / total_levels) * 50))

        try:
            vehicles = load_level(lvl)
            board = BoardState(copy.deepcopy(vehicles))

            # Le BFS trouve le chemin le plus court
            path = SolverBFS.solve(board)

            if path:
                temp_board = BoardState(copy.deepcopy(vehicles))
                for i, (v_id, delta) in enumerate(path):
                    # Conversion du plateau en tenseur pour le réseau
                    state_tensor = state_to_tensor(temp_board)
                    # La cible est le nombre de coups restants
                    remaining_moves = float(len(path) - i)
                    dataset.append((state_tensor, remaining_moves))

                    # On simule le mouvement pour passer à l'état suivant
                    temp_board = temp_board.get_next_state(v_id, delta)
        except Exception as e:
            print(f"Erreur niveau {lvl}: {e}")
            continue

    # 3. PHASE D'ENTRAÎNEMENT
    if dataset:
        update_progress("Entraînement du cerveau...", 55)

        # On lance l'entraînement et on récupère l'historique des erreurs (Loss)
        history = agent.train_supervised(dataset, epochs=60, progress_callback=update_progress)

        # Sauvegarde du modèle (.pth)
        agent.save()

        # GÉNÉRATION DE LA COURBE
        try:
            plot_learning_curve(history)
            msg = "ENTRAÎNEMENT RÉUSSI !\nCourbe 'learning_curve.png' générée."
        except Exception as e:
            print(f"Erreur lors du graphique : {e}")
            msg = "MODÈLE ENTRAÎNÉ !\n(Erreur graphique : matplotlib absent ?)"

        draw_popup(screen, font, msg)
    else:
        draw_popup(screen, font, "ERREUR\nAucune donnée d'entraînement collectée.")

    time.sleep(2)

def watch_ai_play(level_number, font):
    agent = get_global_agent()
    if agent is None:
        draw_popup(pygame.display.get_surface(), font, "Modèle introuvable.\nLancez l'Académie !")
        time.sleep(2)
        return

    try:
        vehicles = load_level(level_number)
    except: return

    set_screen_game()
    game = RushHourGUI(vehicles)
    pygame.display.set_caption(f"IA - Niveau {level_number}")

    steps, max_steps = 0, 500
    running = True

    step_gen = None
    try:
        step_gen = solve_astar_complete(game.board_state, agent)
    except: step_gen = None

    while running and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                set_screen_menu()
                return

        if step_gen:
            try: v_id, delta = next(step_gen)
            except StopIteration: step_gen = None; continue
        else:
            path = SolverBFS.solve(game.board_state)
            if path: v_id, delta = path[0]
            else:
                draw_popup(pygame.display.get_surface(), font, "Impossible à résoudre")
                time.sleep(2)
                set_screen_menu()
                return

        next_board = game.board_state.get_next_state(v_id, delta)
        if next_board:
            game.board_state = next_board
            if v_id in game.g_vehicles:
                game.g_vehicles[v_id].logic = next_board.vehicles[v_id]
                game.g_vehicles[v_id].update_position_from_logic()
            game._draw_board()
            pygame.display.flip()
            time.sleep(0.15)
            steps += 1
            if game.board_state.is_solved(): running = False

    time.sleep(1)
    set_screen_menu()

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
        assets['red_car'] = get_tinted_image(assets['car'], (220, 40, 40))
        assets['btn_1'] = pygame.image.load(os.path.join("images", "button_1.png")).convert_alpha()
        assets['btn_2'] = pygame.image.load(os.path.join("images", "button_2.png")).convert_alpha()
    except Exception as e:
        print(f"Erreur Assets: {e}")

    f_title = pygame.font.SysFont("Segoe UI", 55, bold=True)
    f_btn = pygame.font.SysFont("Segoe UI", 24, bold=True)
    f_lvl = pygame.font.SysFont("Segoe UI", 20, bold=True)
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
            try:
                current_v = load_level(current_level)
                draw_mini_board(screen, current_v, px, py, p_size, assets)
            except: pass

            # Navigation
            btn_prev = draw_button_standard(screen, "<", px - 60, py + 150, 50, 50, f_btn, mouse_pos, COLOR_BLUE, (41, 128, 185))
            btn_next = draw_button_standard(screen, ">", px + p_size + 10, py + 150, 50, 50, f_btn, mouse_pos, COLOR_BLUE, (41, 128, 185))
            btn_grid = draw_custom_button(screen, f"NIVEAU {current_level} (CHANGER)", px - 5, py + p_size + 20, 360, 65, f_btn, mouse_pos, assets['btn_2'], COLOR_BLUE)

            # Boutons Actions
            b_play = draw_custom_button(screen, "JOUER", 140, 580, 240, 65, f_btn, mouse_pos, assets['btn_1'], GREEN_BTN)
            b_ia = draw_custom_button(screen, "IA SOLVER", 420, 580, 240, 65, f_btn, mouse_pos, assets['btn_1'], ORANGE_BTN)

            # Académie et Quitter
            b_acad = draw_custom_button(screen, "APPRENTISSAGE DE L'IA", 30, 700, 320, 65, f_btn, mouse_pos, assets['btn_2'], GRAY_DARK)
            b_quit = draw_custom_button(screen, "QUITTER", MENU_W - 220, 700, 190, 65, f_btn, mouse_pos, assets['btn_1'], RED_BTN)

            if click:
                if btn_prev.collidepoint(mouse_pos): current_level = current_level - 1 if current_level > 1 else max_levels
                elif btn_next.collidepoint(mouse_pos): current_level = current_level + 1 if current_level < max_levels else 1
                elif btn_grid.collidepoint(mouse_pos): show_selector = True
                elif b_acad.collidepoint(mouse_pos): run_academy(screen, f_btn)
                elif b_play.collidepoint(mouse_pos):
                    pygame.display.set_mode((1000, 1000))
                    RushHourGUI(load_level(current_level)).run()
                    screen = set_screen_menu()
                elif b_ia.collidepoint(mouse_pos) and model_exists:
                    watch_ai_play(current_level, f_btn)
                elif b_quit.collidepoint(mouse_pos): running = False
        else:
            overlay = pygame.Surface((MENU_W, MENU_H), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 230))
            screen.blit(overlay, (0, 0))
            for i in range(1, max_levels + 1):
                r, c = (i - 1) // 8, (i - 1) % 8
                bx, by = 60 + c * 85, 120 + r * 85
                rect = draw_custom_button(screen, str(i), bx, by, 75, 75, f_btn, mouse_pos, assets['btn_1'], BLUE_BTN if i != current_level else ORANGE_BTN)
                if click and rect.collidepoint(mouse_pos):
                    current_level = i
                    show_selector = False

            if click and draw_custom_button(screen, "RETOUR", MENU_W//2 - 120, 700, 240, 65, f_btn, mouse_pos, assets['btn_2'], RED_BTN).collidepoint(mouse_pos):
                show_selector = False

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main_menu()