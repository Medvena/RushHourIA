import pygame
import time
import sys
import os
import math

# Imports du projet
from levels import load_level, list_levels
from rush_hour_gui import RushHourGUI
from config import GRID_SIZE, PER_SQ, RED_CAR_ID
from board import BoardState

# On importe les fonctions supervisées
from solver_ia import train_cumulative, get_global_agent, state_to_tensor, SolverBFS

# --- DIMENSIONS ---
GAME_SIZE = GRID_SIZE * PER_SQ
MENU_W = 800
MENU_H = 800

# --- COULEURS ---
WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
BG_COLOR = (240, 240, 245)
PREVIEW_BG = (255, 255, 255)

BLUE_BTN = (52, 152, 219)
BLUE_HOVER = (41, 128, 185)
ORANGE_BTN = (230, 126, 34)
ORANGE_HOVER = (211, 84, 0)
GREEN_BTN = (46, 204, 113)
GREEN_HOVER = (39, 174, 96)
RED_BTN = (231, 76, 60)
RED_HOVER = (192, 57, 43)

GRAY_DARK = (50, 50, 50)
GRAY_LIGHT = (200, 200, 200)
GRAY_TEXT_LIGHT = (100, 100, 100)
WARNING_TEXT = (200, 100, 50)


def draw_button(screen, text, x, y, w, h, font, mouse_pos, color, color_hover, border_radius=12):
    rect = pygame.Rect(x, y, w, h)
    is_hover = rect.collidepoint(mouse_pos)
    actual_color = color_hover if is_hover else color
    pygame.draw.rect(screen, (200, 200, 200), (x + 4, y + 4, w, h), border_radius=border_radius)
    pygame.draw.rect(screen, actual_color, rect, border_radius=border_radius)
    pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=border_radius)
    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)
    return rect, is_hover


def draw_mini_board(screen, vehicles, x, y, size):
    rect_board = pygame.Rect(x, y, size, size)
    pygame.draw.rect(screen, PREVIEW_BG, rect_board, border_radius=8)
    pygame.draw.rect(screen, GRAY_DARK, rect_board, 2, border_radius=8)
    cell_s = size / 6
    for i in range(1, 6):
        pygame.draw.line(screen, (240, 240, 240), (x, y + i * cell_s), (x + size, y + i * cell_s))
        pygame.draw.line(screen, (240, 240, 240), (x + i * cell_s, y), (x + i * cell_s, y + size))
    exit_y = y + 2 * cell_s
    pygame.draw.line(screen, (46, 204, 113), (x + size - 2, exit_y), (x + size - 2, exit_y + cell_s), 4)
    for v in vehicles:
        vx = x + v.col * cell_s
        vy = y + v.row * cell_s
        vw = v.size * cell_s if v.orientation == 'h' else cell_s
        vh = v.size * cell_s if v.orientation == 'v' else cell_s
        margin = 3
        v_rect = pygame.Rect(vx + margin, vy + margin, vw - 2 * margin, vh - 2 * margin)
        c = (231, 76, 60) if v.id == RED_CAR_ID else (52, 73, 94)
        pygame.draw.rect(screen, c, v_rect, border_radius=4)


def draw_popup(screen, font, text):
    overlay = pygame.Surface((MENU_W, MENU_H))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    box_w, box_h = 500, 200
    box_x = (MENU_W - box_w) // 2
    box_y = (MENU_H - box_h) // 2
    pygame.draw.rect(screen, WHITE, (box_x, box_y, box_w, box_h), border_radius=15)
    pygame.draw.rect(screen, ORANGE_BTN, (box_x, box_y, box_w, box_h), 4, border_radius=15)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        txt = font.render(line, True, BLACK)
        screen.blit(txt, (MENU_W // 2 - txt.get_width() // 2, box_y + 60 + i * 35))
    pygame.display.flip()


def draw_loading_screen(screen, font, title, percent):
    overlay = pygame.Surface((MENU_W, MENU_H))
    overlay.set_alpha(255)
    overlay.fill(BG_COLOR)
    screen.blit(overlay, (0, 0))
    center_x = MENU_W // 2
    center_y = MENU_H // 2
    font_big = pygame.font.SysFont("Segoe UI", 30, bold=True)
    txt_title = font_big.render(title, True, BLACK)
    screen.blit(txt_title, (center_x - txt_title.get_width() // 2, center_y - 80))
    bar_w, bar_h = 500, 40
    pygame.draw.rect(screen, (200, 200, 200), (center_x - bar_w // 2, center_y, bar_w, bar_h), border_radius=20)
    fill_w = int(bar_w * (percent / 100))
    if fill_w > 0:
        pygame.draw.rect(screen, ORANGE_BTN, (center_x - bar_w // 2, center_y, fill_w, bar_h), border_radius=20)
    txt_pct = font.render(f"{percent}%", True, WHITE)
    screen.blit(txt_pct, (center_x - txt_pct.get_width() // 2, center_y + 10))
    pygame.display.flip()
    pygame.event.pump()


# --- LOGIQUE ---
def set_screen_game():
    return pygame.display.set_mode((GAME_SIZE, GAME_SIZE))


def set_screen_menu():
    return pygame.display.set_mode((MENU_W, MENU_H))


def run_academy(screen, font):
    def update_progress(title, percent):
        draw_loading_screen(screen, font, title, percent)

    # ON APPREND TOUS LES NIVEAUX SANS EXCEPTION
    total_levels = list_levels()
    train_cumulative(max_level=total_levels, progress_callback=update_progress)

    draw_popup(screen, font, "MODÈLE ENTRAÎNÉ !\nL'IA connaît tous les niveaux.")
    time.sleep(2)


def watch_ai_play(level_number, font):
    agent = get_global_agent()
    if agent is None:
        screen = pygame.display.get_surface()
        draw_popup(screen, font, "Modèle vide.\nLancez l'Académie !")
        time.sleep(2)
        return

    try:
        vehicles = load_level(level_number)
    except:
        return

    set_screen_game()
    game = RushHourGUI(vehicles)
    pygame.display.set_caption(f"IA - Niveau {level_number}")

    done = False
    steps = 0
    max_steps = 150  # On laisse de la marge

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                set_screen_menu()
                return

        state = state_to_tensor(game.board_state)
        action_idx = agent.act(state)
        v_id, delta = agent.decode_action(action_idx)

        # SI L'IA SE TROMPE (Coup Invalide ou Blocage)
        if not game.board_state.is_move_valid(v_id, delta) or steps > max_steps:
            print(">>> ERREUR IA -> AUTO-RÉPARATION <<<")

            # Affichage "Secours"
            font_big = pygame.font.SysFont("Segoe UI", 30, bold=True)
            overlay = pygame.Surface((GAME_SIZE, GAME_SIZE))
            overlay.set_alpha(200)
            overlay.fill(BLACK)
            screen = pygame.display.get_surface()
            screen.blit(overlay, (0, 0))
            txt = font_big.render("AUTO-RÉPARATION...", True, ORANGE_BTN)
            screen.blit(txt, (GAME_SIZE // 2 - txt.get_width() // 2, GAME_SIZE // 2))
            pygame.display.flip()

            # Le Solver prend le relais
            path = SolverBFS.solve(game.board_state)
            if path:
                for s_vid, s_delta in path:
                    nxt = game.board_state.get_next_state(s_vid, s_delta)
                    game.board_state = nxt
                    if s_vid in game.g_vehicles:
                        game.g_vehicles[s_vid].logic = nxt.vehicles[s_vid]
                        game.g_vehicles[s_vid].update_position_from_logic()
                    game._draw_board()
                    pygame.display.flip()
                    time.sleep(0.15)
                done = True
            else:
                set_screen_menu()
                return

        else:
            # Coup Valide
            next_board = game.board_state.get_next_state(v_id, delta)
            if next_board:
                game.board_state = next_board
                if v_id in game.g_vehicles:
                    game.g_vehicles[v_id].logic = next_board.vehicles[v_id]
                    game.g_vehicles[v_id].update_position_from_logic()
                if game.board_state.is_solved():
                    done = True

            game._draw_board()
            pygame.display.flip()
            time.sleep(0.3)
            steps += 1

    time.sleep(1)
    set_screen_menu()


def play_game_manual(level):
    try:
        vehicles = load_level(level)
        set_screen_game()
        game = RushHourGUI(vehicles)
        game.run()
        set_screen_menu()
    except:
        set_screen_menu()


# --- MENU ---
def main_menu():
    pygame.init()
    screen = set_screen_menu()
    pygame.display.set_caption("RUSH HOUR - IA SUPERVISÉE")

    font_title = pygame.font.SysFont("Segoe UI", 60, bold=True)
    font_sub = pygame.font.SysFont("Segoe UI", 30)
    font_btn = pygame.font.SysFont("Segoe UI", 24, bold=True)
    font_small = pygame.font.SysFont("Segoe UI", 18)
    font_mini = pygame.font.SysFont("Segoe UI", 14)

    current_level = 1
    max_levels = list_levels()
    show_level_selector = False

    running = True
    while running:
        model_exists = os.path.exists("rush_hour_brain.pth")

        try:
            preview_vehicles = load_level(current_level)
        except:
            preview_vehicles = []

        mouse_pos = pygame.mouse.get_pos()
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: click = True

        screen.fill(BG_COLOR)

        title = font_title.render("RUSH HOUR IA", True, BLACK)
        title_rect = title.get_rect(center=(MENU_W // 2, 60))
        screen.blit(title, title_rect)

        status_color = GREEN_BTN if model_exists else RED_BTN
        status_text = "IA ONLINE" if model_exists else "IA OFFLINE"
        status_w = 100
        pygame.draw.rect(screen, status_color, (MENU_W - status_w - 50, 30, status_w + 40, 34), border_radius=17)
        lbl_status = font_btn.render(status_text, True, WHITE)
        status_rect = lbl_status.get_rect(center=(MENU_W - (status_w // 2) - 30, 47))
        screen.blit(lbl_status, status_rect)

        if not show_level_selector:
            preview_size = 350
            preview_x = (MENU_W - preview_size) // 2
            preview_y = 150

            lbl_lvl = font_sub.render(f"NIVEAU {current_level}", True, GRAY_DARK)
            screen.blit(lbl_lvl, (MENU_W // 2 - lbl_lvl.get_width() // 2, preview_y - 40))
            draw_mini_board(screen, preview_vehicles, preview_x, preview_y, preview_size)
            btn_change, _ = draw_button(screen, "CHANGER LE NIVEAU", preview_x, preview_y + preview_size + 20,
                                        preview_size, 50, font_btn, mouse_pos, BLUE_BTN, BLUE_HOVER)

            action_y = 600
            btn_w = 260
            gap = 40
            total_w = btn_w * 2 + gap
            start_x = (MENU_W - total_w) // 2

            btn_play, _ = draw_button(screen, "JOUER (HUMAIN)", start_x, action_y, btn_w, 60, font_btn, mouse_pos,
                                      GREEN_BTN, GREEN_HOVER)

            ia_col = ORANGE_BTN if model_exists else GRAY_LIGHT
            ia_hov = ORANGE_HOVER if model_exists else GRAY_LIGHT
            btn_ia, _ = draw_button(screen, "RÉSOUDRE (IA)", start_x + btn_w + gap, action_y, btn_w, 60, font_btn,
                                    mouse_pos, ia_col, ia_hov)

            # BOUTON ACADEMIE (VERSION SIMPLE)
            academy_y = MENU_H - 80
            btn_academy, _ = draw_button(screen, "APPRENTISSAGE DE L'IA", 30, academy_y, 350, 35, font_small,
                                         mouse_pos, GRAY_DARK, (80, 80, 80))

            txt_info = f"Apprend automatiquement les {max_levels} niveaux."
            lbl_info = font_mini.render(txt_info, True, GRAY_TEXT_LIGHT)
            screen.blit(lbl_info, (35, academy_y + 40))

            btn_quit, _ = draw_button(screen, "QUITTER", MENU_W - 150, MENU_H - 60, 120, 30, font_small, mouse_pos,
                                      RED_BTN, RED_HOVER)

            if click:
                if btn_change.collidepoint(mouse_pos):
                    show_level_selector = True
                elif btn_play.collidepoint(mouse_pos):
                    play_game_manual(current_level)
                    screen = set_screen_menu()
                elif btn_ia.collidepoint(mouse_pos) and model_exists:
                    watch_ai_play(current_level, font_sub)
                    screen = set_screen_menu()
                elif btn_academy.collidepoint(mouse_pos):
                    run_academy(screen, font_sub)
                elif btn_quit.collidepoint(mouse_pos):
                    running = False
        else:
            # SELECTEUR
            overlay = pygame.Surface((MENU_W, MENU_H))
            overlay.set_alpha(220)
            overlay.fill(WHITE)
            screen.blit(overlay, (0, 0))
            lbl_sel = font_title.render("SÉLECTIONNER", True, BLACK)
            screen.blit(lbl_sel, (MENU_W // 2 - lbl_sel.get_width() // 2, 50))
            cols = 8
            btn_s = 70
            margin = 20
            grid_w = cols * (btn_s + margin) - margin
            start_x = (MENU_W - grid_w) // 2
            start_y = 150
            for i in range(1, max_levels + 1):
                r = (i - 1) // cols
                c = (i - 1) % cols
                bx = start_x + c * (btn_s + margin)
                by = start_y + r * (btn_s + margin)
                col = BLUE_BTN if i == current_level else GRAY_LIGHT
                hov = BLUE_HOVER if i == current_level else (180, 180, 180)
                rect, _ = draw_button(screen, str(i), bx, by, btn_s, btn_s, font_btn, mouse_pos, col, hov,
                                      border_radius=10)
                if click and rect.collidepoint(mouse_pos):
                    current_level = i
                    show_level_selector = False
            btn_back, _ = draw_button(screen, "RETOUR", MENU_W // 2 - 100, MENU_H - 80, 200, 50, font_btn, mouse_pos,
                                      RED_BTN, RED_HOVER)
            if click and btn_back.collidepoint(mouse_pos): show_level_selector = False
        pygame.display.flip()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_menu()