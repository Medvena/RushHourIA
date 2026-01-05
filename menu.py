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

# On importe les fonctions de l'IA
from solver_ia import train_generalist_model, get_trained_agent, state_to_tensor

# --- DIMENSIONS ---
# Taille du Jeu (Carré 6x6)
GAME_SIZE = GRID_SIZE * PER_SQ

# Taille du Menu (Plus grand pour être joli)
MENU_W = 800
MENU_H = 800

# --- COULEURS & DESIGN ---
WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
BG_COLOR = (240, 240, 245)  # Gris doux
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

GRAY_TEXT_LIGHT = (100, 100, 100) # Gris texte
WARNING_TEXT = (200, 100, 50)     # Orange foncé pour l'avertissement


def draw_button(screen, text, x, y, w, h, font, mouse_pos, color, color_hover, border_radius=12):
    """Dessine un bouton moderne avec effet de survol."""
    rect = pygame.Rect(x, y, w, h)
    is_hover = rect.collidepoint(mouse_pos)
    actual_color = color_hover if is_hover else color

    # Ombre
    pygame.draw.rect(screen, (200, 200, 200), (x + 4, y + 4, w, h), border_radius=border_radius)
    # Corps
    pygame.draw.rect(screen, actual_color, rect, border_radius=border_radius)
    # Bordure blanche interne
    pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=border_radius)

    # Texte centré
    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

    return rect, is_hover


def draw_mini_board(screen, vehicles, x, y, size):
    """Dessine la prévisualisation du niveau."""
    rect_board = pygame.Rect(x, y, size, size)
    pygame.draw.rect(screen, PREVIEW_BG, rect_board, border_radius=8)
    pygame.draw.rect(screen, GRAY_DARK, rect_board, 2, border_radius=8)

    cell_s = size / 6

    # Grille
    for i in range(1, 6):
        pygame.draw.line(screen, (240, 240, 240), (x, y + i * cell_s), (x + size, y + i * cell_s))
        pygame.draw.line(screen, (240, 240, 240), (x + i * cell_s, y), (x + i * cell_s, y + size))

    # Indicateur Sortie
    exit_y = y + 2 * cell_s
    pygame.draw.line(screen, (46, 204, 113), (x + size - 2, exit_y), (x + size - 2, exit_y + cell_s), 4)

    # Véhicules
    for v in vehicles:
        vx = x + v.col * cell_s
        vy = y + v.row * cell_s

        # CORRECTION : v.size au lieu de v.length
        vw = v.size * cell_s if v.orientation == 'h' else cell_s
        vh = v.size * cell_s if v.orientation == 'v' else cell_s

        margin = 3
        v_rect = pygame.Rect(vx + margin, vy + margin, vw - 2 * margin, vh - 2 * margin)

        # Rouge pour la voiture cible, Bleu-Gris pour les autres
        c = (231, 76, 60) if v.id == RED_CAR_ID else (52, 73, 94)

        pygame.draw.rect(screen, c, v_rect, border_radius=4)


def draw_popup(screen, font, text):
    """Overlay noir."""
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


# --- LOGIQUE IA & JEU ---

def set_screen_game():
    """Passe en résolution JEU."""
    return pygame.display.set_mode((GAME_SIZE, GAME_SIZE))


def set_screen_menu():
    """Passe en résolution MENU."""
    return pygame.display.set_mode((MENU_W, MENU_H))


def run_academy(screen, font):
    draw_popup(screen, font,
               "ACADÉMIE EN COURS...\nGénération de niveaux & Entraînement.\nCela prend environ 1 minute.")
    # On lance l'entrainement
    train_generalist_model(num_levels=300)


def watch_ai_generalist(level_number, font):
    """Lance le mode IA avec redimensionnement et indicateur de chargement."""
    agent = get_trained_agent()

    # On utilise la fenêtre actuelle (Menu) pour afficher l'erreur
    if agent is None:
        screen = pygame.display.get_surface()
        draw_popup(screen, font, "Aucun modèle trouvé !\nLancez l'Académie d'abord.")
        time.sleep(2)
        return

    try:
        vehicles = load_level(level_number)
    except:
        return

    # 1. On passe en mode JEU
    screen = set_screen_game()  # On récupère l'objet screen redimensionné
    game = RushHourGUI(vehicles)

    # --- DESSIN DU PLATEAU ---
    game._draw_board()

    # --- INDICATEUR VISUEL DE RÉFLEXION ---
    # On crée un voile noir semi-transparent
    overlay = pygame.Surface((GAME_SIZE, GAME_SIZE))
    overlay.set_alpha(180)  # Transparence (0-255)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    # On ajoute le texte
    font_loading = pygame.font.SysFont("Segoe UI", 40, bold=True)
    font_sub_loading = pygame.font.SysFont("Segoe UI", 20)

    txt_main = font_loading.render("L'IA RÉFLÉCHIT...", True, WHITE)
    txt_rect = txt_main.get_rect(center=(GAME_SIZE // 2, GAME_SIZE // 2 - 20))

    txt_sub = font_sub_loading.render("Analyse des probabilités & Recherche...", True, (200, 200, 200))
    sub_rect = txt_sub.get_rect(center=(GAME_SIZE // 2, GAME_SIZE // 2 + 30))

    screen.blit(txt_main, txt_rect)
    screen.blit(txt_sub, sub_rect)

    pygame.display.set_caption("L'IA réfléchit...")
    pygame.display.flip()  # On force l'affichage AVANT le calcul

    # 2. CALCUL (Bloquant)
    solution_path = agent.solve_with_ai(game.board_state)

    if not solution_path:
        # On repasse en mode MENU pour afficher l'erreur
        screen = set_screen_menu()
        draw_popup(screen, font, "Échec : Niveau trop complexe\npour l'IA actuelle.")
        time.sleep(2)
        return

    # 3. Exécution visuelle
    pygame.display.set_caption(f"IA - Résolution Niveau {level_number}")

    # On redessine le plateau propre avant de commencer l'animation
    game._draw_board()
    pygame.display.flip()
    time.sleep(0.5)

    for v_id, delta in solution_path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
        time.sleep(0.2)

    time.sleep(1)
    # Retour au menu à la fin
    set_screen_menu()


def play_game_manual(level):
    try:
        vehicles = load_level(level)
        # Redimensionnement
        set_screen_game()
        game = RushHourGUI(vehicles)
        game.run()
        # Retour
        set_screen_menu()
    except:
        set_screen_menu()


# --- MENU PRINCIPAL ---

def main_menu():
    pygame.init()
    # On démarre en résolution MENU
    screen = set_screen_menu()
    pygame.display.set_caption("RUSH HOUR - INTELLIGENCE ARTIFICIELLE")

    # Polices (Plus grandes pour le grand écran)
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
        model_exists = os.path.exists("rush_hour_hybrid.pth")

        # Chargement véhicules pour preview
        try:
            preview_vehicles = load_level(current_level)
        except:
            preview_vehicles = []

        mouse_pos = pygame.mouse.get_pos()
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click = True

        screen.fill(BG_COLOR)

        # --- HEADER ---
        title = font_title.render("RUSH HOUR IA", True, BLACK)
        title_rect = title.get_rect(center=(MENU_W // 2, 60))
        screen.blit(title, title_rect)

        # Indicateur Modèle
        status_color = GREEN_BTN if model_exists else RED_BTN
        status_text = "IA : ONLINE" if model_exists else "IA : OFFLINE"
        pygame.draw.rect(screen, status_color, (MENU_W - 165, 31, 140, 34), border_radius=10)
        lbl_status = font_btn.render(status_text, True, WHITE)
        status_rect = lbl_status.get_rect(center=(MENU_W - 95, 47))
        screen.blit(lbl_status, status_rect)

        if not show_level_selector:
            # --- VUE PRINCIPALE ---

            # PREVIEW (Plus grande)
            preview_size = 350
            preview_x = (MENU_W - preview_size) // 2
            preview_y = 150

            # Nom du niveau
            lbl_lvl = font_sub.render(f"NIVEAU SÉLECTIONNÉ : {current_level}", True, GRAY_DARK)
            screen.blit(lbl_lvl, (MENU_W // 2 - lbl_lvl.get_width() // 2, preview_y - 40))

            # Grille
            draw_mini_board(screen, preview_vehicles, preview_x, preview_y, preview_size)

            # Bouton Changer Niveau
            btn_change, _ = draw_button(screen, "CHANGER LE NIVEAU", preview_x, preview_y + preview_size + 20,
                                        preview_size, 50, font_btn, mouse_pos, BLUE_BTN, BLUE_HOVER)

            # --- ACTIONS ---
            action_y = 600
            btn_w = 260
            gap = 40
            total_w = btn_w * 2 + gap
            start_x = (MENU_W - total_w) // 2

            # JOUER
            btn_play, _ = draw_button(screen, "JOUER (HUMAIN)", start_x, action_y, btn_w, 60, font_btn, mouse_pos,
                                      GREEN_BTN, GREEN_HOVER)

            # IA
            ia_col = ORANGE_BTN if model_exists else GRAY_LIGHT
            ia_hov = ORANGE_HOVER if model_exists else GRAY_LIGHT
            btn_ia, _ = draw_button(screen, "RÉSOUDRE (IA)", start_x + btn_w + gap, action_y, btn_w, 60, font_btn,
                                    mouse_pos, ia_col, ia_hov)

            # --- ACADÉMIE ---
            academy_y = MENU_H - 100
            btn_academy, _ = draw_button(screen, "GÉNÉRER & ENTRAÎNER (ACADÉMIE)", 30, MENU_H - 80, 300, 30, font_small,
                                         mouse_pos, GRAY_DARK, (80, 80, 80))

            # Textes explicatifs en dessous
            txt_info_1 = "Requis si l'IA est OFFLINE (Génère le modèle)."
            txt_info_2 = "(Attention : L'opération prend un peu de temps)"

            # Sinon : font_mini = pygame.font.SysFont("Segoe UI", 14)
            lbl_info_1 = font_mini.render(txt_info_1, True, GRAY_TEXT_LIGHT)
            lbl_info_2 = font_mini.render(txt_info_2, True, WARNING_TEXT)

            screen.blit(lbl_info_1, (35, academy_y + 55))
            screen.blit(lbl_info_2, (35, academy_y + 70))

            # QUITTER
            btn_quit, _ = draw_button(screen, "QUITTER", MENU_W - 150, MENU_H - 50, 120, 30, font_small, mouse_pos,
                                      RED_BTN, RED_HOVER)

            # Clics
            if click:
                if btn_change.collidepoint(mouse_pos):
                    show_level_selector = True
                elif btn_play.collidepoint(mouse_pos):
                    play_game_manual(current_level)
                    # Au retour, on s'assure d'être en taille Menu
                    screen = set_screen_menu()
                elif btn_ia.collidepoint(mouse_pos) and model_exists:
                    watch_ai_generalist(current_level, font_sub)
                    screen = set_screen_menu()
                elif btn_academy.collidepoint(mouse_pos):
                    run_academy(screen, font_sub)
                elif btn_quit.collidepoint(mouse_pos):
                    running = False

        else:
            # --- VUE SÉLECTEUR ---
            overlay = pygame.Surface((MENU_W, MENU_H))
            overlay.set_alpha(220)
            overlay.fill(WHITE)
            screen.blit(overlay, (0, 0))

            lbl_sel = font_title.render("SÉLECTIONNER UN NIVEAU", True, BLACK)
            screen.blit(lbl_sel, (MENU_W // 2 - lbl_sel.get_width() // 2, 50))

            # Grille de niveaux
            cols = 8
            btn_s = 70
            margin = 20

            # Centrage de la grille
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

            # Retour
            btn_back, _ = draw_button(screen, "RETOUR AU MENU", MENU_W // 2 - 100, MENU_H - 80, 220, 50, font_btn,
                                      mouse_pos, RED_BTN, RED_HOVER)
            if click and btn_back.collidepoint(mouse_pos):
                show_level_selector = False

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_menu()