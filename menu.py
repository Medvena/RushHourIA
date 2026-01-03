import pygame
import time
import sys

# Imports
from levels import load_level, list_levels
from rush_hour_gui import RushHourGUI
from solver_ia import train_ai, state_to_tensor
from config import GRID_SIZE, PER_SQ

# --- CONFIG ---
WINDOW_SIZE = GRID_SIZE * PER_SQ
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
BLUE_HOVER = (100, 149, 237)
GRAY_TEXT = (50, 50, 50)
RED_BTN = (200, 50, 50)


def draw_button(screen, text, x, y, w, h, font, mouse_pos, color=BLUE, color_hover=BLUE_HOVER):
    """Dessine un bouton standard."""
    rect = pygame.Rect(x, y, w, h)
    actual_color = color_hover if rect.collidepoint(mouse_pos) else color

    pygame.draw.rect(screen, (50, 50, 50), (x + 2, y + 2, w, h), border_radius=8)
    pygame.draw.rect(screen, actual_color, rect, border_radius=8)
    pygame.draw.rect(screen, BLACK, rect, 2, border_radius=8)

    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

    return rect


def play_game_manual(level_to_play):
    """Jeu humain sur le niveau choisi."""
    try:
        vehicles = load_level(level_to_play)
    except Exception:
        print(f"Niveau {level_to_play} introuvable.")
        return

    game = RushHourGUI(vehicles)
    game.run()


def watch_ai_play(agent, level_number):
    """L'IA joue le niveau."""
    try:
        vehicles = load_level(level_number)
    except:
        return

    game = RushHourGUI(vehicles)
    agent.epsilon = 0.0
    state = state_to_tensor(game.board_state)

    done = False
    steps = 0
    pygame.display.set_caption(f"IA en action - Niveau {level_number}")

    while not done and steps < 150:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        action = agent.act(state)
        v_id, delta = agent.decode_action(action)
        next_board = game.board_state.get_next_state(v_id, delta)

        if next_board:
            game.board_state = next_board
            state = state_to_tensor(game.board_state)

            # Update visuel
            game.g_vehicles[v_id].logic = next_board.vehicles[v_id]
            game.g_vehicles[v_id].update_position_from_logic()

            if game.board_state.is_solved():
                print("Victoire de l'IA !")
                done = True

        game._draw_board()
        pygame.display.flip()
        time.sleep(0.3)
        steps += 1

    time.sleep(1)


def run_ai_demo(screen, font, level_number):
    """Entraîne l'IA sur le niveau demandé puis lance la démo."""
    screen.fill(WHITE)

    # Texte de chargement
    msg1 = font.render(f"Analyse du Niveau {level_number}...", True, BLACK)
    msg2 = font.render("(Calcul de la solution optimale)", True, GRAY_TEXT)

    screen.blit(msg1, (WINDOW_SIZE // 2 - msg1.get_width() // 2, WINDOW_SIZE // 2 - 20))
    screen.blit(msg2, (WINDOW_SIZE // 2 - msg2.get_width() // 2, WINDOW_SIZE // 2 + 20))
    pygame.display.flip()

    try:
        # On charge le niveau spécifique
        vehicles = load_level(level_number)

        # Le Professeur BFS trouve la solution pour CE niveau
        # et l'IA apprend cette solution par cœur.
        agent = train_ai(vehicles, episodes=1)

        # On regarde le résultat
        watch_ai_play(agent, level_number)

    except Exception as e:
        print(f"Erreur IA : {e}")


def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Menu Rush Hour")

    font_title = pygame.font.SysFont("Arial", 50, bold=True)
    font_btn = pygame.font.SysFont("Arial", 28)
    font_small = pygame.font.SysFont("Arial", 24)

    # État du menu
    current_level = 1
    max_levels = list_levels()  # Fonction importée de levels.py

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = True

        screen.fill(WHITE)

        # Titre
        title = font_title.render("RUSH HOUR", True, BLACK)
        screen.blit(title, (WINDOW_SIZE // 2 - title.get_width() // 2, 50))

        # --- SÉLECTEUR DE NIVEAU ---
        # On dessine "Niveau X" au milieu
        lvl_text = font_btn.render(f"Niveau {current_level}", True, BLACK)
        screen.blit(lvl_text, (WINDOW_SIZE // 2 - lvl_text.get_width() // 2, 140))

        # Bouton Moins (-)
        rect_minus = draw_button(screen, "-", 120, 130, 40, 40, font_btn, mouse_pos, GRAY_TEXT, (100, 100, 100))
        # Bouton Plus (+)
        rect_plus = draw_button(screen, "+", WINDOW_SIZE - 160, 130, 40, 40, font_btn, mouse_pos, GRAY_TEXT,
                                (100, 100, 100))

        # --- BOUTONS D'ACTION ---
        btn_w, btn_h = 240, 55
        center_x = (WINDOW_SIZE - btn_w) // 2

        rect_play = draw_button(screen, "Jouer (Humain)", center_x, 220, btn_w, btn_h, font_btn, mouse_pos)
        rect_ai = draw_button(screen, "IA résout ce niveau", center_x, 300, btn_w, btn_h, font_btn, mouse_pos,
                              (70, 180, 130), (100, 200, 150))  # Bouton Vert pour l'IA
        rect_quit = draw_button(screen, "Quitter", center_x, 380, btn_w, btn_h, font_btn, mouse_pos, RED_BTN,
                                (220, 80, 80))

        # LOGIQUE DES CLICS
        if click:
            if rect_minus.collidepoint(mouse_pos):
                if current_level > 1:
                    current_level -= 1

            elif rect_plus.collidepoint(mouse_pos):
                if current_level < max_levels:
                    current_level += 1

            elif rect_play.collidepoint(mouse_pos):
                play_game_manual(current_level)
                screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

            elif rect_ai.collidepoint(mouse_pos):
                # On passe le niveau choisi à la fonction
                run_ai_demo(screen, font_small, current_level)
                screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

            elif rect_quit.collidepoint(mouse_pos):
                running = False

        pygame.display.flip()

    pygame.quit()
    sys.exit()