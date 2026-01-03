import pygame
import time
import sys

# Imports
from levels import load_level, list_levels
from rush_hour_gui import RushHourGUI
from config import GRID_SIZE, PER_SQ
from solver_ia import train_global_model, get_trained_agent, state_to_tensor
import os

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


def watch_ai_play_global(level_number):
    """Charge le modèle global et tente de résoudre."""
    agent = get_trained_agent()
    if agent is None:
        print("Erreur : Aucun modèle entraîné trouvé ! Cliquez sur 'Entraîner' d'abord.")
        return

    try:
        vehicles = load_level(level_number)
    except:
        return

    game = RushHourGUI(vehicles)
    agent.epsilon = 0.0
    state = state_to_tensor(game.board_state)

    done = False
    steps = 0
    pygame.display.set_caption(f"IA (Modèle Global) - Niveau {level_number}")

    while not done and steps < 150:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # L'IA utilise son cerveau pré-entraîné
        action_idx = agent.act(state)
        v_id, delta = agent.decode_action(action_idx)

        # Vérif si la voiture existe dans ce niveau
        if v_id not in game.board_state.vehicles:
            # L'IA hallucine une voiture qui n'est pas là (ça peut arriver sur les nouveaux niveaux)
            print(f"IA essaie de bouger {v_id} (n'existe pas ici)")
            steps += 1
            continue

        next_board = game.board_state.get_next_state(v_id, delta)

        if next_board:
            game.board_state = next_board
            state = state_to_tensor(game.board_state)

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


def run_global_training(screen, font):
    """Lance l'entraînement sur TOUS les niveaux."""
    screen.fill(WHITE)
    msg = font.render("Entraînement Global en cours...", True, BLACK)
    msg2 = font.render("(Ceci peut prendre quelques secondes)", True, GRAY_TEXT)
    screen.blit(msg, (50, WINDOW_SIZE // 2 - 20))
    screen.blit(msg2, (50, WINDOW_SIZE // 2 + 20))
    pygame.display.flip()

    # Appel de la fonction lourde
    train_global_model()


def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Projet IA - Rush Hour")

    font_title = pygame.font.SysFont("Arial", 40, bold=True)
    font_btn = pygame.font.SysFont("Arial", 24)

    current_level = 1
    max_levels = list_levels()

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: click = True

        screen.fill(WHITE)

        # Titre
        title = font_title.render("RUSH HOUR - IA", True, BLACK)
        screen.blit(title, (WINDOW_SIZE // 2 - title.get_width() // 2, 30))

        # --- ETAT DU MODÈLE ---
        # On vérifie si le fichier existe pour afficher un indicateur
        model_exists = os.path.exists("rush_hour_model.pth")
        status_color = (50, 200, 50) if model_exists else (200, 50, 50)
        status_text = "Modèle : PRÊT" if model_exists else "Modèle : VIDE"
        pygame.draw.circle(screen, status_color, (30, 30), 10)

        # --- GROS BOUTON ENTRAINEMENT ---
        rect_train = draw_button(screen, "GÉNÉRER & ENTRAÎNER LE MODÈLE", 40, 90, WINDOW_SIZE - 80, 50, font_btn,
                                 mouse_pos, (255, 140, 0), (255, 165, 0))

        # --- SELECTEUR ---
        lvl_text = font_btn.render(f"Niveau {current_level}", True, BLACK)
        screen.blit(lvl_text, (WINDOW_SIZE // 2 - lvl_text.get_width() // 2, 170))
        rect_minus = draw_button(screen, "-", 120, 160, 40, 40, font_btn, mouse_pos, GRAY_TEXT, (100, 100, 100))
        rect_plus = draw_button(screen, "+", WINDOW_SIZE - 160, 160, 40, 40, font_btn, mouse_pos, GRAY_TEXT,
                                (100, 100, 100))

        # --- ACTIONS ---
        btn_w = 220
        rect_play = draw_button(screen, "Jouer (Humain)", (WINDOW_SIZE - btn_w) // 2, 240, btn_w, 50, font_btn,
                                mouse_pos)

        # Bouton IA désactivé (gris) si pas de modèle
        ia_color = (70, 180, 130) if model_exists else (200, 200, 200)
        rect_ai = draw_button(screen, "IA (Modèle Global)", (WINDOW_SIZE - btn_w) // 2, 310, btn_w, 50, font_btn,
                              mouse_pos, ia_color, ia_color)

        rect_quit = draw_button(screen, "Quitter", (WINDOW_SIZE - btn_w) // 2, 400, btn_w, 50, font_btn, mouse_pos,
                                (200, 50, 50), (220, 80, 80))

        if click:
            if rect_train.collidepoint(mouse_pos):
                run_global_training(screen, font_btn)
            elif rect_minus.collidepoint(mouse_pos) and current_level > 1:
                current_level -= 1
            elif rect_plus.collidepoint(mouse_pos) and current_level < max_levels:
                current_level += 1
            elif rect_play.collidepoint(mouse_pos):
                play_game_manual(current_level)
                screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            elif rect_ai.collidepoint(mouse_pos) and model_exists:
                watch_ai_play_global(current_level)
                screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            elif rect_quit.collidepoint(mouse_pos):
                running = False

        pygame.display.flip()

    pygame.quit()
    sys.exit()