import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
from collections import deque
from config import GRID_SIZE, RED_CAR_ID
from board import BoardState
from levels import load_level, list_levels


# --- 1. LE MODÈLE ---
class RushHourNet(nn.Module):
    def __init__(self, output_size):
        super(RushHourNet, self).__init__()
        # On garde une architecture capable
        self.fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, GRID_SIZE * GRID_SIZE)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


def state_to_tensor(board_state):
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_id = board_state.grid[r][c]
            if cell_id is not None:
                if cell_id == RED_CAR_ID:
                    matrix[r][c] = 1.0  # La voiture cible (X)
                else:
                    matrix[r][c] = 0.5  # Les autres
    return torch.tensor(matrix).unsqueeze(0)


# --- 2. LE PROFESSEUR (BFS) ---
class SolverBFS:
    @staticmethod
    def solve(start_board: BoardState):
        queue = deque([(start_board, [])])
        visited = set()
        visited.add(hash(start_board))

        # Sécurité pour ne pas chercher indéfiniment si pas de solution
        max_depth = 50000
        loops = 0

        while queue:
            current_board, path = queue.popleft()
            loops += 1
            if loops > max_depth:
                return None

            if current_board.is_solved():
                return path

            for v_id in current_board.vehicles:
                for delta in [-1, 1]:
                    if current_board.is_move_valid(v_id, delta):
                        next_board = current_board.get_next_state(v_id, delta)
                        h = hash(next_board)
                        if h not in visited:
                            visited.add(h)
                            queue.append((next_board, path + [(v_id, delta)]))
        return None


# --- 3. L'AGENT ---
class Agent:
    def __init__(self):
        # CORRECTION ICI : On prévoit assez de place pour A-Z
        # Z est la 26ème lettre. 26 * 2 directions = 52.
        # On met 60 pour être large et éviter le bug de la voiture X
        self.action_size = 60

        self.model = RushHourNet(self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = 0.0

    def encode_action(self, v_id, delta):
        """Transforme (Voiture, Direction) en un chiffre unique."""
        # On utilise le code ASCII pour avoir un ID unique par lettre
        # 'A' = 65. Donc ord(v) - 65 donne 0 pour A, 1 pour B... 23 pour X.
        if not v_id: return 0

        # On prend juste la première lettre de l'ID pour simplifier
        idx = ord(v_id[0].upper()) - ord('A')

        # Sécurité : si c'est un caractère bizarre, on le met à 0
        if idx < 0 or idx > 29:
            idx = 0

        # Pair = Avancer (+), Impair = Reculer (-)
        direction_bit = 0 if delta > 0 else 1
        action = (idx * 2) + direction_bit

        # Double sécurité pour ne pas dépasser la taille du réseau
        if action >= self.action_size:
            return 0

        return action

    def decode_action(self, action_idx):
        """Transforme un chiffre en (Voiture, Direction)."""
        idx = action_idx // 2
        direction_bit = action_idx % 2

        delta = 1 if direction_bit == 0 else -1
        # On retrouve la lettre : 0 -> A, 1 -> B...
        v_id = chr(ord('A') + idx)

        return v_id, delta

    def act(self, state_tensor):
        with torch.no_grad():
            logits = self.model(state_tensor)
        return torch.argmax(logits).item()

    def train_supervised(self, training_data, epochs=50):
        self.model.train()
        print(f"   -> Entraînement sur {len(training_data)} mouvements...")
        for epoch in range(epochs):
            random.shuffle(training_data)
            batch_size = 64
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                states = torch.cat([x[0] for x in batch])
                actions = torch.tensor([x[1] for x in batch], dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(states)
                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()

    # --- SAUVEGARDE ET CHARGEMENT ---
    def save(self, filename="rush_hour_model.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"Modèle sauvegardé avec succès dans {filename}")

    def load(self, filename="rush_hour_model.pth"):
        if os.path.exists(filename):
            try:
                self.model.load_state_dict(torch.load(filename))
                self.model.eval()
                print(f"Modèle chargé : {filename}")
                return True
            except:
                print("Erreur de chargement du modèle (taille incompatible ?).")
                return False
        return False


# --- 4. ENTRAINEMENT GLOBAL ---
def train_global_model():
    """Entraîne l'IA sur TOUS les niveaux disponibles."""
    print("\n--- GÉNÉRATION DE LA BASE DE DONNÉES ---")

    agent = Agent()  # Initialisation propre
    global_training_data = []

    num_levels = list_levels()
    print(f"Analyse de {num_levels} niveaux...")

    for i in range(1, num_levels + 1):
        try:
            vehicles = load_level(i)
            board = BoardState(copy.deepcopy(vehicles))

            # Le Professeur résout
            path = SolverBFS.solve(board)

            if path:
                # Data Augmentation : On apprend chaque étape du chemin gagnant
                temp_board = BoardState(copy.deepcopy(vehicles))
                for v_id, delta in path:
                    state = state_to_tensor(temp_board)
                    action = agent.encode_action(v_id, delta)

                    global_training_data.append((state, action))

                    # On avance le plateau virtuel
                    temp_board = temp_board.get_next_state(v_id, delta)
                print(f"[OK] Niveau {i} : {len(path)} coups ajoutés.")
            else:
                print(f"[X] Niveau {i} : Trop complexe ou impossible.")
        except Exception as e:
            print(f"[!] Erreur niveau {i}: {e}")

    print(f"\nBase de données terminée : {len(global_training_data)} exemples.")

    # L'IA apprend tout d'un coup
    print("Démarrage du Deep Learning (Supervisé)...")
    # On met plus d'epochs pour bien ancrer les connaissances
    agent.train_supervised(global_training_data, epochs=80)

    agent.save("rush_hour_model.pth")
    return agent


def get_trained_agent():
    """Charge l'agent sauvegardé."""
    agent = Agent()
    if agent.load("rush_hour_model.pth"):
        return agent
    return None