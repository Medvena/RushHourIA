import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import time
from collections import deque
from config import GRID_SIZE, RED_CAR_ID
from board import BoardState
from levels import load_level


# ==========================================
# 1. LE CERVEAU (Réseau de Neurones)
# ==========================================
class RushHourNet(nn.Module):
    def __init__(self, output_size):
        super(RushHourNet, self).__init__()
        # Architecture CNN classique pour reconnaitre les formes
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(GRID_SIZE * GRID_SIZE * 128, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def state_to_tensor(board_state):
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_id = board_state.grid[r][c]
            if cell_id is not None:
                if cell_id == RED_CAR_ID:
                    matrix[r][c] = 1.0  # Cible
                else:
                    matrix[r][c] = 0.5  # Obstacle
    return torch.tensor(matrix).unsqueeze(0)


# ==========================================
# 2. LE PROFESSEUR (Solver BFS)
# ==========================================
class SolverBFS:
    """Trouve le chemin le plus court pour créer la 'Vérité Terrain'."""

    @staticmethod
    def solve(start_board: BoardState, max_depth=30000):
        queue = deque([(start_board, [])])
        visited = set()
        visited.add(hash(start_board))
        loops = 0

        while queue:
            current_board, path = queue.popleft()
            loops += 1
            if loops > max_depth: return None

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


# ==========================================
# 3. L'AGENT (L'IA)
# ==========================================
class Agent:
    def __init__(self):
        self.action_size = 60
        self.model = RushHourNet(self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.model_file = "rush_hour_brain.pth"

    def encode_action(self, v_id, delta):
        if not v_id: return 0
        idx = ord(v_id[0].upper()) - ord('A')
        if idx < 0 or idx > 29: idx = 0
        direction_bit = 0 if delta > 0 else 1
        return (idx * 2) + direction_bit

    def decode_action(self, action_idx):
        idx = action_idx // 2
        direction_bit = action_idx % 2
        delta = 1 if direction_bit == 0 else -1
        v_id = chr(ord('A') + idx)
        return v_id, delta

    def act(self, state_tensor):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state_tensor)
        return torch.argmax(logits).item()

    def train_supervised(self, training_data, epochs=30, progress_callback=None):
        self.model.train()
        random.shuffle(training_data)

        batch_size = 32
        for epoch in range(epochs):
            if progress_callback and epoch % 5 == 0:
                pct = 50 + int((epoch / epochs) * 50)
                progress_callback(f"Optimisation Neuronale ({epoch}/{epochs})", pct)

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                if not batch: continue

                states = torch.cat([x[0] for x in batch])
                actions = torch.tensor([x[1] for x in batch], dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(states)
                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()

    def save(self):
        torch.save(self.model.state_dict(), self.model_file)

    def load(self):
        if os.path.exists(self.model_file):
            try:
                self.model.load_state_dict(torch.load(self.model_file))
                self.model.eval()
                return True
            except:
                return False
        return False


# ==========================================
# 4. FONCTIONS GLOBALES
# ==========================================

def get_global_agent():
    agent = Agent()
    if agent.load(): return agent
    return None


def train_cumulative(max_level, progress_callback=None):
    """
    Apprend TOUS les niveaux de 1 jusqu'à max_level.
    """
    agent = Agent()
    # On peut charger l'ancien cerveau pour le compléter, ou repartir de zéro.
    # Ici, on repart de zéro pour être sûr que c'est propre.
    # if os.path.exists("rush_hour_brain.pth"): agent.load()

    dataset = []
    print(f"--- APPRENTISSAGE TOTAL (Niveaux 1 à {max_level}) ---")

    # PHASE 1 : GÉNÉRATION DES DONNÉES (Le Professeur travaille)
    for lvl in range(1, max_level + 1):
        if progress_callback:
            pct = int((lvl / max_level) * 50)
            progress_callback(f"Analyse & Résolution Niveau {lvl}...", pct)

        try:
            vehicles = load_level(lvl)
            board = BoardState(copy.deepcopy(vehicles))

            # Le Solver résout le niveau
            path = SolverBFS.solve(board)

            if path:
                # On enregistre chaque étape : Situation -> Mouvement
                temp_board = BoardState(copy.deepcopy(vehicles))
                for v_id, delta in path:
                    state = state_to_tensor(temp_board)
                    action = agent.encode_action(v_id, delta)
                    dataset.append((state, action))
                    temp_board = temp_board.get_next_state(v_id, delta)
        except Exception as e:
            print(f"Niveau {lvl} ignoré (erreur ou introuvable).")

    # PHASE 2 : ENTRAÎNEMENT DU MODÈLE
    if dataset:
        print(f"Entraînement sur {len(dataset)} situations...")
        # 60 Epochs pour être sûr que ça rentre bien dans le crâne
        agent.train_supervised(dataset, epochs=60, progress_callback=progress_callback)
        agent.save()
        if progress_callback: progress_callback("Terminé !", 100)
        return agent
    else:
        return None