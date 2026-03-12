import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
from collections import deque
from config import GRID_SIZE
from board import BoardState
from levels import load_level
import heapq


# ==========================================
# 1. LE MODÈLE NEURONAL (PyTorch)
# ==========================================
class RushHourNet(nn.Module):
    def __init__(self):
        super(RushHourNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(GRID_SIZE * GRID_SIZE * 128, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def state_to_tensor(board_state):
    """Transforme le plateau en image mathématique pour l'IA."""
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_id = board_state.grid[r][c]
            if cell_id is not None:
                matrix[r][c] = ord(cell_id) / 100.0
    return torch.tensor(matrix).unsqueeze(0)


# ==========================================
# 2. LE PROFESSEUR (Génère le chemin idéal)
# ==========================================
class SolverBFS:
    @staticmethod
    def solve(start_board: BoardState, max_depth=30000):
        queue = deque([(start_board, [])])
        visited = set([hash(start_board)])

        while queue:
            board, path = queue.popleft()
            if len(path) > max_depth: return None
            if board.is_solved(): return path

            for v_id, delta in board.get_possible_moves():
                nxt = board.get_next_state(v_id, delta)
                h = hash(nxt)
                if h not in visited:
                    visited.add(h)
                    queue.append((nxt, path + [(v_id, delta)]))
        return None


# ==========================================
# 3. L'AGENT (Apprentissage et Sauvegarde)
# ==========================================
class Agent:
    def __init__(self):
        self.model = RushHourNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model_file = "rush_hour_brain.pth"

    def heuristic(self, board):
        self.model.eval()
        with torch.no_grad():
            state = state_to_tensor(board)
            return self.model(state).item()

    def train_supervised(self, dataset, epochs=100, progress_callback=None):
        self.model.train()
        batch_size = 32

        for epoch in range(epochs):
            random.shuffle(dataset)
            if progress_callback and epoch % 5 == 0:
                pct = 50 + int((epoch / epochs) * 50)
                progress_callback(f"Entraînement Neuronal ({epoch}/{epochs})", pct)

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                states = torch.cat([x[0] for x in batch])
                targets = torch.tensor([x[1] for x in batch], dtype=torch.float32).unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(states)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def save(self):
        torch.save(self.model.state_dict(), self.model_file)

    def load(self):
        if os.path.exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
            return True
        return False


def get_global_agent():
    agent = Agent()
    if agent.load(): return agent
    return None


def train_cumulative(start_level=1, end_level=None, progress_callback=None):
    agent = Agent()
    dataset = []
    from levels import list_levels
    if end_level is None: end_level = list_levels()

    for lvl in range(start_level, end_level + 1):
        if progress_callback:
            pct = int(((lvl - start_level) / (end_level - start_level + 1)) * 50)
            progress_callback(f"BFS Niveau {lvl}...", pct)
        try:
            vehicles = load_level(lvl)
            board = BoardState(copy.deepcopy(vehicles))
            path = SolverBFS.solve(board)

            if path:
                temp_board = board
                for i, (v_id, delta) in enumerate(path):
                    # 1. On montre le BON chemin
                    state = state_to_tensor(temp_board)
                    remaining = float(len(path) - i)
                    dataset.append((state, remaining))

                    # 2. L'ASTUCE : On montre les MAUVAIS chemins avec une note affreuse (+10)
                    for bad_v, bad_d in temp_board.get_possible_moves():
                        if (bad_v, bad_d) != (v_id, delta):
                            bad_board = temp_board.get_next_state(bad_v, bad_d)
                            bad_state = state_to_tensor(bad_board)
                            dataset.append((bad_state, remaining + 10.0))

                    temp_board = temp_board.get_next_state(v_id, delta)
        except:
            continue

    if dataset:
        agent.train_supervised(dataset, epochs=100, progress_callback=progress_callback)
        agent.save()

    if progress_callback: progress_callback("Apprentissage terminé !", 100)
    return agent


# ==========================================
# 4. RÉSOLUTION GLUTTONE PAR LE RÉSEAU
# ==========================================
def solve_with_nn(start_board, agent, max_steps=20000):
    """
    L'IA est guidée par le réseau de neurones, mais elle a le droit
    de revenir en arrière (A*) si elle s'est trompée à une intersection.
    """
    open_set = []
    counter = 0
    # On stocke: (score_total, identifiant_unique, plateau_actuel, chemin_parcouru)
    heapq.heappush(open_set, (0, counter, start_board, []))
    visited = set()

    while open_set:
        f_score, _, current_board, path = heapq.heappop(open_set)

        if current_board.is_solved():
            return path

        board_h = hash(current_board)
        if board_h in visited:
            continue
        visited.add(board_h)

        # Si l'IA tourne en rond trop longtemps, on arrête
        if counter > max_steps:
            return None

        for v_id, delta in current_board.get_possible_moves():
            next_board = current_board.get_next_state(v_id, delta)
            if hash(next_board) not in visited:
                # Le réseau de neurones donne sa note
                nn_score = agent.heuristic(next_board)

                # Le "vrai" score = coups déjà joués + prédiction du réseau
                total_score = len(path) + 1 + nn_score

                counter += 1
                heapq.heappush(open_set, (total_score, counter, next_board, path + [(v_id, delta)]))

    return None