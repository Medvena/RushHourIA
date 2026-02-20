import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
import heapq
from collections import deque
from config import GRID_SIZE, RED_CAR_ID
from board import BoardState
from levels import load_level

# ==========================================
# 1. LE CERVEAU (Heuristic Network)
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
        self.fc2 = nn.Linear(512, 1)  # OUTPUT = distance to goal

    def forward(self, x):
        x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def state_to_tensor(board_state):
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_id = board_state.grid[r][c]
            if cell_id is not None:
                if cell_id == RED_CAR_ID:
                    matrix[r][c] = 1.0
                else:
                    matrix[r][c] = 0.5
    return torch.tensor(matrix).unsqueeze(0)


# ==========================================
# 2. BFS PROF POUR GÉNÉRER DATASET
# ==========================================
class SolverBFS:
    @staticmethod
    def solve(start_board: BoardState, max_depth=30000):
        queue = deque([(start_board, [])])
        visited = set([hash(start_board)])

        while queue:
            board, path = queue.popleft()
            if len(path) > max_depth:
                return None

            if board.is_solved():
                return path

            for v_id in board.vehicles:
                for delta in [-1, 1]:
                    if board.is_move_valid(v_id, delta):
                        nxt = board.get_next_state(v_id, delta)
                        h = hash(nxt)
                        if h not in visited:
                            visited.add(h)
                            queue.append((nxt, path + [(v_id, delta)]))
        return None


# ==========================================
# 3. AGENT (Neural Heuristic)
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

    def train_supervised(self, dataset, epochs=50, progress_callback=None):
        self.model.train()
        batch_size = 32
        history_loss = []

        for epoch in range(epochs):
            random.shuffle(dataset)
            epoch_loss = 0
            num_batches = 0

            if progress_callback and epoch % 5 == 0:
                pct = 50 + int((epoch / epochs) * 50)
                progress_callback(f"Training Heuristic NN ({epoch}/{epochs})", pct)

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                states = torch.cat([x[0] for x in batch])
                targets = torch.tensor([x[1] for x in batch], dtype=torch.float32).unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(states)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            history_loss.append(epoch_loss / num_batches)

        return history_loss

    # --- CES MÉTHODES DOIVENT ÊTRE ICI ---
    def save(self):
        """Sauvegarde les poids du réseau de neurones"""
        torch.save(self.model.state_dict(), self.model_file)
        print(f"Modèle sauvegardé sous {self.model_file}")

    def load(self):
        """Charge les poids si le fichier existe"""
        if os.path.exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
            return True
        return False

# ==========================================
# 4. A* SOLVER GUIDÉ PAR LE NN
# ==========================================
def solve_astar_complete(start_board, agent):
    """
    Version complète de A* qui retourne le chemin optimal calculé.
    f(n) = g(n) + h(n)
    """
    # priority_queue: (f_score, counter, current_board, path_taken)
    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, counter, start_board, []))

    visited = {}  # board_hash -> min_cost_to_reach

    while open_set:
        f, _, current_board, path = heapq.heappop(open_set)

        if current_board.is_solved():
            return path  # On a trouvé la solution !

        board_h = hash(current_board)
        if board_h in visited and visited[board_h] <= len(path):
            continue
        visited[board_h] = len(path)

        for v_id, delta in current_board.get_possible_moves():  # Supposant que tu as cette méthode
            next_state = current_board.get_next_state(v_id, delta)
            new_path = path + [(v_id, delta)]

            g_score = len(new_path)
            h_score = agent.heuristic(next_state)  # L'IA estime la distance restante
            f_score = g_score + h_score

            counter += 1
            heapq.heappush(open_set, (f_score, counter, next_state, new_path))

    return None  # Pas de solution trouvée


# ==========================================
# 5. TRAINING GLOBAL
# ==========================================
def get_global_agent():
    agent = Agent()
    if agent.load():
        return agent
    return None


def train_cumulative(start_level=1, end_level=None, progress_callback=None):
    """
    Apprend les niveaux de start_level jusqu'à end_level inclus.
    """
    agent = Agent()
    dataset = []

    if end_level is None:
        from levels import list_levels
        end_level = list_levels()

    print(f"--- TRAINING HEURISTIC NN LEVELS {start_level} -> {end_level} ---")

    # Génération des données
    for lvl in range(start_level, end_level + 1):
        if progress_callback:
            pct = int((lvl - start_level) / (end_level - start_level + 1) * 50)
            progress_callback(f"Solving Level {lvl}", pct)

        try:
            vehicles = load_level(lvl)
            board = BoardState(copy.deepcopy(vehicles))
            path = SolverBFS.solve(board)

            if not path:
                continue

            temp_board = BoardState(copy.deepcopy(vehicles))
            for i, (v_id, delta) in enumerate(path):
                state = state_to_tensor(temp_board)
                remaining = len(path) - i
                dataset.append((state, float(remaining)))
                temp_board = temp_board.get_next_state(v_id, delta)
        except:
            print(f"Level {lvl} skipped")

    # Entraînement
    if dataset:
        print(f"Training on {len(dataset)} states")
        agent.train_supervised(dataset, epochs=60, progress_callback=progress_callback)
        agent.save()
        if progress_callback:
            progress_callback("Done!", 100)
        return agent

    return None


def plot_learning_curve(losses):
    """
    Génère et sauvegarde un graphique montrant l'évolution de l'erreur.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Erreur (MSE)', color='#e67e22', linewidth=2)
    plt.title("Courbe d'apprentissage du modèle Rush Hour", fontsize=14)
    plt.xlabel("Époque (Passages sur les données)", fontsize=12)
    plt.ylabel("Perte (Loss)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Sauvegarde le fichier image dans le dossier du projet
    plt.savefig("learning_curve.png")
    plt.close()  # Ferme la figure pour libérer la mémoire
    print("Graphique sauvegardé sous 'learning_curve.png'")