import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

    # Predict heuristic value
    def heuristic(self, board):
        self.model.eval()
        with torch.no_grad():
            state = state_to_tensor(board)
            return self.model(state).item()

    def train_supervised(self, dataset, epochs=50, progress_callback=None):
        self.model.train()
        batch_size = 32

        for epoch in range(epochs):
            random.shuffle(dataset)
            if progress_callback and epoch % 5 == 0:
                pct = 50 + int((epoch / epochs) * 50)
                progress_callback(f"Training Heuristic NN ({epoch}/{epochs})", pct)

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
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


# ==========================================
# 4. A* SOLVER GUIDÉ PAR LE NN
# ==========================================
def solve_astar_stepwise(start_board, agent):
    """
    Generator A* pas-à-pas : yield un coup (v_id, delta) à la fois.
    """
    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, counter, start_board, []))
    visited = set()
    max_steps = 10000
    loops = 0

    while open_set:
        f, _, board, path = heapq.heappop(open_set)
        loops += 1
        if loops > max_steps:
            return  # Timeout

        if board.is_solved():
            for move in path:
                yield move
            return

        if hash(board) in visited:
            continue
        visited.add(hash(board))

        g = len(path)
        h = agent.heuristic(board)

        for v_id in board.vehicles:
            for delta in [-1, 1]:
                if board.is_move_valid(v_id, delta):
                    nxt = board.get_next_state(v_id, delta)
                    new_path = path + [(v_id, delta)]
                    counter += 1
                    f_score = g + h
                    heapq.heappush(open_set, (f_score, counter, nxt, new_path))

        # Yield le prochain coup disponible pour l’affichage
        if path:
            yield path[0]


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