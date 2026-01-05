import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import heapq  # Pour la file de priorité
from collections import deque
from config import GRID_SIZE, RED_CAR_ID
from board import BoardState
from vehicle import Vehicle


# ==========================================
# 1. LE CERVEAU (CNN)
# ==========================================
class RushHourNet(nn.Module):
    def __init__(self, output_size):
        super(RushHourNet, self).__init__()
        # On garde ton CNN performant
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
                    matrix[r][c] = 1.0
                else:
                    matrix[r][c] = 0.5
    return torch.tensor(matrix).unsqueeze(0)


# ==========================================
# 2. GÉNÉRATEUR (ACADÉMIE)
# ==========================================
class LevelGenerator:
    @staticmethod
    def generate_random_level(num_vehicles=8):
        vehicles = []
        grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Voiture rouge
        red_col = random.randint(0, 3)
        vehicles.append(Vehicle(RED_CAR_ID, 'h', 2, 2, red_col))
        grid[2][red_col] = RED_CAR_ID
        grid[2][red_col + 1] = RED_CAR_ID

        attempts = 0
        ids = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) != 'X']
        cur_idx = 0

        while len(vehicles) < num_vehicles and attempts < 1000:
            attempts += 1
            orientation = random.choice(['h', 'v'])
            size = random.choice([2, 2, 3])
            if orientation == 'h':
                row = random.randint(0, 5)
                col = random.randint(0, 6 - size)
                if row == 2: continue
            else:
                row = random.randint(0, 6 - size)
                col = random.randint(0, 5)

            collision = False
            if orientation == 'h':
                for k in range(size):
                    if grid[row][col + k] is not None: collision = True
            else:
                for k in range(size):
                    if grid[row + k][col] is not None: collision = True

            if not collision:
                v_id = ids[cur_idx % len(ids)]
                cur_idx += 1
                vehicles.append(Vehicle(v_id, orientation, size, row, col))
                if orientation == 'h':
                    for k in range(size): grid[row][col + k] = v_id
                else:
                    for k in range(size): grid[row + k][col] = v_id

        return vehicles


# ==========================================
# 3. LE PROFESSEUR (BFS CLASSIQUE)
# ==========================================
class SolverBFS:
    """Sert à générer la vérité terrain pour l'entraînement."""

    @staticmethod
    def solve(start_board: BoardState, max_depth=5000):
        queue = deque([(start_board, [])])
        visited = set()
        visited.add(hash(start_board))
        loops = 0
        while queue:
            current, path = queue.popleft()
            loops += 1
            if loops > max_depth: return None
            if current.is_solved(): return path

            for v_id in current.vehicles:
                for delta in [-1, 1]:
                    if current.is_move_valid(v_id, delta):
                        nxt = current.get_next_state(v_id, delta)
                        h = hash(nxt)
                        if h not in visited:
                            visited.add(h)
                            queue.append((nxt, path + [(v_id, delta)]))
        return None


# ==========================================
# 4. L'AGENT HYBRIDE (IA + RECHERCHE)
# ==========================================
class Agent:
    def __init__(self):
        self.action_size = 60
        self.model = RushHourNet(self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()
        self.model_file = "rush_hour_hybrid.pth"

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

    def get_action_probabilities(self, board):
        """Demande au cerveau ce qu'il pense de la situation."""
        state = state_to_tensor(board)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()[0]

    # --- C'EST ICI QUE LA MAGIE OPÈRE ---
    def solve_with_ai(self, start_board, max_nodes=50000):
        """
        Recherche A* guidée par l'IA.
        Correction du bug 'TypeError': On ajoute un compteur pour départager les égalités.
        """
        # Compteur pour départager les nœuds ayant le même score
        tie_breaker = 0

        # Structure : (Score_IA, Tie_Breaker, Board, Chemin)
        priority_queue = []
        heapq.heappush(priority_queue, (0, tie_breaker, start_board, []))

        visited = set()
        visited.add(hash(start_board))

        nodes_explored = 0

        while priority_queue:
            # On récupère les 4 éléments (le _ sert à ignorer le tie_breaker)
            score, _, current_board, path = heapq.heappop(priority_queue)
            nodes_explored += 1

            if nodes_explored > max_nodes:
                print("   [IA] Trop complexe, j'abandonne...")
                return None

            if current_board.is_solved():
                return path

            # L'IA analyse le plateau
            probs = self.get_action_probabilities(current_board)

            for v_id in current_board.vehicles:
                for delta in [-1, 1]:
                    if current_board.is_move_valid(v_id, delta):
                        action_idx = self.encode_action(v_id, delta)
                        move_prob = probs[action_idx]

                        next_board = current_board.get_next_state(v_id, delta)

                        if hash(next_board) not in visited:
                            visited.add(hash(next_board))

                            # Calcul du score heuristique
                            heuristic_score = len(path) - (move_prob * 10)

                            # On incrémente le compteur pour que chaque entrée soit unique
                            tie_breaker += 1

                            # On pousse avec le tie_breaker
                            heapq.heappush(priority_queue,
                                           (heuristic_score, tie_breaker, next_board, path + [(v_id, delta)]))

        return None

    # --- ENTRAINEMENT ---
    def train_on_dataset(self, dataset, epochs=10):
        self.model.train()
        random.shuffle(dataset)
        batch_size = 64
        for epoch in range(epochs):
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
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
        print("Cerveau Hybride sauvegardé.")

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
# 5. FONCTIONS GLOBALES
# ==========================================
def get_trained_agent():
    agent = Agent()
    if agent.load(): return agent
    return None


def train_generalist_model(num_levels=300):
    print(f"\n--- ACADÉMIE HYBRIDE : {num_levels} Niveaux ---")
    agent = Agent()
    dataset = []

    success = 0
    while success < num_levels:
        nb_cars = random.randint(6, 12)
        vehicles = LevelGenerator.generate_random_level(nb_cars)
        board = BoardState(vehicles)
        path = SolverBFS.solve(board)  # Vérité terrain

        if path and len(path) > 4:
            success += 1
            print(f"\rGénération {success}/{num_levels}", end="")
            temp = copy.deepcopy(board)
            for v_id, delta in path:
                st = state_to_tensor(temp)
                act = agent.encode_action(v_id, delta)
                dataset.append((st, act))
                temp = temp.get_next_state(v_id, delta)

    print(f"\nEntraînement sur {len(dataset)} positions...")
    agent.train_on_dataset(dataset, epochs=50)
    agent.save()