import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque
from config import GRID_SIZE, RED_CAR_ID
from board import BoardState


# --- 1. LE MODÈLE (L'Élève) ---
class RushHourNet(nn.Module):
    def __init__(self, output_size):
        super(RushHourNet, self).__init__()
        # On garde le même cerveau
        self.fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, GRID_SIZE * GRID_SIZE)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def state_to_tensor(board_state):
    # Transformation de la grille en chiffres pour l'IA
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_id = board_state.grid[r][c]
            if cell_id is not None:
                if cell_id == RED_CAR_ID:
                    matrix[r][c] = 1.0  # La cible
                else:
                    matrix[r][c] = 0.5  # Les obstacles
    return torch.tensor(matrix).unsqueeze(0)


# --- 2. LE PROFESSEUR (Algorithme BFS) ---
class SolverBFS:
    """Trouve le chemin le plus court pour résoudre le niveau."""

    @staticmethod
    def solve(start_board: BoardState):
        # File d'attente : (BoardState, liste_des_actions_pour_arriver_la)
        queue = deque([(start_board, [])])

        # Pour ne pas tourner en rond
        visited = set()
        visited.add(hash(start_board))

        print("Le Professeur (BFS) réfléchit à la solution...")

        while queue:
            current_board, path = queue.popleft()

            if current_board.is_solved():
                return path  # On retourne la liste des coups gagnants

            # On teste tous les mouvements possibles
            for v_id in current_board.vehicles:
                for delta in [-1, 1]:  # Reculer ou Avancer
                    if current_board.is_move_valid(v_id, delta):
                        next_board = current_board.get_next_state(v_id, delta)
                        h = hash(next_board)

                        if h not in visited:
                            visited.add(h)
                            # On ajoute ce nouvel état à explorer
                            # Action stockée sous forme (v_id, delta)
                            new_path = path + [(v_id, delta)]
                            queue.append((next_board, new_path))

        return None  # Pas de solution trouvée


# --- 3. L'AGENT HYBRIDE ---
class Agent:
    def __init__(self, vehicle_ids):
        self.vehicle_ids = vehicle_ids
        self.action_size = len(vehicle_ids) * 2

        # Le cerveau
        self.model = RushHourNet(self.action_size)

        # Pour l'apprentissage supervisé, on utilise CrossEntropyLoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.CrossEntropyLoss()

        self.epsilon = 0.0  # Plus besoin d'aléatoire, l'IA suit le prof

    def act(self, state_tensor):
        # L'IA prédit le meilleur coup
        with torch.no_grad():
            logits = self.model(state_tensor)
        return torch.argmax(logits).item()

    def encode_action(self, v_id, delta):
        # Transforme ("A", 1) en un chiffre (ex: 3) pour le réseau
        try:
            v_idx = self.vehicle_ids.index(v_id)
            # Pair = Avancer (+1), Impair = Reculer (-1)
            is_backward = 1 if delta < 0 else 0
            return (v_idx * 2) + is_backward
        except ValueError:
            return 0

    def decode_action(self, action_idx):
        # Transforme un chiffre en ("A", 1)
        v_idx = action_idx // 2
        delta = 1 if action_idx % 2 == 0 else -1
        if v_idx < len(self.vehicle_ids):
            return self.vehicle_ids[v_idx], delta
        return None, 0

    def train_supervised(self, training_data, epochs=50):
        """
        training_data : liste de (Tensor_Etat, Action_Correcte_Index)
        """
        self.model.train()
        print(f"Entraînement de l'IA sur {len(training_data)} exemples...")

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)

            # On découpe en batchs de 16
            batch_size = 16
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]

                states = torch.cat([x[0] for x in batch])
                target_actions = torch.tensor([x[1] for x in batch], dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(states)

                loss = self.criterion(outputs, target_actions)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Perte (Erreur): {total_loss:.4f}")


# --- 4. FONCTION PRINCIPALE ---
def train_ai(initial_vehicles, episodes=1):
    # Note : 'episodes' ne sert plus ici car on résout une fois, puis on apprend.

    # 1. Préparer l'agent
    vehicle_ids = [v.id for v in initial_vehicles]
    agent = Agent(vehicle_ids)

    # 2. Le Professeur résout le niveau
    initial_board = BoardState(copy.deepcopy(initial_vehicles))
    solution_path = SolverBFS.solve(initial_board)

    if not solution_path:
        print("Erreur : Le professeur n'a pas trouvé de solution (niveau impossible ?).")
        return agent

    print(f"Solution trouvée en {len(solution_path)} coups !")
    print("Génération des données d'entraînement...")

    # 3. On crée les données d'entraînement (Data Augmentation)
    # On rejoue la partie gagnante et on enregistre : "Dans cet état -> Fais ça"
    training_data = []
    board = BoardState(copy.deepcopy(initial_vehicles))

    for v_id, delta in solution_path:
        state_tensor = state_to_tensor(board)
        action_idx = agent.encode_action(v_id, delta)

        # On ajoute l'exemple à la liste
        training_data.append((state_tensor, action_idx))

        # On avance le plateau
        board = board.get_next_state(v_id, delta)

    # 4. L'IA apprend par cœur la leçon
    # On entraîne sur 100 epochs pour être sûr qu'elle retienne bien
    agent.train_supervised(training_data, epochs=200)

    print("L'IA a appris la solution !")
    return agent