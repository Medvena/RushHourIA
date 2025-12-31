# Rush Hour – Projet IA

Ce projet implémente le jeu **Rush Hour** avec une séparation stricte entre :

- la **logique du jeu** (indépendante de toute interface),
- l’**interface graphique** (Pygame),
- le **solveur IA** (exploration et résolution des puzzles).

L’objectif est de disposer d’un moteur fiable pour tester et comparer des algorithmes d’IA sur des puzzles Rush Hour.

---

## Architecture générale

Le projet est organisé en trois couches principales.

---

### 1. Noyau logique (Game Logic)

**Fichier : `vehicle_logic.py`**  
(classes `Vehicle`, `BoardState`)

Responsabilités :
- représentation des véhicules (`Vehicle`)
- représentation complète de l’état du plateau (`BoardState`)
- validation des mouvements (`is_move_valid`)
- génération de nouveaux états (`get_next_state`)
- détection de la victoire (`is_solved`)
- comparaison et hachage des états (`__eq__`, `__hash__`)

Cette couche constitue le **cœur du projet**.  
Elle est **totalement indépendante du GUI et de l’IA** et peut être utilisée seule.

---

### 2. Interface graphique (GUI)

**Fichiers :**
- `rush_hour_gui.py`
- `gui_vehicle.py`

Responsabilités :
- affichage du plateau et des véhicules
- gestion des événements souris (drag & drop)
- synchronisation entre logique et rendu graphique
- affichage des messages (victoire, fermeture)

Classes principales :
- `GUIVehicle` : lien entre un `Vehicle` logique et sa représentation Pygame
- `RushHourGUI` : contrôleur principal de l’interface

Le GUI **ne contient aucune logique de jeu** :  
toutes les règles passent par `BoardState`.

---

### 3. Solveur IA (Solver)

**Fichier : `solver_ia.py`**

Responsabilités :
- implémentation des algorithmes de résolution (ex : BFS, DFS, A*)
- structures IA (ex : `Node`)
- exploration de l’espace des états via `BoardState`

Le solveur fonctionne **sans interface graphique**, uniquement avec la logique pure.

---

## Fichiers de niveaux

Les niveaux sont stockés dans un **fichier texte unique** (`levels.txt`), sous forme de grilles 6×6.

Convention :
- `.` : case vide
- `X` : voiture rouge (cible)
- `A–K` : voitures (taille 2)
- `O–R` : camions (taille 3)

### Exemple

```text
# 1 | Beginner
AA...O
P..Q.O
PXXQ.O
P..Q..
B...CC
B.RRR.

Les orientations et tailles des véhicules sont **déduites automatiquement** lors du chargement.

---

## Objectifs du projet

- implémenter un moteur Rush Hour propre et robuste
- disposer d’un ensemble de niveaux exploitables (40 niveaux)
- tester et comparer des stratégies IA
- garantir une séparation stricte entre logique, interface et IA

---

## Philosophie de conception

- séparation claire **Model / View / Solver**
- logique testable sans interface graphique
- compatibilité avec un mode *headless* (IA seule)
- données (niveaux) séparées du code

---

## Évolutions possibles

- ajout d’heuristiques avancées (A*)
- mode CLI sans GUI
- calcul automatique de la difficulté réelle des niveaux
- génération de statistiques IA (nombre de coups, profondeur, etc.)