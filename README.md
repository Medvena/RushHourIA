# Rush-Hour

vehicle_logic.py 
Classes Vehicle, BoardState. Toute la logique du jeu (is_move_valid, is_solved, __hash__). Le Noyau. 


rush_hour_gui.py 
Classes GraphicalCar, RushHourGUI. Toute la gestion Pygame, les événements souris, le dessin. L'Interface. 


solver_ia.py 
Fonctions et classes pour l'IA (ex: solve_bfs, Node). Utilise BoardState pour générer et tester des mouvements. Le Solveur. 
