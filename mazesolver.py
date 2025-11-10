import pygame
import sys
import random as rd
from collections import defaultdict
from collections import deque
import heapq
import heapq

pygame.init()

WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Solver")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

ml, mw = 20, 20  # maze length and width in cells
maze_grid = [(i, j) for i in range(1, ml+1) for j in range(1, mw + 1)]



TILE_SIZE_X = WIDTH // mw
TILE_SIZE_Y = HEIGHT // ml
NUM_COLS = WIDTH // TILE_SIZE_X
NUM_ROWS = HEIGHT // TILE_SIZE_Y
GRID_COLOR = (100, 100, 100)

graph = defaultdict(set)

for row, column in maze_grid:
    
    for rowcomp, columncomp in maze_grid: 
        if abs(row-rowcomp) <= 1 and abs(column-columncomp) <= 1: #only gets rows
            if (row, column) != (rowcomp, columncomp):
                if not abs(row-rowcomp) == 1 and abs(column-columncomp) == 1:
                    graph[(row, column)].add((rowcomp, columncomp))
                    
    if row > 1:
        graph[(row, column)].add((row-1, column))   
    
    if row < ml:
        graph[(row, column)].add((row+1, column)) 
                    


def gen_maze(G, start=(1,1), weightsplit=(0.5, 0.5)):
    """Generate a maze using iterative randomized DFS (recursive backtracker).

    Returns a defaultdict(set) mapping each cell to connected neighbors.
    """
    visited = set([start])
    stack = [start]
    maze = defaultdict(set)

    while stack:
        x = rd.choices((0,-1), weights=weightsplit)[0]
        cell = stack[x]
        unvisited = [i for i in G[cell] if i not in visited] #everything not visited yet
        if unvisited:
            nxt = rd.choice(unvisited) #add to maze
            maze[cell].add(nxt)
            maze[nxt].add(cell)
            visited.add(nxt)
            stack.append(nxt)
        else:
            stack.pop(x)

    return maze

# Generator function for animated maze generation
def gen_maze_animated(G, start=(1,1), weightsplit=(0.5, 0.5)):
    """Generator that yields maze state at each step for visualization.
    
    Yields: (maze, current_cell, stack, visited) at each step
    """
    visited = set([start])
    stack = [start]
    maze = defaultdict(set)

    while stack:
        x = rd.choices((0,-1), weights=weightsplit)[0]
        cell = stack[x]
        unvisited = [i for i in G[cell] if i not in visited]
        if unvisited:
            nxt = rd.choice(unvisited)
            maze[cell].add(nxt)
            maze[nxt].add(cell)
            visited.add(nxt)
            stack.append(nxt)
            yield maze, nxt, stack.copy(), visited.copy()
        else:
            stack.pop(x)
            if stack:
                yield maze, stack[-1], stack.copy(), visited.copy()
    
    yield maze, None, [], visited

# Initialize maze generator
maze_generator = gen_maze_animated(graph, (1,1), (0.7, 0.3))
maze = defaultdict(set)
current_cell = None
generating = True


# Pathfinding algorithm generators
def bfs_solver(maze, start, goal):
    """BFS pathfinding generator."""
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        yield current, visited.copy(), parent
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            yield current, visited, parent, path[::-1]
            return
        
        for neighbor in maze.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    yield current, visited, parent, []

def dfs_solver(maze, start, goal):
    """DFS pathfinding generator."""
    stack = [start]
    visited = {start}
    parent = {start: None}
    
    while stack:
        current = stack.pop()
        yield current, visited.copy(), parent
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            yield current, visited, parent, path[::-1]
            return
        
        for neighbor in maze.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
    
    yield current, visited, parent, []

def dijkstra_solver(maze, start, goal):
    """Dijkstra's algorithm generator (all edges weight 1)."""
    
    pq = [(0, start)]
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        yield current, visited.copy(), parent
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            yield current, visited, parent, path[::-1]
            return
        
        for neighbor in maze.get(current, set()):
            new_cost = current_cost + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
    
    yield current, visited, parent, []


# Algorithm configurations
algorithms = {
    'BFS': {'color': CYAN, 'generator': None, 'current': None, 'visited': set(), 'path': [], 'finished': False},
    'DFS': {'color': YELLOW, 'generator': None, 'current': None, 'visited': set(), 'path': [], 'finished': False},
    'Dijkstra': {'color': ORANGE, 'generator': None, 'current': None, 'visited': set(), 'path': [], 'finished': False},
}

solving = False
winner = None  # Track which algorithm won




# Game loop variables
running = True
clock = pygame.time.Clock()
FPS = 60

# Initialize font for labels
pygame.font.init()
font = pygame.font.Font(None, 24)
small_font = pygame.font.Font(None, 18)




while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    SCREEN.fill(WHITE)

    # Step through maze generation
    if generating:
        try:
            maze, current_cell, stack, visited = next(maze_generator)
        except StopIteration:
            generating = False
            current_cell = None
            print("Maze generation complete!")
            # Initialize all algorithm generators
            start = (1, 1)
            goal = (ml, mw)
            algorithms['BFS']['generator'] = bfs_solver(maze, start, goal)
            algorithms['DFS']['generator'] = dfs_solver(maze, start, goal)
            algorithms['Dijkstra']['generator'] = dijkstra_solver(maze, start, goal)
            solving = True
    
    # Step through all solving algorithms
    if solving:
        all_finished = True
        for name, algo in algorithms.items():
            if not algo['finished']:
                all_finished = False
                try:
                    result = next(algo['generator'])
                    if len(result) == 4:  # Has path - algorithm finished
                        algo['current'], algo['visited'], _, algo['path'] = result
                        algo['finished'] = True
                        # Check if this is the first to finish (winner!)
                        if winner is None and algo['path']:
                            winner = name
                            print(f"WINNER: {name}! Path length: {len(algo['path'])}, Cells explored: {len(algo['visited'])}")
                        else:
                            print(f"{name} finished! Path length: {len(algo['path'])}, Cells explored: {len(algo['visited'])}")
                    else:  # Still searching
                        algo['current'], algo['visited'], _ = result
                except StopIteration:
                    algo['finished'] = True
        
        if all_finished:
            solving = False
            print("All algorithms finished!")

    # Draw border around the entire maze
    border_thickness = 3
    maze_width = NUM_COLS * TILE_SIZE_X
    maze_height = NUM_ROWS * TILE_SIZE_Y
    pygame.draw.rect(SCREEN, BLACK, (0, 0, maze_width, maze_height), border_thickness)

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            # Calculate top-left pixel of the tile (cell coords are 1-based)
            x = col * TILE_SIZE_X
            y = row * TILE_SIZE_Y
            cell = (row + 1, col + 1)

            # Draw algorithm exploration (visited cells) - blend colors if multiple algorithms visited
            if not generating:
                colors_to_blend = []
                for name, algo in algorithms.items():
                    if cell in algo['visited'] and cell not in algo['path']:
                        colors_to_blend.append(algo['color'])
                
                if colors_to_blend:
                    # Average the colors
                    avg_color = tuple(sum(c[i] for c in colors_to_blend) // len(colors_to_blend) for i in range(3))
                    pygame.draw.rect(SCREEN, avg_color, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))
            
            # Draw algorithm paths (final solution) on top
            if not generating:
                for name, algo in algorithms.items():
                    if cell in algo['path']:
                        pygame.draw.rect(SCREEN, algo['color'], (x + 3, y + 3, TILE_SIZE_X - 6, TILE_SIZE_Y - 6))
            
            # Draw start cell (top-left) in green
            if cell == (1, 1):
                pygame.draw.rect(SCREEN, GREEN, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))
            
            # Draw finish cell (bottom-right) in red
            if cell == (ml, mw):
                pygame.draw.rect(SCREEN, RED, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))
            
            # Highlight current cell being processed (in blue) during maze generation
            if generating and cell == current_cell:
                pygame.draw.rect(SCREEN, BLUE, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))

            # neighbor coords in 1-based indexing
            north = (cell[0] - 1, cell[1])
            south = (cell[0] + 1, cell[1])
            east = (cell[0], cell[1] + 1)
            west = (cell[0], cell[1] - 1)

            neighbors = maze.get(cell, set())

            # Draw walls where there is no carved passage
            if north not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x, y), (x + TILE_SIZE_X, y))  # top
            if west not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x, y), (x, y + TILE_SIZE_Y))  # left
            if east not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x + TILE_SIZE_X, y), (x + TILE_SIZE_X, y + TILE_SIZE_Y))  # right
            if south not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x, y + TILE_SIZE_Y), (x + TILE_SIZE_X, y + TILE_SIZE_Y))  # bottom
    
    # Draw winner banner
    if winner is not None:
        banner_height = 50
        banner_y = (maze_height - banner_height) // 2
        # Semi-transparent background
        banner_surface = pygame.Surface((WIDTH, banner_height))
        banner_surface.set_alpha(220)
        banner_surface.fill(WHITE)
        SCREEN.blit(banner_surface, (0, banner_y))
        
        # Draw border
        pygame.draw.rect(SCREEN, algorithms[winner]['color'], (0, banner_y, WIDTH, banner_height), 4)
        
        # Winner text
        winner_font = pygame.font.Font(None, 48)
        winner_text = f"{winner} WINS!"
        text_surface = winner_font.render(winner_text, True, algorithms[winner]['color'])
        text_rect = text_surface.get_rect(center=(WIDTH // 2, banner_y + banner_height // 2))
        SCREEN.blit(text_surface, text_rect)
    
    # Draw algorithm labels and statistics
    if not generating:
        y_offset = maze_height + 10
        x_offset = 10
        for name, algo in algorithms.items():
            # Draw color box
            pygame.draw.rect(SCREEN, algo['color'], (x_offset, y_offset, 20, 20))
            pygame.draw.rect(SCREEN, BLACK, (x_offset, y_offset, 20, 20), 1)
            
            # Draw label with winner indicator
            status = "✓" if algo['finished'] else "..."
            if name == winner:
                status = "done"
            text = f"{name} {status}"
            if algo['finished'] and algo['path']:
                text += f" | Path: {len(algo['path'])} | Explored: {len(algo['visited'])}"
            label = small_font.render(text, True, BLACK)
            SCREEN.blit(label, (x_offset + 25, y_offset + 2))
            
            x_offset += 250
            if x_offset > WIDTH - 200:
                x_offset = 10
                y_offset += 25
               
   
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
    
    

    


import pygame
import sys
import random as rd
from collections import defaultdict
from collections import deque
import heapq
import heapq

pygame.init()

WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Solver")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

ml, mw = 20, 20  # maze length and width in cells
maze_grid = [(i, j) for i in range(1, ml+1) for j in range(1, mw + 1)]



TILE_SIZE_X = WIDTH // mw
TILE_SIZE_Y = HEIGHT // ml
NUM_COLS = WIDTH // TILE_SIZE_X
NUM_ROWS = HEIGHT // TILE_SIZE_Y
GRID_COLOR = (100, 100, 100)

graph = defaultdict(set)

for row, column in maze_grid:
    
    for rowcomp, columncomp in maze_grid: 
        if abs(row-rowcomp) <= 1 and abs(column-columncomp) <= 1: #only gets rows
            if (row, column) != (rowcomp, columncomp):
                if not abs(row-rowcomp) == 1 and abs(column-columncomp) == 1:
                    graph[(row, column)].add((rowcomp, columncomp))
                    
    if row > 1:
        graph[(row, column)].add((row-1, column))   
    
    if row < ml:
        graph[(row, column)].add((row+1, column)) 
                    


def gen_maze(G, start=(1,1), weightsplit=(0.5, 0.5)):
    """Generate a maze using iterative randomized DFS (recursive backtracker).

    Returns a defaultdict(set) mapping each cell to connected neighbors.
    """
    visited = set([start])
    stack = [start]
    maze = defaultdict(set)

    while stack:
        x = rd.choices((0,-1), weights=weightsplit)[0]
        cell = stack[x]
        unvisited = [i for i in G[cell] if i not in visited] #everything not visited yet
        if unvisited:
            nxt = rd.choice(unvisited) #add to maze
            maze[cell].add(nxt)
            maze[nxt].add(cell)
            visited.add(nxt)
            stack.append(nxt)
        else:
            stack.pop(x)

    return maze

# Generator function for animated maze generation
def gen_maze_animated(G, start=(1,1), weightsplit=(0.5, 0.5)):
    """Generator that yields maze state at each step for visualization.
    
    Yields: (maze, current_cell, stack, visited) at each step
    """
    visited = set([start])
    stack = [start]
    maze = defaultdict(set)

    while stack:
        x = rd.choices((0,-1), weights=weightsplit)[0]
        cell = stack[x]
        unvisited = [i for i in G[cell] if i not in visited]
        if unvisited:
            nxt = rd.choice(unvisited)
            maze[cell].add(nxt)
            maze[nxt].add(cell)
            visited.add(nxt)
            stack.append(nxt)
            yield maze, nxt, stack.copy(), visited.copy()
        else:
            stack.pop(x)
            if stack:
                yield maze, stack[-1], stack.copy(), visited.copy()
    
    yield maze, None, [], visited

# Initialize maze generator
maze_generator = gen_maze_animated(graph, (1,1), (0.7, 0.3))
maze = defaultdict(set)
current_cell = None
generating = True


# Pathfinding algorithm generators
def bfs_solver(maze, start, goal):
    """BFS pathfinding generator."""
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        yield current, visited.copy(), parent
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            yield current, visited, parent, path[::-1]
            return
        
        for neighbor in maze.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    yield current, visited, parent, []

def dfs_solver(maze, start, goal):
    """DFS pathfinding generator."""
    stack = [start]
    visited = {start}
    parent = {start: None}
    
    while stack:
        current = stack.pop()
        yield current, visited.copy(), parent
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            yield current, visited, parent, path[::-1]
            return
        
        for neighbor in maze.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
    
    yield current, visited, parent, []

def dijkstra_solver(maze, start, goal):
    """Dijkstra's algorithm generator (all edges weight 1)."""
    
    pq = [(0, start)]
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        yield current, visited.copy(), parent
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            yield current, visited, parent, path[::-1]
            return
        
        for neighbor in maze.get(current, set()):
            new_cost = current_cost + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
    
    yield current, visited, parent, []


# Algorithm configurations
algorithms = {
    'BFS': {'color': CYAN, 'generator': None, 'current': None, 'visited': set(), 'path': [], 'finished': False},
    'DFS': {'color': YELLOW, 'generator': None, 'current': None, 'visited': set(), 'path': [], 'finished': False},
    'Dijkstra': {'color': ORANGE, 'generator': None, 'current': None, 'visited': set(), 'path': [], 'finished': False},
}

solving = False
winner = None  # Track which algorithm won




# Game loop variables
running = True
clock = pygame.time.Clock()
FPS = 60

# Initialize font for labels
pygame.font.init()
font = pygame.font.Font(None, 24)
small_font = pygame.font.Font(None, 18)




while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    SCREEN.fill(WHITE)

    # Step through maze generation
    if generating:
        try:
            maze, current_cell, stack, visited = next(maze_generator)
        except StopIteration:
            generating = False
            current_cell = None
            print("Maze generation complete!")
            # Initialize all algorithm generators
            start = (1, 1)
            goal = (ml, mw)
            algorithms['BFS']['generator'] = bfs_solver(maze, start, goal)
            algorithms['DFS']['generator'] = dfs_solver(maze, start, goal)
            algorithms['Dijkstra']['generator'] = dijkstra_solver(maze, start, goal)
            solving = True
    
    # Step through all solving algorithms
    if solving:
        all_finished = True
        for name, algo in algorithms.items():
            if not algo['finished']:
                all_finished = False
                try:
                    result = next(algo['generator'])
                    if len(result) == 4:  # Has path - algorithm finished
                        algo['current'], algo['visited'], _, algo['path'] = result
                        algo['finished'] = True
                        # Check if this is the first to finish (winner!)
                        if winner is None and algo['path']:
                            winner = name
                            print(f"🏆 WINNER: {name}! Path length: {len(algo['path'])}, Cells explored: {len(algo['visited'])}")
                        else:
                            print(f"{name} finished! Path length: {len(algo['path'])}, Cells explored: {len(algo['visited'])}")
                    else:  # Still searching
                        algo['current'], algo['visited'], _ = result
                except StopIteration:
                    algo['finished'] = True
        
        if all_finished:
            solving = False
            print("All algorithms finished!")

    # Draw border around the entire maze
    border_thickness = 3
    maze_width = NUM_COLS * TILE_SIZE_X
    maze_height = NUM_ROWS * TILE_SIZE_Y
    pygame.draw.rect(SCREEN, BLACK, (0, 0, maze_width, maze_height), border_thickness)

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            # Calculate top-left pixel of the tile (cell coords are 1-based)
            x = col * TILE_SIZE_X
            y = row * TILE_SIZE_Y
            cell = (row + 1, col + 1)

            # Draw algorithm exploration (visited cells) - blend colors if multiple algorithms visited
            if not generating:
                colors_to_blend = []
                for name, algo in algorithms.items():
                    if cell in algo['visited'] and cell not in algo['path']:
                        colors_to_blend.append(algo['color'])
                
                if colors_to_blend:
                    # Average the colors
                    avg_color = tuple(sum(c[i] for c in colors_to_blend) // len(colors_to_blend) for i in range(3))
                    pygame.draw.rect(SCREEN, avg_color, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))
            
            # Draw algorithm paths (final solution) on top
            if not generating:
                for name, algo in algorithms.items():
                    if cell in algo['path']:
                        pygame.draw.rect(SCREEN, algo['color'], (x + 3, y + 3, TILE_SIZE_X - 6, TILE_SIZE_Y - 6))
            
            # Draw start cell (top-left) in green
            if cell == (1, 1):
                pygame.draw.rect(SCREEN, GREEN, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))
            
            # Draw finish cell (bottom-right) in red
            if cell == (ml, mw):
                pygame.draw.rect(SCREEN, RED, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))
            
            # Highlight current cell being processed (in blue) during maze generation
            if generating and cell == current_cell:
                pygame.draw.rect(SCREEN, BLUE, (x + 2, y + 2, TILE_SIZE_X - 4, TILE_SIZE_Y - 4))

            # neighbor coords in 1-based indexing
            north = (cell[0] - 1, cell[1])
            south = (cell[0] + 1, cell[1])
            east = (cell[0], cell[1] + 1)
            west = (cell[0], cell[1] - 1)

            neighbors = maze.get(cell, set())

            # Draw walls where there is no carved passage
            if north not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x, y), (x + TILE_SIZE_X, y))  # top
            if west not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x, y), (x, y + TILE_SIZE_Y))  # left
            if east not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x + TILE_SIZE_X, y), (x + TILE_SIZE_X, y + TILE_SIZE_Y))  # right
            if south not in neighbors:
                pygame.draw.line(SCREEN, GRID_COLOR, (x, y + TILE_SIZE_Y), (x + TILE_SIZE_X, y + TILE_SIZE_Y))  # bottom
    
    # Draw winner banner
    if winner is not None:
        banner_height = 50
        banner_y = (maze_height - banner_height) // 2
        # Semi-transparent background
        banner_surface = pygame.Surface((WIDTH, banner_height))
        banner_surface.set_alpha(220)
        banner_surface.fill(WHITE)
        SCREEN.blit(banner_surface, (0, banner_y))
        
        # Draw border
        pygame.draw.rect(SCREEN, algorithms[winner]['color'], (0, banner_y, WIDTH, banner_height), 4)
        
        # Winner text
        winner_font = pygame.font.Font(None, 48)
        winner_text = f"{winner} WINS!"
        text_surface = winner_font.render(winner_text, True, algorithms[winner]['color'])
        text_rect = text_surface.get_rect(center=(WIDTH // 2, banner_y + banner_height // 2))
        SCREEN.blit(text_surface, text_rect)
    
    # Draw algorithm labels and statistics
    if not generating:
        y_offset = maze_height + 10
        x_offset = 10
        for name, algo in algorithms.items():
            # Draw color box
            pygame.draw.rect(SCREEN, algo['color'], (x_offset, y_offset, 20, 20))
            pygame.draw.rect(SCREEN, BLACK, (x_offset, y_offset, 20, 20), 1)
            
            # Draw label with winner indicator
            status = "✓" if algo['finished'] else "..."
            if name == winner:
                status = "done"
            text = f"{name} {status}"
            if algo['finished'] and algo['path']:
                text += f" | Path: {len(algo['path'])} | Explored: {len(algo['visited'])}"
            label = small_font.render(text, True, BLACK)
            SCREEN.blit(label, (x_offset + 25, y_offset + 2))
            
            x_offset += 250
            if x_offset > WIDTH - 200:
                x_offset = 10
                y_offset += 25
               
   
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
    
    

    

