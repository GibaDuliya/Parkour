import numpy as np
from src.environment.parkour_env import Action, ParkourEnv
from src.landscape.landscape import build_graph, dijkstra

class BaseAgent:
    """Интерфейс для всех агентов в рамках сравнения."""
    def reset(self, start_state: tuple, env: ParkourEnv):
        pass

    def act(self, state: tuple, env: ParkourEnv) -> int:
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def act(self, state: tuple, env: ParkourEnv) -> int:
        return np.random.choice(list(Action))

class SafestPathAgent(BaseAgent):
    def __init__(self):
        self.path = None
        self.target_idx = 0

    def reset(self, start_state: tuple, env: ParkourEnv):
        i, j, _ = start_state
        # В новой архитектуре build_graph принимает 3 аргумента
        graph = build_graph(env.height_map, env.max_jump_up, env.safe_jump_down)
        goal = (env.rows - 1, env.cols - 1)
        # dijkstra возвращает (total_damage, path)
        _, path = dijkstra(graph, (i, j), goal)
        self.path = path  # Список кортежей (i, j)
        self.target_idx = 1  # Индекс 0 — это стартовая позиция

    def act(self, state: tuple, env: ParkourEnv) -> int:
        # Если путь не найден или мы дошли до конца пути, делаем случайный шаг
        if not self.path or self.target_idx >= len(self.path):
            return np.random.choice(list(Action))
        
        curr_pos = (state[0], state[1])
        next_pos = self.path[self.target_idx]
        
        # Определяем действие по разнице координат
        di = next_pos[0] - curr_pos[0]
        dj = next_pos[1] - curr_pos[1]
        
        self.target_idx += 1
        
        if di == -1: return Action.UP
        if di == 1: return Action.DOWN
        if dj == -1: return Action.LEFT
        if dj == 1: return Action.RIGHT
        
        return Action.RIGHT

class BudgetAwareGreedyAgent(BaseAgent):
    def act(self, state: tuple, env: ParkourEnv) -> int:
        i, j, hp = state
        goal = (env.rows - 1, env.cols - 1)
        
        # Manhattan distance
        dist = abs(goal[0] - i) + abs(goal[1] - j)
        if dist == 0:
            return Action.RIGHT # Уже на месте
            
        budget = (hp - 1) / dist
        
        action_deltas = {
            Action.UP: (-1, 0), Action.DOWN: (1, 0),
            Action.LEFT: (0, -1), Action.RIGHT: (0, 1)
        }
        
        best_action = None
        min_dist_to_goal = float('inf')
        
        fallback_action = None
        min_damage = float('inf')
        
        for action in Action:
            di, dj = action_deltas[action]
            ni, nj = i + di, j + dj
            
            # Проверка на границы и максимальный прыжок вверх
            if not (0 <= ni < env.rows and 0 <= nj < env.cols):
                continue
            if env.height_map[ni, nj] - env.height_map[i, j] > env.max_jump_up:
                continue
                
            drop = env.height_map[i, j] - env.height_map[ni, nj]
            damage = max(0, drop - env.safe_jump_down)
            
            # Сохраняем самое безопасное действие на случай, если бюджет превышен для всех
            if damage < min_damage:
                min_damage = damage
                fallback_action = action
                
            new_dist = abs(goal[0] - ni) + abs(goal[1] - nj)
            
            # Если укладываемся в бюджет, ищем кратчайший путь (жадность)
            if damage <= budget:
                if new_dist < min_dist_to_goal:
                    min_dist_to_goal = new_dist
                    best_action = action
                    
        if best_action is not None:
            return best_action
        if fallback_action is not None:
            return fallback_action
            
        return np.random.choice(list(Action))

class DPAgent(BaseAgent):
    """Обертка для обученных RL-политик (VI, PI, QL)"""
    def __init__(self, policy: np.ndarray, state_to_id: dict, actions: list):
        self.policy = policy
        self.state_to_id = state_to_id
        self.actions = actions

    def act(self, state: tuple, env: ParkourEnv) -> int:
        s_id = self.state_to_id.get(state)
        if s_id is None:
            return np.random.choice(self.actions) # Fallback для неизвестных состояний
        a_id = int(self.policy[s_id])
        return self.actions[a_id]
    
class ShortestPathAgent(BaseAgent):
    """
    Глобальный, но рискованный агент.
    Ищет путь с минимальным количеством шагов, полностью игнорируя урон.
    Доходит быстро, но часто разбивается.
    """
    def __init__(self):
        self.path = None
        self.target_idx = 0

    def reset(self, start_state: tuple, env: ParkourEnv):
        i, j, _ = start_state
        rows, cols = env.rows, env.cols
        goal = (rows - 1, cols - 1)
        
        # Строим граф, где вес любого валидного перехода = 1
        # Валидный переход — это только ограничение по прыжку ВВЕРХ
        dist = {(i, j): 0}
        prev = {(i, j): None}
        import heapq
        queue = [(0, (i, j))]
        
        action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            d, u = heapq.heappop(queue)
            if d > dist.get(u, float('inf')): continue
            if u == goal: break
            
            curr_i, curr_j = u
            h_curr = env.height_map[curr_i, curr_j]
            
            for di, dj in action_deltas:
                ni, nj = curr_i + di, curr_j + dj
                
                if 0 <= ni < rows and 0 <= nj < cols:
                    h_next = env.height_map[ni, nj]
                    # Проверяем только физическую проходимость (прыжок вверх)
                    if h_next - h_curr <= env.max_jump_up:
                        if d + 1 < dist.get((ni, nj), float('inf')):
                            dist[(ni, nj)] = d + 1
                            prev[(ni, nj)] = u
                            heapq.heappush(queue, (d + 1, (ni, nj)))
        
        # Восстанавливаем путь
        path = []
        curr = goal
        if curr in prev:
            while curr is not None:
                path.append(curr)
                curr = prev[curr]
            self.path = path[::-1]
        else:
            self.path = None
        
        self.target_idx = 1

    def act(self, state: tuple, env: ParkourEnv) -> int:
        if not self.path or self.target_idx >= len(self.path):
            return np.random.choice(list(Action))
        
        curr_pos = (state[0], state[1])
        next_pos = self.path[self.target_idx]
        self.target_idx += 1
        
        di, dj = next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1]
        if di == -1: return Action.UP
        if di == 1: return Action.DOWN
        if dj == -1: return Action.LEFT
        if dj == 1: return Action.RIGHT
        return Action.RIGHT