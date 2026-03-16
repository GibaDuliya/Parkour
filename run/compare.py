import sys
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
from src.landscape.landscape import generate_height_map, build_graph, dijkstra
from src.algorithms import ValueIteration, PolicyIteration
from src.algorithms.baselines import RandomAgent, SafestPathAgent, BudgetAwareGreedyAgent, DPAgent

def load_yaml(path: str) -> dict:
    with open(PROJECT_ROOT / path, "r") as f:
        return yaml.safe_load(f)

def evaluate_agent(agent, env: ParkourEnv, start_state: tuple, max_steps: int) -> dict:
    """Выполняет одну эпоху для агента и возвращает метрики по заданным формулам."""
    agent.reset(start_state, env)
    state = start_state
    
    total_reward = 0.0
    steps = 0
    done = False
    dead = False

    while not done and steps < max_steps:
        action = agent.act(state, env)
        next_state, reward, done, dead = env.step(state, action)
        
        total_reward += reward
        steps += 1
        state = next_state

    victory = done and not dead
    
    # Формула из задания: penalised S_max для провала
    penalised_steps = steps if victory else max_steps

    return {
        "victory": 1 if victory else 0,
        "steps": penalised_steps,
        "reward": total_reward
    }

def main():
    print("Loading configurations...")
    comp_cfg = load_yaml("configs/compare.yaml")
    base_env_cfg = load_yaml("configs/env.yaml")
    
    N_MAPS = comp_cfg["N_MAPS_EVAL"]
    N_EVAL_PER_MAP = comp_cfg["N_EVAL_PER_MAP"]
    RANDOM_INIT = comp_cfg["RANDOM_INIT_EVAL"]
    MAX_STEPS = comp_cfg["MAX_STEPS"]

    agents_dict = {
        "Random": RandomAgent(),
        "Safest Path": SafestPathAgent(),
        "Budget Greedy": BudgetAwareGreedyAgent(),
        "Value Iteration": None,  
        "Policy Iteration": None  
    }

    # Хранилище результатов: agent_name -> { "sr": [], "steps": [], "reward": [] }
    results = {name: {"sr": [], "steps": [], "reward": []} for name in agents_dict.keys()}

    total_episodes = N_MAPS * N_EVAL_PER_MAP
    print(f"\nStarting Evaluation:")
    print(f"Maps: {N_MAPS} | Episodes per map: {N_EVAL_PER_MAP} | Random Init: {RANDOM_INIT}")
    print(f"Total episodes per agent: {total_episodes}\n")

    valid_maps_found = 0
    current_seed = 0

    with tqdm(total=N_MAPS, desc="Evaluating across maps") as pbar:
        while valid_maps_found < N_MAPS:
            # 1. Сгенерировать карту
            height_map = generate_height_map(
                comp_cfg["grid_size"], comp_cfg["min_building_height"], 
                comp_cfg["max_building_height"], seed=current_seed
            )
            
            # 2. Калибровка HP (находим cost_min от 0,0 до цели)
            graph = build_graph(height_map, comp_cfg["max_jump_up"], comp_cfg["safe_jump_down"])
            goal = (comp_cfg["grid_size"] - 1, comp_cfg["grid_size"] - 1)
            cost_min, _ = dijkstra(graph, (0, 0), goal)
            
            # Если карта физически непроходима, пропускаем её и генерируем следующую!
            if cost_min == float('inf'):
                current_seed += 1
                continue

            hp_start = int(cost_min) + 1

            # 3. Инициализация Env для текущей карты
            env_config = {
                "height_map": height_map.tolist(),
                "hp_start": hp_start, 
                "rewards": base_env_cfg["rewards"],
                "max_jump_up": comp_cfg["max_jump_up"],
                "safe_jump_down": comp_cfg["safe_jump_down"]
            }
            env = ParkourEnv(env_config)

            # 4. Обучение VI и PI "на лету" для данной карты
            vi_config = load_yaml("configs/value_iteration.yaml")
            pi_config = load_yaml("configs/policy_iteration.yaml")
            
            vi_algo = ValueIteration(env, vi_config)
            vi_algo.solve()
            agents_dict["Value Iteration"] = DPAgent(vi_algo.get_policy(), vi_algo._state_to_id, vi_algo.actions)

            pi_algo = PolicyIteration(env, pi_config)
            pi_algo.solve()
            agents_dict["Policy Iteration"] = DPAgent(pi_algo.get_policy(), pi_algo._state_to_id, pi_algo.actions)

            # 5. Прогоняем эпизоды
            start_states = []
            if RANDOM_INIT:
                valid_cells = [(i, j) for i in range(env.rows) for j in range(env.cols) if (i, j) != goal]
                rng = np.random.default_rng(current_seed)
                chosen = rng.choice(len(valid_cells), size=N_EVAL_PER_MAP, replace=True)
                for idx in chosen:
                    i, j = valid_cells[idx]
                    start_states.append((i, j, hp_start))
            else:
                start_states = [(0, 0, hp_start)] * N_EVAL_PER_MAP

            for start_state in start_states:
                for agent_name, agent in agents_dict.items():
                    metrics = evaluate_agent(agent, env, start_state, MAX_STEPS)
                    results[agent_name]["sr"].append(metrics["victory"])
                    results[agent_name]["steps"].append(metrics["steps"])
                    results[agent_name]["reward"].append(metrics["reward"])

            valid_maps_found += 1
            current_seed += 1
            pbar.update(1)

    # 6. Агрегация и форматирование вывода
    print("\n" + "="*70)
    print(f"{'Agent Name':<20} | {'Success Rate':<12} | {'Avg Steps':<12} | {'Avg Reward':<12}")
    print("-" * 70)
    
    # Сортировка по Average Reward (чтобы соответствовало Expected Ordering)
    sorted_agents = sorted(
        results.keys(), 
        key=lambda k: np.mean(results[k]["reward"]), 
        reverse=True
    )

    for name in sorted_agents:
        sr = np.mean(results[name]["sr"])
        avg_steps = np.mean(results[name]["steps"])
        avg_reward = np.mean(results[name]["reward"])
        
        print(f"{name:<20} | {sr:>11.2%} | {avg_steps:>12.1f} | {avg_reward:>12.1f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()