import sys
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Настройка путей
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
from src.algorithms.baselines import RandomAgent, SafestPathAgent, BudgetAwareGreedyAgent, DPAgent, ShortestPathAgent

MAX_STEPS = 2000
GAMMA = 0.99  # Коэффициент дисконтирования

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_trained_dp_agent(algo_name: str, landscape_id: int):
    """Ищет последний прогон агента для конкретного ландшафта."""
    algo_dir = PROJECT_ROOT / "agents" / algo_name
    if not algo_dir.exists():
        return None

    valid_runs = []
    for run_dir in algo_dir.iterdir():
        if not run_dir.is_dir(): continue
        meta_path = run_dir / "meta.yaml"
        if meta_path.exists():
            meta = load_yaml(meta_path)
            if meta.get("landscape_id") == landscape_id:
                valid_runs.append(run_dir)
                
    if not valid_runs:
        return None

    valid_runs.sort()
    latest_run = valid_runs[-1]
    
    policy = np.load(latest_run / "policy.npy")
    states = [tuple(s) for s in np.load(latest_run / "states.npy")]
    state_to_id = {s: i for i, s in enumerate(states)}
    
    return policy, state_to_id

def evaluate_agent(agent, env: ParkourEnv, start_state: tuple, max_steps: int, gamma: float) -> dict:
    """Выполняет одну эпоху для агента и возвращает дисконтированные метрики."""
    agent.reset(start_state, env)
    state = start_state
    
    total_reward = 0.0
    discount = 1.0  # Начальный дисконт gamma^0 = 1
    steps = 0
    done = False
    dead = False

    while not done and steps < max_steps:
        # ЗАЩИТА: Если это обученная модель, ограничиваем HP её максимумом (env.hp_start),
        # чтобы не было KeyError, если бонусный HP больше обученного.
        if isinstance(agent, DPAgent):
            clipped_hp = min(state[2], env.hp_start)
            obs_state = (state[0], state[1], clipped_hp)
        else:
            obs_state = state

        action = agent.act(obs_state, env)
        next_state, reward, done, dead = env.step(state, action)
        
        # Дисконтированная награда: R_total = r0 + g*r1 + g^2*r2 ...
        total_reward += discount * reward
        discount *= gamma
        
        steps += 1
        state = next_state

    victory = done and not dead
    penalised_steps = steps if victory else max_steps

    return {
        "victory": 1 if victory else 0,
        "steps": penalised_steps,
        "reward": total_reward
    }

def main():
    log_file_path = PROJECT_ROOT / f"compare_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    
    def log_print(text=""):
        print(text)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    log_print("="*80)
    log_print(f" PARKOUR EVALUATION (Discount Factor Gamma: {GAMMA})")
    log_print("="*80)

    env_base_cfg = load_yaml(PROJECT_ROOT / "configs/env.yaml")
    landscape_dir = PROJECT_ROOT / "landscape"
    
    landscape_paths = sorted([p for p in landscape_dir.iterdir() if p.is_dir() and p.name.startswith("landscape_")])
    if not landscape_paths:
        log_print("No landscapes found in 'landscape/' directory.")
        return

    algo_names = [
        "Random", 
        "Safest Path", 
        "Budget Greedy",
        "Shortest Path",
        "value_iteration", 
        "policy_iteration", 
    ]
    
    results = {name: defaultdict(lambda: {"sr": [], "steps": [], "reward": []}) for name in algo_names}

    for l_path in landscape_paths:
        landscape_id = int(l_path.name.split("_")[1])
        log_print(f"\nEvaluating on Landscape ID: {landscape_id}")
        
        env_config = env_base_cfg.copy()
        env_config["landscape_id"] = landscape_id
        env = ParkourEnv(env_config)
        actions = env.get_actions()

        min_hp_map = np.load(l_path / "min_hp.npy")
        eval_cells = np.load(l_path / "eval_cells.npy")
        
        agents_dict = {
            "Random": RandomAgent(),
            "Safest Path": SafestPathAgent(),
            "Budget Greedy": BudgetAwareGreedyAgent(),
            "Shortest Path": ShortestPathAgent(),
        }

        for dp_algo in ["value_iteration", "policy_iteration"]:
            model_data = find_trained_dp_agent(dp_algo, landscape_id)
            if model_data:
                policy, state_to_id = model_data
                agents_dict[dp_algo] = DPAgent(policy, state_to_id, actions)
            else:
                log_print(f"  [Warning] No model found for {dp_algo} on landscape {landscape_id}")

        hp_bonus_pct = env_base_cfg.get("hp_bonus_pct", 0.0)

        for (i, j) in eval_cells:
            # Считаем HP с бонусом
            base_min_hp = int(min_hp_map[i, j])
            start_hp = int(base_min_hp * (1 + hp_bonus_pct))
            start_state = (int(i), int(j), start_hp)
            
            for agent_name, agent in agents_dict.items():
                metrics = evaluate_agent(agent, env, start_state, MAX_STEPS, GAMMA)
                results[agent_name][landscape_id]["sr"].append(metrics["victory"])
                
                # Записываем шаги только если победа (чтобы считать честное среднее позже)
                if metrics["victory"] == 1:
                    results[agent_name][landscape_id]["steps"].append(metrics["steps"])
                else:
                    results[agent_name][landscape_id]["steps"].append(np.nan)
                    
                results[agent_name][landscape_id]["reward"].append(metrics["reward"])
                
        # Вывод результатов для текущей карты
        log_print("-" * 75)
        log_print(f"{'Agent Name':<28} | {'Map SR':<8} | {'Map Steps':<10} | {'Map Reward':<10}")
        log_print("-" * 75)
        for agent_name in algo_names:
            if not results[agent_name][landscape_id]["sr"]: continue
            map_sr = np.mean(results[agent_name][landscape_id]["sr"])
            # np.nanmean игнорирует провалы в статистике шагов
            map_steps = np.nanmean(results[agent_name][landscape_id]["steps"])
            map_steps = 0 if np.isnan(map_steps) else map_steps
            map_rew = np.mean(results[agent_name][landscape_id]["reward"])
            log_print(f"{agent_name:<28} | {map_sr:>7.1%} | {map_steps:>9.1f} | {map_rew:>10.1f}")


    # Финальная агрегация
    log_print("\n\n" + "="*80)
    log_print(" FINAL AGGREGATED RESULTS (Averaged across all maps)")
    log_print("="*80)
    log_print(f"{'Agent Name':<28} | {'Total SR':<10} | {'Avg Steps':<10} | {'Avg Reward':<10}")
    log_print("-" * 80)

    final_stats = []
    for agent_name in algo_names:
        map_srs, map_steps_list, map_rews = [], [], []
        
        for l_id in results[agent_name]:
            if not results[agent_name][l_id]["sr"]: continue
            map_srs.append(np.mean(results[agent_name][l_id]["sr"]))
            
            # Собираем среднее по шагам с карты
            m_steps = np.nanmean(results[agent_name][l_id]["steps"])
            if not np.isnan(m_steps):
                map_steps_list.append(m_steps)
            
            map_rews.append(np.mean(results[agent_name][l_id]["reward"]))
                
        if not map_srs: continue
        
        avg_sr = np.mean(map_srs)
        avg_step = np.mean(map_steps_list) if map_steps_list else 0
        avg_rew = np.mean(map_rews)
        final_stats.append((agent_name, avg_sr, avg_step, avg_rew))

    final_stats.sort(key=lambda x: x[3], reverse=True)

    for name, sr, steps, rew in final_stats:
        log_print(f"{name:<28} | {sr:>9.1%} | {steps:>10.1f} | {rew:>10.1f}")
        
    log_print("="*80)
    print(f"\nFull logs saved to: {log_file_path.name}")

if __name__ == "__main__":
    main()