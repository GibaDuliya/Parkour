import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Настройка путей
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
# Убедись, что ShortestPathAgent добавлен в baselines.py
from src.algorithms.baselines import ShortestPathAgent

MAX_STEPS = 2000

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate_agent(agent, env: ParkourEnv, start_state: tuple, max_steps: int) -> dict:
    """Выполняет одну эпоху для агента и возвращает метрики."""
    agent.reset(start_state, env)
    state = start_state
    
    total_reward = 0.0
    steps = 0
    done = False
    dead = False

    while not done and steps < max_steps:
        safe_state = (int(state[0]), int(state[1]), int(state[2]))
        action = agent.act(safe_state, env)
        next_state, reward, done, dead = env.step(safe_state, action)
        
        total_reward += reward
        steps += 1
        state = next_state

    victory = done and not dead
    penalised_steps = steps if victory else max_steps

    return {
        "victory": 1 if victory else 0,
        # Возвращаем None для шагов, если это провал, чтобы потом отфильтровать
        "steps": steps if victory else None, 
        "reward": total_reward
    }

def main():
    print("="*80)
    print(" HP BONUS EXPERIMENT: SHORTEST PATH AGENT ONLY ")
    print("="*80)

    # Задаем массив надбавок к HP
    hp_bonuses = [0, 10, 20, 30, 50, 60]
    max_bonus = max(hp_bonuses)

    env_base_cfg = load_yaml(PROJECT_ROOT / "configs/env.yaml")
    landscape_dir = PROJECT_ROOT / "landscape"
    
    landscape_paths = sorted([p for p in landscape_dir.iterdir() if p.is_dir() and p.name.startswith("landscape_")])
    if not landscape_paths:
        print("No landscapes found in 'landscape/' directory.")
        return

    # ТОЛЬКО Shortest Path
    agent_classes = {
        "Shortest Path": ShortestPathAgent
    }

    results = {name: defaultdict(lambda: {"sr": [], "steps": [], "reward": []}) for name in agent_classes}

    for l_path in landscape_paths:
        landscape_id = int(l_path.name.split("_")[1])
        print(f"\nProcessing Landscape ID: {landscape_id}")
        
        eval_cells = np.load(l_path / "eval_cells.npy")
        min_hp_map = np.load(l_path / "min_hp.npy")
        height_map = np.load(l_path / "height_map.npy")
        l_cfg = load_yaml(l_path / "config.yaml")
        
        base_max_hp = int(np.max(min_hp_map))
        custom_hp_start = base_max_hp + max_bonus
        
        env_config = {
            "height_map": height_map.tolist(),
            "hp_start": custom_hp_start,
            "rewards": env_base_cfg["rewards"],
            "max_jump_up": l_cfg["max_jump_up"],
            "safe_jump_down": l_cfg["safe_jump_down"]
        }
        
        env = ParkourEnv(env_config)

        for bonus in hp_bonuses:
            # Создаем агента один раз для этого бонуса
            agent = ShortestPathAgent()
            
            for cell in eval_cells:
                i, j = int(cell[0]), int(cell[1])
                base_hp = int(min_hp_map[i, j])
                start_hp = base_hp + bonus
                start_state = (i, j, start_hp)
                
                metrics = evaluate_agent(agent, env, start_state, MAX_STEPS)
                results["Shortest Path"][bonus]["sr"].append(metrics["victory"])
                results["Shortest Path"][bonus]["steps"].append(metrics["steps"])
                results["Shortest Path"][bonus]["reward"].append(metrics["reward"])

        
    print("\nSimulations completed. Plotting...")

    # Подготовка данных
    bonuses = hp_bonuses
    sr_vals = []
    step_vals = [] # Теперь здесь будут средние только по успешным
    reward_vals = []

    for b in bonuses:
        # Успешные эпизоды для данного бонуса
        success_indices = [idx for idx, val in enumerate(results["Shortest Path"][b]["sr"]) if val == 1]
        
        # 1. SR
        sr_vals.append(np.mean(results["Shortest Path"][b]["sr"]) * 100)
        
        # 2. Avg Steps (только по успешным)
        if success_indices:
            all_steps = np.array(results["Shortest Path"][b]["steps"])
            step_vals.append(np.mean(all_steps[success_indices]))
        else:
            step_vals.append(np.nan) # Если побед 0, ставим NaN
            
        # 3. Avg Reward (по всем, как и раньше, так как это общая метрика)
        reward_vals.append(np.mean(results["Shortest Path"][b]["reward"]))

    # Построение графиков
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Shortest Path Agent Sensitivity to HP Bonus", fontsize=16, fontweight='bold')

    # Цвета и стиль
    color = "darkorange"
    marker = "D"

    # 1. Success Rate
    axes[0].plot(bonuses, sr_vals, marker=marker, color=color, linewidth=2.5)
    axes[0].set_title("Success Rate (%)", fontsize=14)
    axes[0].set_ylabel("Victory %")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Avg Steps
    axes[1].plot(bonuses, step_vals, marker=marker, color=color, linewidth=2.5)
    axes[1].set_title("Average Steps", fontsize=14)
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Avg Reward
    axes[2].plot(bonuses, reward_vals, marker=marker, color=color, linewidth=2.5)
    axes[2].set_title("Average Total Reward", fontsize=14)
    axes[2].set_ylabel("Reward")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    for ax in axes:
        ax.set_xlabel("Extra HP Bonus")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = PROJECT_ROOT / f"shortest_path_experiment_{timestamp}.png"
    plt.savefig(save_path, dpi=200)
    print(f"\nGraph saved to: {save_path.name}")
    plt.show()

if __name__ == "__main__":
    main()