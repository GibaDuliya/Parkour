# FILE: ./run/evaluate_gif.py
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
from src.utils.metrics import rollout_policy

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_gif_for_agent(agent_dir: Path):
    """Run rollout for the agent and save a polished GIF animation."""
    print(f"\n[{agent_dir.name}] Generating GIF...")
    
    # 1. Load agent artifacts
    try:
        policy = np.load(agent_dir / "policy.npy")
        states_arr = np.load(agent_dir / "states.npy")
        states = [tuple(s) for s in states_arr]
        meta = load_yaml(agent_dir / "meta.yaml")
    except FileNotFoundError as e:
        print(f"Error loading files in {agent_dir}: {e}")
        return

    state_to_id = {s: i for i, s in enumerate(states)}

    # 2. Setup Environment
    env_config = load_yaml(PROJECT_ROOT / "configs/env.yaml")
    env_config["landscape_id"] = meta["landscape_id"]
    env = ParkourEnv(env_config)
    actions = env.get_actions()

    # 3. Rollout
    rollout = rollout_policy(env, policy, states, actions, state_to_id)
    
    hp_start = getattr(env, "hp_init", env.hp_start)
    start_state = (0, 0, hp_start)
    
    # Build history for animation
    path_states = [start_state]
    rewards_history = [0.0]
    
    current_reward = 0.0
    for (next_state, _action, step_reward) in rollout["trajectory"]:
        path_states.append(next_state)
        current_reward += step_reward
        rewards_history.append(current_reward)

    # Pad the end with a few duplicate frames so the GIF pauses on the final screen
    pause_frames = 10
    path_states.extend([path_states[-1]] * pause_frames)
    rewards_history.extend([rewards_history[-1]] * pause_frames)

    is_victory = rollout['victory']
    print(f"[{meta['algorithm']}] Steps: {rollout['steps']}, Reward: {rollout['total_reward']}, Victory: {is_victory}")

    # 4. Visualization Setup
    height_map = env.height_map
    rows, cols = height_map.shape

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    fig.patch.set_facecolor('#f8f9fa')  # Light background
    ax.set_facecolor('#f8f9fa')
    
    # Draw height map
    im = ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper", alpha=0.85)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Building Height", rotation=270, labelpad=15, fontweight='bold')

    if rows <= 20 and cols <= 20:
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, int(height_map[i, j]), ha="center", va="center", 
                        color="black", fontsize=9, alpha=0.6)

    # Markers for Start and Goal
    ax.scatter(0, 0, c="lime", s=250, marker="s", edgecolors="black", linewidth=1.5, label="Start", zorder=3)
    ax.scatter(cols - 1, rows - 1, c="gold", s=350, marker="*", edgecolors="black", linewidth=1.5, label="Goal", zorder=3)
    
    # Trail line and Agent marker
    trail_line, = ax.plot([], [], color='white', linewidth=2, linestyle='--', alpha=0.7, zorder=4)
    agent_dot, = ax.plot([], [], marker='o', color='crimson', markersize=14, markeredgecolor='white', markeredgewidth=2, zorder=5, label='Agent')
    
    # ==========================================
    # ПЛАШКА С НАЗВАНИЕМ АЛГОРИТМА (WATERMARK)
    # ==========================================
    algo_name = meta['algorithm'].replace("_", " ").upper()
    ax.text(0.5, 0.95, algo_name, 
            transform=ax.transAxes, fontsize=14, fontweight='bold', color='white', 
            ha='center', va='top', zorder=10,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))

    # Title/HUD
    title_text = ax.set_title("", fontsize=12, family='monospace', pad=15)
    
    # Grid styling
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
    ax.tick_params(which="minor", size=0)
    
    ax.set_xticks(np.arange(0, cols, max(1, cols // 10)))
    ax.set_yticks(np.arange(0, rows, max(1, rows // 10)))
    ax.set_xlabel("i", fontweight='bold')
    ax.set_ylabel("j", fontweight='bold')
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()

    # Animation init
    def init():
        agent_dot.set_data([], [])
        trail_line.set_data([], [])
        title_text.set_text("")
        return agent_dot, trail_line, title_text

    # Frame update function
    def update(frame):
        state = path_states[frame]
        rew = rewards_history[frame]
        i, j, hp = state
        
        # Draw trail
        current_path = path_states[:frame+1]
        trail_j = [s[1] for s in current_path]
        trail_i = [s[0] for s in current_path]
        trail_line.set_data(trail_j, trail_i)
        
        # Update agent position
        agent_dot.set_data([j], [i])
        
        # Determine Status
        is_last_real_frame = (frame >= len(rollout["trajectory"]))
        
        if hp <= 0:
            status = "DEAD"
            color = "darkred"
            agent_dot.set_color('black')
            agent_dot.set_marker('X')
            agent_dot.set_markersize(16)
        elif is_last_real_frame and is_victory:
            status = "VICTORY!"
            color = "green"
            agent_dot.set_color('gold')
            agent_dot.set_marker('*')
            agent_dot.set_markersize(20)
        else:
            status = "ALIVE"
            color = "black"
            agent_dot.set_color('crimson')
            agent_dot.set_marker('o')
            agent_dot.set_markersize(14)

        # Update HUD Text (Убрали название алгоритма из текста, так как оно теперь на плашке)
        real_step = min(frame, len(rollout["trajectory"]))
        hud = (
            f"Step: {real_step:03d} | HP: {hp:02d}/{hp_start:02d}\n"
            f"Reward: {rew:+.1f} | Status: {status}"
        )
        
        title_text.set_text(hud)
        title_text.set_color(color)
        
        return agent_dot, trail_line, title_text

    # Generate Animation
    ani = FuncAnimation(
        fig, update, frames=len(path_states),
        init_func=init, blit=False, repeat=False
    )

    # ==========================================
    # СОХРАНЯЕМ 2 ВЕРСИИ GIF (1x и 2x скорость)
    # ==========================================
    try:
        # 1. Обычная скорость (4 кадра в секунду)
        gif_normal = agent_dir / "gameplay.gif"
        ani.save(gif_normal, writer=PillowWriter(fps=4))
        print(f"Saved Normal Speed GIF -> {gif_normal}")

        # 2. Ускоренная версия (8 кадров в секунду)
        gif_fast = agent_dir / "gameplay_2x.gif"
        ani.save(gif_fast, writer=PillowWriter(fps=8))
        print(f"Saved 2x Speed GIF     -> {gif_fast}")
    except Exception as e:
        print(f"Failed to save GIF: {e}. (Make sure 'Pillow' is installed: pip install Pillow)")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate polished gameplay GIFs for trained Parkour agents.")
    parser.add_argument("--all", action="store_true", help="Evaluate and generate GIFs for all agents in agents/ dir")
    parser.add_argument("--agent", type=str, help="Path to a specific agent directory (e.g., agents/2026-03-16_12-00-00)")
    args = parser.parse_args()

    if args.all:
        agents_dir = PROJECT_ROOT / "agents"
        if not agents_dir.exists():
            print("Directory 'agents/' not found. Please run training first (run/train.py).")
            return
        
        agent_paths = [p for p in agents_dir.iterdir() if p.is_dir() and (p / "meta.yaml").exists()]
        if not agent_paths:
            print("No trained agents found in 'agents/'.")
            return

        for path in agent_paths:
            make_gif_for_agent(path)

    elif args.agent:
        agent_path = Path(args.agent)
        if not agent_path.is_absolute():
            agent_path = PROJECT_ROOT / agent_path
        if not agent_path.exists():
            print(f"Agent directory {agent_path} does not exist.")
            return
        make_gif_for_agent(agent_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()