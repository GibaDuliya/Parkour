import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import animation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
from src.utils.metrics import rollout_policy


def load_yaml(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_state_mapping(states_arr: np.ndarray) -> list[tuple[int, int, int]]:
    return [tuple(s) for s in states_arr]


def _make_gif(
    height_map: np.ndarray,
    trajectory: list,
    save_path: Path,
    fps: int = 6,
) -> None:
    """2D heatmap with zoomed path and per-step stats overlay."""
    rows, cols = height_map.shape

    # Extract positions, HP and rewards along the path (include start state)
    path_i = [0]
    path_j = [0]
    hps = []
    rewards = []
    cum_rewards = []
    total_r = 0.0
    for (state, _action, reward) in trajectory:
        path_i.append(state[0])
        path_j.append(state[1])
        hp = state[2] if len(state) > 2 else None
        hps.append(hp)
        total_r += float(reward)
        rewards.append(float(reward))
        cum_rewards.append(total_r)

    if len(path_i) <= 1:
        return

    # Zoomed crop around trajectory so клетки крупнее
    pad = 2
    i_min = max(0, min(path_i) - pad)
    i_max = min(rows - 1, max(path_i) + pad)
    j_min = max(0, min(path_j) - pad)
    j_max = min(cols - 1, max(path_j) + pad)

    sub_height = height_map[i_min : i_max + 1, j_min : j_max + 1]

    # Coordinates в системе кропа
    path_i_sub = [i - i_min for i in path_i]
    path_j_sub = [j - j_min for j in path_j]
    sub_rows, sub_cols = sub_height.shape

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    im = ax.imshow(
        sub_height,
        cmap="viridis",
        aspect="equal",
        origin="upper",
        interpolation="nearest",
    )

    # Ensure full cropped grid (включая угол с goal) всегда в кадре
    ax.set_xlim(-0.5, sub_cols - 0.5)
    ax.set_ylim(sub_rows - 0.5, -0.5)

    # Sparse ticks so не зашумляет картинку
    max_ticks = 12
    step_x = max(1, sub_cols // max_ticks)
    step_y = max(1, sub_rows // max_ticks)
    xs_ticks = np.arange(0, sub_cols, step_x)
    ys_ticks = np.arange(0, sub_rows, step_y)
    ax.set_xticks(xs_ticks)
    ax.set_yticks(ys_ticks)
    ax.set_xticklabels((xs_ticks + j_min).astype(int), color="#DDDDDD", fontsize=7)
    ax.set_yticklabels((ys_ticks + i_min).astype(int), color="#DDDDDD", fontsize=7)

    for spine in ax.spines.values():
        spine.set_color("#DDDDDD")

    ax.set_xlabel("j (horizontal position)", color="#DDDDDD")
    ax.set_ylabel("i (vertical position)", color="#DDDDDD")
    ax.set_title("Parkour agent trajectory over city height map", color="#FFFFFF", fontsize=12, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height", color="#FFFFFF")
    for tick in cbar.ax.get_yticklabels():
        tick.set_color("#FFFFFF")

    # Static markers
    start_xy = (path_j_sub[0], path_i_sub[0])
    goal_xy = (path_j_sub[-1], path_i_sub[-1])
    ax.scatter(
        [start_xy[0]],
        [start_xy[1]],
        c="#4CAF50",
        s=120,
        marker="s",
        zorder=4,
        edgecolors="black",
        linewidths=0.8,
    )
    ax.scatter(
        [goal_xy[0]],
        [goal_xy[1]],
        c="#FFC107",
        s=220,
        marker="*",
        zorder=6,
        edgecolors="black",
        linewidths=1.0,
    )

    # Animated objects
    path_line, = ax.plot([], [], color="#FF5252", linewidth=2.0, alpha=0.9)
    agent_point = ax.scatter([], [], c="#FFFFFF", s=80, marker="o", zorder=5, edgecolors="#000000")

    max_tail = max(3, len(path_i) // 8)

    # Text overlay inside axes (bottom‑left): step, HP, reward, cumulative reward
    stats_text = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        color="#FFFFFF",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#000000CC", edgecolor="none"),
    )

    def init():
        path_line.set_data([], [])
        agent_point.set_offsets(np.empty((0, 2)))
        stats_text.set_text("")
        return path_line, agent_point, stats_text

    def update(frame_idx: int):
        if frame_idx >= len(path_i):
            frame_idx = len(path_i) - 1

        xs = path_j_sub[: frame_idx + 1]
        ys = path_i_sub[: frame_idx + 1]

        # Full path seen so far (не режем хвост)
        path_line.set_data(xs, ys)

        agent_point.set_offsets(np.array([[xs[-1], ys[-1]]]))

        # stats: шаги считаем от 0, hp и reward — по доступным данным
        step_no = frame_idx
        if frame_idx == 0 or not hps:
            stats_text.set_text(f"step: {step_no}")
        else:
            idx = min(frame_idx - 1, len(hps) - 1)
            hp_val = hps[idx]
            r_step = rewards[idx]
            r_cum = cum_rewards[idx]
            hp_str = "NA" if hp_val is None else f"{hp_val}"
            stats_text.set_text(
                f"step: {step_no}   hp: {hp_str}\n"
                f"reward: {r_step:.1f}   total R: {r_cum:.1f}"
            )

        return path_line, agent_point, stats_text

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(path_i),
        interval=int(1000 / fps),
        blit=True,
        repeat=False,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def evaluate_agent(agent_dir: str) -> None:
    """Custom evaluation with pretty GIF visualisation."""
    agent_path = Path(agent_dir)
    if not agent_path.is_absolute():
        agent_path = PROJECT_ROOT / agent_path

    if not agent_path.exists():
        raise FileNotFoundError(f"Agent directory not found: {agent_path}")

    # 1. Load agent artifacts
    V = np.load(agent_path / "value_function.npy")
    policy = np.load(agent_path / "policy.npy")
    states_arr = np.load(agent_path / "states.npy")
    states = _build_state_mapping(states_arr)
    meta = load_yaml(agent_path / "meta.yaml")

    state_to_id = {s: i for i, s in enumerate(states)}

    # 2. Reconstruct environment
    env_config = load_yaml(PROJECT_ROOT / "configs/env.yaml")
    env_config["landscape_id"] = meta["landscape_id"]
    env = ParkourEnv(env_config)

    # 3. Rollout
    actions = env.get_actions()
    rollout = rollout_policy(env, policy, states, actions, state_to_id)

    print("=" * 60)
    print(f"Agent dir   : {agent_path}")
    print(f"Algorithm   : {meta['algorithm']}")
    print(f"Landscape   : {meta['landscape_id']}")
    print(f"Steps       : {rollout['steps']}")
    print(f"Total reward: {rollout['total_reward']:.3f}")
    print(f"Final HP    : {rollout['final_hp']}")
    print(f"Victory     : {rollout['victory']}")
    print("=" * 60)

    # 4. Visuals
    out_dir = agent_path / "eval_custom"
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_path = out_dir / "trajectory.gif"
    _make_gif(env.height_map, rollout["trajectory"], gif_path, fps=4)
    print(f"GIF saved to {gif_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parkour RL — Custom evaluate saved agent")
    parser.add_argument(
        "agent_dir",
        help="Path to agent folder (e.g. agents/2026-03-16_12-00-00)",
    )
    args = parser.parse_args()
    evaluate_agent(args.agent_dir)
