import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import animation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
from src.utils.metrics import rollout_policy
from run.evaluate import _build_state_mapping  # reuse helper


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _rollout_noisy(
    env: ParkourEnv,
    policy: np.ndarray,
    states: list,
    actions: list,
    state_to_id: dict,
    eps: float,
    max_steps: int = 10000,
) -> dict:
    """Epsilon-greedy rollout to introduce stochasticity in the path."""
    # старт как в rollout_policy
    state = (0, 0, getattr(env, "hp_init", env.hp_start))
    trajectory = []
    total_reward = 0.0
    steps = 0
    done = False
    dead = False

    rng = np.random.default_rng(0)  # фиксируем для воспроизводимости

    while not done and steps < max_steps:
        s_id = state_to_id.get(state)
        if s_id is None:
            break

        if rng.random() < eps:
            a_id = rng.integers(0, len(actions))
        else:
            a_id = int(policy[s_id])

        action = actions[a_id]
        next_state, reward, done, dead = env.step(state, action)
        trajectory.append((next_state, action, reward))
        total_reward += float(reward)
        steps += 1
        state = next_state

    final_hp = state[2]
    victory = done and not dead
    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "steps": steps,
        "final_hp": final_hp,
        "victory": victory,
    }


def _load_agent(agent_dir: Path, eps: float) -> Tuple[str, np.ndarray, list, dict, ParkourEnv]:
    """Load agent artifacts and rollout in current env (with epsilon-greedy noise)."""
    agent_path = agent_dir
    V = np.load(agent_path / "value_function.npy")
    policy = np.load(agent_path / "policy.npy")
    states_arr = np.load(agent_path / "states.npy")
    states = _build_state_mapping(states_arr)
    meta = load_yaml(agent_path / "meta.yaml")

    env_config = load_yaml(PROJECT_ROOT / "configs/env.yaml")
    env_config["landscape_id"] = meta["landscape_id"]
    env = ParkourEnv(env_config)

    actions = env.get_actions()
    state_to_id = {s: i for i, s in enumerate(states)}
    rollout = _rollout_noisy(env, policy, states, actions, state_to_id, eps=eps)

    return meta["algorithm"], env, rollout["trajectory"], rollout


def evaluate_all(agent_dirs: List[str]) -> None:
    if len(agent_dirs) != 3:
        raise ValueError("Pass exactly three agent directories.")

    agent_paths = [PROJECT_ROOT / d for d in agent_dirs]

    # разные eps, чтобы стокхастика была различимой
    epsilons = [0.05, 0.15, 0.3]

    algos_envs_trajs: List[Tuple[str, ParkourEnv, list, dict]] = []
    for p, eps in zip(agent_paths, epsilons):
        if not p.exists():
            raise FileNotFoundError(p)
        algos_envs_trajs.append(_load_agent(p, eps=eps))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False, sharey=False)
    fig.patch.set_facecolor("#000000")

    artists = []
    max_len = 0

    for ax, (algo_name, env, traj, rollout) in zip(axes, algos_envs_trajs):
        height_map = env.height_map
        rows, cols = height_map.shape

        # --- копия логики из evaluate._make_gif ---
        path_i = [0]
        path_j = [0]
        hps = []
        rewards = []
        cum_rewards = []
        total_r = 0.0
        for (state, _action, reward) in traj:
            path_i.append(state[0])
            path_j.append(state[1])
            hp = state[2] if len(state) > 2 else None
            hps.append(hp)
            total_r += float(reward)
            rewards.append(float(reward))
            cum_rewards.append(total_r)

        if len(path_i) <= 1:
            continue

        pad = 2
        i_min = max(0, min(path_i) - pad)
        i_max = min(rows - 1, max(path_i) + pad)
        j_min = max(0, min(path_j) - pad)
        j_max = min(cols - 1, max(path_j) + pad)

        sub_height = height_map[i_min : i_max + 1, j_min : j_max + 1]
        path_i_sub = [i - i_min for i in path_i]
        path_j_sub = [j - j_min for j in path_j]
        sub_rows, sub_cols = sub_height.shape

        im = ax.imshow(
            sub_height,
            cmap="viridis",
            aspect="equal",
            origin="upper",
            interpolation="nearest",
        )

        # Colorbar справа, как в одиночном eval
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Height", color="#FFFFFF", fontsize=8)
        for tick in cbar.ax.get_yticklabels():
            tick.set_color("#FFFFFF")
            tick.set_fontsize(7)

        ax.set_xlim(-0.5, sub_cols - 0.5)
        ax.set_ylim(sub_rows - 0.5, -0.5)

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

        ax.set_xlabel("j", color="#DDDDDD")
        ax.set_ylabel("i", color="#DDDDDD")
        ax.set_title(algo_name.replace("_", " "), color="#FFFFFF", fontsize=11, pad=10)

        start_xy = (path_j_sub[0], path_i_sub[0])
        goal_xy = (path_j_sub[-1], path_i_sub[-1])
        ax.scatter(
            [start_xy[0]],
            [start_xy[1]],
            c="#FFFFFF",
            s=80,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )
        ax.scatter(
            [goal_xy[0]],
            [goal_xy[1]],
            c="#FFC107",
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

        path_line, = ax.plot([], [], color="#FF5252", linewidth=2.0, alpha=0.9, zorder=3)
        agent_point = ax.scatter([], [], c="#FFFFFF", s=80, marker="o", zorder=5, edgecolors="#000000")

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

        artists.append(
            (
                path_line,
                agent_point,
                stats_text,
                path_i,
                path_j,
                path_i_sub,
                path_j_sub,
                hps,
                rewards,
                cum_rewards,
            )
        )

        max_len = max(max_len, len(path_i))

    # Если не набралось ни одной непустой траектории, не пытаемся строить GIF.
    if not artists or max_len <= 1:
        plt.close(fig)
        print("No non-empty trajectories found; combined GIF was not created.")
        return

    def init():
        draw = []
        for (
            path_line,
            agent_point,
            stats_text,
            path_i,
            path_j,
            path_i_sub,
            path_j_sub,
            *_,
        ) in artists:
            path_line.set_data([], [])
            agent_point.set_offsets(np.empty((0, 2)))
            stats_text.set_text("")
            draw.extend([path_line, agent_point, stats_text])
        return draw

    def update(frame_idx: int):
        draw_artists = []
        for (
            path_line,
            agent_point,
            stats_text,
            path_i,
            path_j,
            path_i_sub,
            path_j_sub,
            hps,
            rewards,
            cum_rewards,
        ) in artists:
            t = min(frame_idx, len(path_i) - 1)
            xs = path_j_sub[: t + 1]
            ys = path_i_sub[: t + 1]
            path_line.set_data(xs, ys)
            agent_point.set_offsets(np.array([[xs[-1], ys[-1]]]))

            step_no = t
            if t == 0 or not hps:
                stats_text.set_text(f"step: {step_no}")
            else:
                idx = min(t - 1, len(hps) - 1)
                hp_val = hps[idx]
                r_step = rewards[idx]
                r_cum = cum_rewards[idx]
                hp_str = "NA" if hp_val is None else f"{hp_val}"
                stats_text.set_text(
                    f"step: {step_no}   hp: {hp_str}\n"
                    f"reward: {r_step:.1f}   total R: {r_cum:.1f}"
                )

            draw_artists.extend([path_line, agent_point, stats_text])
        return draw_artists

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=max_len,
        interval=250,
        blit=True,
        repeat=False,
    )

    out_path = PROJECT_ROOT / "agents" / "all_algorithms.gif"
    anim.save(out_path, writer="pillow", fps=4)
    plt.close(fig)
    print(f"Combined GIF saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate three agents (VI, PI, QL-VI) side by side"
    )
    parser.add_argument(
        "agent_dirs",
        nargs=3,
        help="Three agent directories, e.g. agents/.. agents/.. agents/..",
    )
    args = parser.parse_args()
    evaluate_all(args.agent_dirs)

