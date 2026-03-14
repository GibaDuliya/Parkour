from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.environment.parkour_env import Action

_MAX_TICKS = 20
_MAX_CELL_LABELS = 20


def _set_ticks(ax, rows: int, cols: int) -> None:
    x_step = max(1, cols // _MAX_TICKS)
    y_step = max(1, rows // _MAX_TICKS)
    ax.set_xticks(np.arange(0, cols, x_step))
    ax.set_yticks(np.arange(0, rows, y_step))
    ax.set_xticklabels(np.arange(0, cols, x_step))
    ax.set_yticklabels(np.arange(0, rows, y_step))


def plot_height_map(height_map: np.ndarray, save_path: str | Path | None = None) -> None:
    rows, cols = height_map.shape
    fig, ax = plt.subplots()
    im = ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax, label="Height")

    if rows <= _MAX_CELL_LABELS and cols <= _MAX_CELL_LABELS:
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, int(height_map[i, j]), ha="center", va="center",
                        color="black", fontsize=10)

    _set_ticks(ax, rows, cols)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_value_function(
    V: np.ndarray,
    height_map: np.ndarray,
    min_hp_map: np.ndarray,
    state_to_id: dict,
    save_path: str | Path | None = None,
) -> None:
    """Plot value function heatmap using min HP per cell.

    For each cell (i, j), looks up V(i, j, min_hp_map[i, j]).

    Args:
        V: (n_states,) value function array
        height_map: grid shape reference
        min_hp_map: (rows, cols) array of min starting HP per cell
        state_to_id: state tuple -> state index
        save_path: if set, save figure to this path before showing
    """
    rows, cols = height_map.shape
    grid = np.full((rows, cols), np.nan)
    for i in range(rows):
        for j in range(cols):
            hp = int(min_hp_map[i, j])
            if hp < 0:
                continue
            state = (i, j, hp)
            if state in state_to_id:
                grid[i, j] = V[state_to_id[state]]

    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap="viridis", aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax, label="V(s)")

    _set_ticks(ax, rows, cols)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Value function (min HP per cell)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_policy(
    policy: np.ndarray,
    height_map: np.ndarray,
    hp: int,
    state_to_id: dict,
    actions: list,
    save_path: str | Path | None = None,
) -> None:
    action_uv = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
    }

    rows, cols = height_map.shape
    Xc, Yc = np.meshgrid(np.arange(cols), np.arange(rows))
    xs, ys, us, vs = [], [], [], []

    for i in range(rows):
        for j in range(cols):
            state = (i, j, hp)
            if state in state_to_id:
                s_id = state_to_id[state]
                a_id = int(policy[s_id])
                action = actions[a_id]
                u, v = action_uv[action]
                xs.append(Xc[i, j])
                ys.append(Yc[i, j])
                us.append(u * 0.4)
                vs.append(v * 0.4)

    fig, ax = plt.subplots()
    ax.imshow(
        height_map,
        cmap="YlGnBu",
        aspect="equal",
        origin="upper",
        alpha=0.7,
        extent=[-0.5, cols - 0.5, rows - 0.5, -0.5],
    )
    ax.quiver(
        xs, ys, us, vs,
        color="darkblue",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        angles="xy",
        width=0.012,
        headwidth=4,
        headlength=5,
    )
    _set_ticks(ax, rows, cols)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(f"Policy (hp={hp})")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory(trajectory: list, height_map: np.ndarray, save_path: str | Path | None = None) -> None:
    rows, cols = height_map.shape
    path_i = [0]
    path_j = [0]
    for (state, _action, _reward) in trajectory:
        path_i.append(state[0])
        path_j.append(state[1])

    fig, ax = plt.subplots()
    ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper")
    if path_i and path_j:
        ax.plot(path_j, path_i, "r-", linewidth=2, label="path")
        ax.scatter([path_j[0]], [path_i[0]], c="lime", s=150, marker="s", label="start", zorder=5)
        ax.scatter([path_j[-1]], [path_i[-1]], c="gold", s=150, marker="*", label="end", zorder=5)
    _set_ticks(ax, rows, cols)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Trajectory")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_convergence(info: dict, save_path: str | Path | None = None) -> None:
    delta_history = info.get("delta_history", [])
    if not delta_history:
        return

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(delta_history) + 1), delta_history, "b-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |V_new - V_old|")
    ax.set_title("Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
