import numpy as np
import matplotlib.pyplot as plt

from src.environment.parkour_env import Action


def plot_height_map(height_map: np.ndarray) -> None:
    """Plot 2D heatmap of building heights.

    Args:
        height_map: 8x8 array of heights
    """
    fig, ax = plt.subplots()
    im = ax.imshow(height_map, cmap="terrain", aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax, label="Height")

    rows, cols = height_map.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, int(height_map[i, j]), ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.tight_layout()
    plt.show()


def plot_value_function(V: dict, height_map: np.ndarray, hp: int) -> None:
    """Plot value function heatmap for a fixed HP level.

    Args:
        V: {(i, j, hp): float} value function
        height_map: 8x8 array for reference
        hp: HP level to visualize
    """
    rows, cols = height_map.shape
    grid = np.full((rows, cols), np.nan)
    for i in range(rows):
        for j in range(cols):
            state = (i, j, hp)
            if state in V:
                grid[i, j] = V[state]

    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap="viridis", aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax, label="V(s)")

    for i in range(rows):
        for j in range(cols):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(f"Value function (hp={hp})")
    plt.tight_layout()
    plt.show()


def plot_policy(policy: dict, height_map: np.ndarray, hp: int) -> None:
    """Plot policy as arrows on the grid for a fixed HP level.

    Args:
        policy: {(i, j, hp): action} mapping
        height_map: 8x8 array for reference
        hp: HP level to visualize
    """
    # (U, V) in (col, row): UP=row-1 -> (0,-1), DOWN=(0,1), LEFT=(-1,0), RIGHT=(1,0)
    action_uv = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
    }

    rows, cols = height_map.shape
    X = np.arange(cols) + 0.5
    Y = np.arange(rows) + 0.5
    U = np.zeros((rows, cols))
    V = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            state = (i, j, hp)
            if state in policy:
                u, v = action_uv[policy[state]]
                U[i, j] = u
                V[i, j] = v

    fig, ax = plt.subplots()
    ax.imshow(height_map, cmap="terrain", aspect="equal", origin="upper", alpha=0.7)
    ax.quiver(
        np.meshgrid(X, Y)[0],
        np.meshgrid(X, Y)[1],
        U,
        V,
        color="darkblue",
        scale=1.0,
        scale_units="xy",
    )
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(f"Policy (hp={hp})")
    plt.tight_layout()
    plt.show()


def plot_trajectory(trajectory: list, height_map: np.ndarray) -> None:
    """Plot agent's path on top of the height map.

    Args:
        trajectory: list of (state, action, reward); state is the state after the step
        height_map: 8x8 array
    """
    rows, cols = height_map.shape
    # Path: start (0,0), then each state from trajectory (state after each step)
    path_i = [0]
    path_j = [0]
    for (state, _action, _reward) in trajectory:
        path_i.append(state[0])
        path_j.append(state[1])

    fig, ax = plt.subplots()
    ax.imshow(height_map, cmap="terrain", aspect="equal", origin="upper")
    if path_i and path_j:
        ax.plot(path_j, path_i, "r-", linewidth=2, label="path")
        ax.scatter(path_j, path_i, c="red", s=80, zorder=5)
        ax.scatter([path_j[0]], [path_i[0]], c="lime", s=150, marker="s", label="start", zorder=5)
        ax.scatter([path_j[-1]], [path_i[-1]], c="gold", s=150, marker="*", label="end", zorder=5)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Trajectory")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_convergence(info: dict) -> None:
    """Plot convergence curve (max delta per iteration).

    Args:
        info: dict from algorithm.solve() containing 'delta_history'
    """
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
    plt.show()
