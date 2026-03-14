import heapq
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Height map generation
# ---------------------------------------------------------------------------

def generate_height_map(grid_size: int, min_building_height: int, max_building_height: int, seed: int) -> np.ndarray:
    """Generate a grid_size x grid_size height map with uniform random integers in [min_building_height, max_building_height].

    The same seed always produces the same map.

    Args:
        grid_size: number of rows and columns
        min_building_height: minimum possible building height (inclusive)
        max_building_height: maximum possible building height (inclusive)
        seed: RNG seed for reproducibility

    Returns:
        np.ndarray of shape (grid_size, grid_size) with dtype int
    """
    rng = np.random.default_rng(seed)
    return rng.integers(min_building_height, max_building_height + 1, size=(grid_size, grid_size))


# ---------------------------------------------------------------------------
# 2. Graph construction
# ---------------------------------------------------------------------------

def build_graph(height_map: np.ndarray, max_jump_up: int, safe_jump_down: int) -> dict:
    """Build a directed adjacency-list graph from the height map.

    An edge (u -> v) exists between horizontally/vertically adjacent cells when:
        height[v] - height[u] <= max_jump_up  (can jump up at most max_jump_up)

    Edge weight = fall damage when moving from u to v:
        drop = height[u] - height[v]   (positive when going down)
        damage = max(0, drop - safe_jump_down)

    Args:
        height_map: 2D array of building heights
        max_jump_up: maximum upward height difference allowed
        safe_jump_down: fall distance that causes no damage

    Returns:
        dict mapping (i, j) -> list of ((i2, j2), damage)
    """
    rows, cols = height_map.shape
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    graph: dict = {}

    for i in range(rows):
        for j in range(cols):
            graph[(i, j)] = []
            h = int(height_map[i, j])
            for di, dj in deltas:
                i2, j2 = i + di, j + dj
                if not (0 <= i2 < rows and 0 <= j2 < cols):
                    continue
                h2 = int(height_map[i2, j2])
                dh = h2 - h  # positive = going up
                if dh > max_jump_up:
                    continue
                drop = h - h2  # positive = going down
                damage = max(0, drop - safe_jump_down)
                graph[(i, j)].append(((i2, j2), damage))

    return graph


# ---------------------------------------------------------------------------
# 3. Dijkstra
# ---------------------------------------------------------------------------

def dijkstra(graph: dict, start: tuple, end: tuple) -> tuple:
    """Find the path with minimum total damage from start to end using Dijkstra's algorithm.

    Args:
        graph: adjacency list from build_graph — dict[(i,j)] -> [((i2,j2), weight), ...]
        start: (i, j) start cell
        end: (i, j) goal cell

    Returns:
        (total_damage, path) where path is list[(i, j)] from start to end,
        or (float('inf'), None) if end is unreachable.
    """
    dist = {start: 0}
    prev: dict = {start: None}
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u == end:
            break
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if end not in dist:
        return float("inf"), None

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return dist[end], path


# ---------------------------------------------------------------------------
# 4. Min-HP map
# ---------------------------------------------------------------------------

def compute_min_hp_map(graph: dict, height_map: np.ndarray, goal: tuple) -> np.ndarray:
    """Compute minimum starting HP required at each cell to reach the goal.

    Builds a reversed graph and runs Dijkstra from the goal.
    min_hp[i, j] = min_damage_on_path_(i,j)->goal + 1.
    Unreachable cells get value -1.

    Args:
        graph: adjacency list from build_graph — dict[(i,j)] -> [((i2,j2), damage)]
        height_map: 2D array of building heights
        goal: (i, j) goal cell

    Returns:
        np.ndarray of shape (rows, cols) with dtype int
    """
    rows, cols = height_map.shape
    n_cells = rows * cols

    # Build reversed graph: for each edge u->v with weight w, add v->u with weight w
    rev_graph: dict = {(i, j): [] for i in range(rows) for j in range(cols)}
    for u, neighbours in graph.items():
        for v, w in neighbours:
            rev_graph[v].append((u, w))

    dist = {goal: 0}
    heap = [(0, goal)]

    with tqdm(total=n_cells, desc="min_hp map") as pbar:
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            pbar.update(1)
            for v, w in rev_graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

    min_hp_map = np.full((rows, cols), -1, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if (i, j) in dist:
                min_hp_map[i, j] = int(dist[(i, j)]) + 1

    return min_hp_map


# ---------------------------------------------------------------------------
# 5. Unit graph + min-time Dijkstra
# ---------------------------------------------------------------------------

def build_unit_graph(height_map: np.ndarray, max_jump_up: int) -> dict:
    """Build a directed adjacency-list graph where each valid edge has weight 1.

    An edge (u -> v) exists when adjacent cells satisfy:
        height[v] - height[u] <= max_jump_up

    No damage is stored — this graph is used for finding shortest paths by steps.

    Args:
        height_map: 2D array of building heights
        max_jump_up: maximum upward height difference allowed

    Returns:
        dict mapping (i, j) -> list of (i2, j2) reachable neighbours
    """
    rows, cols = height_map.shape
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    graph: dict = {}

    for i in range(rows):
        for j in range(cols):
            graph[(i, j)] = []
            h = int(height_map[i, j])
            for di, dj in deltas:
                i2, j2 = i + di, j + dj
                if not (0 <= i2 < rows and 0 <= j2 < cols):
                    continue
                if int(height_map[i2, j2]) - h > max_jump_up:
                    continue
                graph[(i, j)].append((i2, j2))

    return graph


def dijkstra_min_steps(
    unit_graph: dict,
    height_map: np.ndarray,
    safe_jump_down: int,
    start: tuple,
    end: tuple,
    hp_start: int,
) -> tuple:
    """Find the shortest path (min steps) with a fixed HP budget.

    State is (i, j, hp_remaining). A neighbour is only enqueued when the agent
    survives the jump (hp_remaining after damage > 0). Priority is steps only,
    so the first time the goal cell is reached it is via the fastest route.

    Args:
        unit_graph: adjacency list from build_unit_graph
        height_map: 2D array of building heights (used to compute damage)
        safe_jump_down: fall distance that causes no damage
        start: (i, j) start cell
        end: (i, j) goal cell
        hp_start: starting HP budget

    Returns:
        (steps, path)
        - steps: number of moves on the optimal path
        - path: list[(i, j)] from start to end
        or (float('inf'), None) if end is unreachable with this HP.
    """
    start_state = (start[0], start[1], hp_start)
    dist: dict = {start_state: 0}
    prev: dict = {start_state: None}
    heap = [(0, start_state)]  # (steps, (i, j, hp))

    goal_state = None

    while heap:
        steps, u = heapq.heappop(heap)
        if steps > dist.get(u, float("inf")):
            continue
        i, j, hp = u
        if (i, j) == end:
            goal_state = u
            break
        for (i2, j2) in unit_graph.get((i, j), []):
            drop = int(height_map[i, j]) - int(height_map[i2, j2])
            dmg = max(0, drop - safe_jump_down)
            hp_new = hp - dmg
            if hp_new <= 0:
                continue  # agent dies on this jump — skip
            next_state = (i2, j2, hp_new)
            new_steps = steps + 1
            if new_steps < dist.get(next_state, float("inf")):
                dist[next_state] = new_steps
                prev[next_state] = u
                heapq.heappush(heap, (new_steps, next_state))

    if goal_state is None:
        return float("inf"), None

    path = []
    node = goal_state
    while node is not None:
        path.append((node[0], node[1]))
        node = prev[node]
    path.reverse()
    return dist[goal_state], path


# ---------------------------------------------------------------------------
# 5. Run: solve + save images
# ---------------------------------------------------------------------------

def run_landscape(config: dict) -> None:
    """Generate the landscape, run Dijkstra, and save all images.

    Saves to landscape/landscape_seed_{seed}/:
        - height_map.png  : heatmap of building heights
        - optimal_path.png: height map with the minimum-damage path overlaid

    Args:
        config: dict loaded from configs/landscape.yaml
    """
    seed: int = config["seed"]
    grid_size: int = config["grid_size"]
    min_building_height: int = config["min_building_height"]
    max_building_height: int = config["max_building_height"]
    max_jump_up: int = config["max_jump_up"]
    safe_jump_down: int = config["safe_jump_down"]

    # Output directory: landscape_{next_number}
    base_dir = "landscape"
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        int(name.split("_")[1])
        for name in os.listdir(base_dir)
        if name.startswith("landscape_") and name.split("_")[1].isdigit()
    ]
    next_num = max(existing, default=0) + 1
    out_dir = os.path.join(base_dir, f"landscape_{next_num}")
    os.makedirs(out_dir)

    # Step 1: generate map
    height_map = generate_height_map(grid_size, min_building_height, max_building_height, seed)

    start = (0, 0)
    end = (grid_size - 1, grid_size - 1)

    # Step 2a: damage-optimal graph + Dijkstra
    graph = build_graph(height_map, max_jump_up, safe_jump_down)
    total_damage, path_damage = dijkstra(graph, start, end)

    # Step 2b: min-HP map for every cell (min starting HP at that cell to reach goal)
    min_hp_map = compute_min_hp_map(graph, height_map, end)

    # Step 2c: unit graph + min-steps Dijkstra with landscape min HP
    hp_budget = (int(total_damage) + 1) if total_damage != float("inf") else 1
    unit_graph = build_unit_graph(height_map, max_jump_up)
    total_steps, path_time = dijkstra_min_steps(unit_graph, height_map, safe_jump_down, start, end, hp_budget)

    # Save config
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Save arrays
    np.save(os.path.join(out_dir, "height_map.npy"), height_map)
    np.save(os.path.join(out_dir, "min_hp.npy"), min_hp_map)

    # Save images
    _save_height_map(height_map, out_dir)
    _save_path(height_map, path_damage, total_damage, out_dir)  # -> optimal_hp_path.png
    _save_min_time_path(height_map, path_time, total_steps, hp_budget, out_dir)
    _save_min_hp_map(min_hp_map, out_dir)

    if path_damage is None:
        print("No damage-optimal path found.")
    else:
        print(f"Min-damage path: damage={total_damage}, steps={len(path_damage) - 1}")
    if path_time is None:
        print(f"No time-optimal path found with hp={hp_budget}.")
    else:
        print(f"Min-time path:   steps={total_steps}, min HP={hp_budget}")
    print(f"Saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_TICKS = 20
_MAX_CELL_LABELS = 20  # hide per-cell numbers above this grid size


def _set_ticks(ax, rows: int, cols: int) -> None:
    x_step = max(1, cols // _MAX_TICKS)
    y_step = max(1, rows // _MAX_TICKS)
    ax.set_xticks(np.arange(0, cols, x_step))
    ax.set_yticks(np.arange(0, rows, y_step))
    ax.set_xticklabels(np.arange(0, cols, x_step))
    ax.set_yticklabels(np.arange(0, rows, y_step))


def _save_height_map(height_map: np.ndarray, out_dir: str) -> None:
    rows, cols = height_map.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Height")

    if rows <= _MAX_CELL_LABELS and cols <= _MAX_CELL_LABELS:
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, int(height_map[i, j]), ha="center", va="center",
                        color="black", fontsize=10)

    _set_ticks(ax, rows, cols)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Height map")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "height_map.png"), dpi=150)
    plt.close(fig)


def _save_min_hp_map(min_hp_map: np.ndarray, out_dir: str) -> None:
    rows, cols = min_hp_map.shape
    display = min_hp_map.astype(float)
    display[display < 0] = np.nan  # unreachable cells

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(display, cmap="YlGnBu", aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax, label="Min HP to reach")

    # Auto-skip tick labels to avoid overlap
    max_ticks = 20
    x_step = max(1, cols // max_ticks)
    y_step = max(1, rows // max_ticks)
    ax.set_xticks(np.arange(0, cols, x_step))
    ax.set_yticks(np.arange(0, rows, y_step))
    ax.set_xticklabels(np.arange(0, cols, x_step))
    ax.set_yticklabels(np.arange(0, rows, y_step))

    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Min HP per cell")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "min_hp.png"), dpi=150)
    plt.close(fig)


def _save_min_time_path(height_map: np.ndarray, path, total_steps, hp_budget, out_dir: str) -> None:
    rows, cols = height_map.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Height")

    if rows <= _MAX_CELL_LABELS and cols <= _MAX_CELL_LABELS:
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, int(height_map[i, j]), ha="center", va="center",
                        color="black", fontsize=8)

    if path is not None:
        path_i = [p[0] for p in path]
        path_j = [p[1] for p in path]
        ax.plot(path_j, path_i, "-", color="red", linewidth=2, zorder=7)
        ax.scatter([path_j[0]], [path_i[0]], c="lime", s=200, marker="s", label="start", zorder=8)
        ax.scatter([path_j[-1]], [path_i[-1]], c="gold", s=200, marker="*", label="goal", zorder=8)
        title = f"Min-time path (steps={total_steps}, HP={hp_budget})"
    else:
        title = "No path found"

    _set_ticks(ax, rows, cols)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "min_time_path.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_path(height_map: np.ndarray, path, total_damage, out_dir: str) -> None:
    rows, cols = height_map.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Height")

    if rows <= _MAX_CELL_LABELS and cols <= _MAX_CELL_LABELS:
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, int(height_map[i, j]), ha="center", va="center",
                        color="black", fontsize=8)

    if path is not None:
        path_i = [p[0] for p in path]
        path_j = [p[1] for p in path]
        ax.plot(path_j, path_i, "-", color="red", linewidth=2, zorder=7)
        ax.scatter([path_j[0]], [path_i[0]], c="lime", s=200, marker="s",
                   label="start", zorder=8)
        ax.scatter([path_j[-1]], [path_i[-1]], c="gold", s=200, marker="*",
                   label="goal", zorder=8)
        title = f"Optimal HP path (total damage = {total_damage})"
    else:
        title = "No path found"

    _set_ticks(ax, rows, cols)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(title)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "optimal_hp_path.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
