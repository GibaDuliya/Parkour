import heapq
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


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
# 4. Run: solve + save images
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

    # Step 2: build graph
    graph = build_graph(height_map, max_jump_up, safe_jump_down)

    # Step 3 & 4: find optimal path
    start = (0, 0)
    end = (grid_size - 1, grid_size - 1)
    total_damage, path = dijkstra(graph, start, end)

    # Save config
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Save height map data
    np.save(os.path.join(out_dir, "height_map.npy"), height_map)

    # Save height map image
    _save_height_map(height_map, out_dir)

    # Save path image
    _save_path(height_map, path, total_damage, out_dir)

    if path is None:
        print(f"No path found from {start} to {end}.")
    else:
        print(f"Optimal path: total damage = {total_damage}, steps = {len(path) - 1}")
        print(f"Path: {path}")
    print(f"Saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_height_map(height_map: np.ndarray, out_dir: str) -> None:
    rows, cols = height_map.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Height")

    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, int(height_map[i, j]), ha="center", va="center",
                    color="black", fontsize=10)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticklabels(np.arange(rows))
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Height map")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "height_map.png"), dpi=150)
    plt.close(fig)


def _save_path(height_map: np.ndarray, path, total_damage, out_dir: str) -> None:
    rows, cols = height_map.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(height_map, cmap="YlGnBu", aspect="equal", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Height")

    # Height values as text
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, int(height_map[i, j]), ha="center", va="center",
                    color="black", fontsize=8)

    if path is not None:
        # Draw red arrows between consecutive cells
        for k in range(len(path) - 1):
            i0, j0 = path[k]
            i1, j1 = path[k + 1]
            di = i1 - i0
            dj = j1 - j0
            ax.annotate(
                "",
                xy=(j1, i1),
                xytext=(j0, i0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="red",
                    lw=2.0,
                    mutation_scale=14,
                ),
                zorder=7,
            )
        path_i = [p[0] for p in path]
        path_j = [p[1] for p in path]
        ax.scatter([path_j[0]], [path_i[0]], c="lime", s=200, marker="s",
                   label="start", zorder=8)
        ax.scatter([path_j[-1]], [path_i[-1]], c="gold", s=200, marker="*",
                   label="goal", zorder=8)
        title = f"Optimal path (total damage = {total_damage})"
    else:
        title = "No path found"

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticklabels(np.arange(rows))
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
    plt.savefig(os.path.join(out_dir, "optimal_path.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
