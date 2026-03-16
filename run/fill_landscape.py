"""
Generate height_map.npy and min_hp.npy for an existing landscape folder
that has config.yaml but is missing the .npy files.

Usage (from project root):
  python run/fill_landscape.py 1
"""
import sys
import yaml
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landscape.landscape import (
    generate_height_map,
    build_graph,
    compute_min_hp_map,
)


def main(landscape_id: int) -> None:
    landscape_dir = PROJECT_ROOT / "landscape" / f"landscape_{landscape_id}"
    if not landscape_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {landscape_dir}")

    config_path = landscape_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    grid_size = config["grid_size"]
    min_h = config["min_building_height"]
    max_h = config["max_building_height"]
    seed = config["seed"]
    max_jump_up = config["max_jump_up"]
    safe_jump_down = config["safe_jump_down"]

    height_map = generate_height_map(grid_size, min_h, max_h, seed)
    goal = (grid_size - 1, grid_size - 1)
    graph = build_graph(height_map, max_jump_up, safe_jump_down)
    min_hp_map = compute_min_hp_map(graph, height_map, goal)

    np.save(landscape_dir / "height_map.npy", height_map)
    np.save(landscape_dir / "min_hp.npy", min_hp_map)

    print(f"Saved height_map.npy and min_hp.npy to {landscape_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fill landscape folder with height_map.npy and min_hp.npy")
    parser.add_argument("landscape_id", type=int, help="e.g. 1 for landscape_1")
    args = parser.parse_args()
    main(args.landscape_id)
