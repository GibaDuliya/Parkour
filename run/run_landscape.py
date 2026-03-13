"""
Generate landscape, find optimal (min-damage) path via Dijkstra, save images.
Run from project root: python run/run_landscape.py
"""
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landscape import run_landscape


def load_config():
    with open(PROJECT_ROOT / "configs" / "landscape.yaml", "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    run_landscape(config)
