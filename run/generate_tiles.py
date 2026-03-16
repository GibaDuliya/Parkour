import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def _save_tile(name: str, color: tuple[float, float, float]):
    ASSETS_DIR.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=64)
    fig.patch.set_alpha(0.0)
    ax.set_axis_off()

    # simple beveled square
    grad = np.linspace(0.9, 0.6, 64)
    base = np.outer(np.ones_like(grad), grad)
    r, g, b = color
    img = np.stack([base * r, base * g, base * b], axis=-1)
    ax.imshow(img, origin="lower")

    fig.savefig(ASSETS_DIR / name, dpi=64, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    _save_tile("tile_low.png", (0.2, 0.4, 0.8))
    _save_tile("tile_mid.png", (0.2, 0.6, 0.9))
    _save_tile("tile_high.png", (0.9, 0.9, 0.3))


if __name__ == "__main__":
    main()

