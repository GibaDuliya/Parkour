"""Plot mean ± 1 std training curves (returns and convergence) across multiple agent runs.

Usage:
    python run/plot_training.py value_iteration
    python run/plot_training.py policy_iteration --save results.png
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_runs(algorithm_name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load returns_matrix.npy and delta_history.npy from all agent runs.

    Returns:
        (return_matrices, delta_arrays)
        return_matrices: list of [n_eval_iters, n_eval_cells] arrays
        delta_arrays:    list of [n_iters] arrays
    """
    algo_dir = PROJECT_ROOT / "agents" / algorithm_name
    if not algo_dir.exists():
        raise FileNotFoundError(f"No agents found for algorithm '{algorithm_name}' at {algo_dir}")

    return_matrices, delta_arrays = [], []
    for run_dir in sorted(d for d in algo_dir.iterdir() if d.is_dir()):
        r_path = run_dir / "returns_matrix.npy"
        d_path = run_dir / "delta_history.npy"
        if r_path.exists():
            return_matrices.append(np.load(r_path))
        if d_path.exists():
            delta_arrays.append(np.load(d_path))

    if not return_matrices and not delta_arrays:
        raise FileNotFoundError(f"No training data found under {algo_dir}")
    return return_matrices, delta_arrays


def _normalize(a: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]. Returns zeros if range is zero."""
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _plot_mean_std(ax, arrays: list[np.ndarray], color: str, label_prefix: str,
                   normalize: bool = False) -> None:
    """Plot mean ± 1 std over a list of 1-D arrays (possibly different lengths).

    Truncates to the shortest array so all runs are aligned.
    If normalize=True, each array is min-max normalized to [0, 1] before averaging.
    """
    min_len = min(len(a) for a in arrays)
    processed = [_normalize(a[:min_len]) if normalize else a[:min_len] for a in arrays]
    mat = np.stack(processed, axis=0)  # [n_runs, min_len]
    iters = np.arange(1, min_len + 1)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    ax.plot(iters, mean, color=color, linewidth=2,
            label=f"{label_prefix} (mean, {len(arrays)} runs)")
    ax.fill_between(iters, mean - std, mean + std, alpha=0.3, color=color, label="± 1 std")


def plot_training_curves(
    algorithm_name: str,
    return_matrices: list[np.ndarray],
    delta_arrays: list[np.ndarray],
    save_path: Path | None = None,
) -> None:
    has_returns = bool(return_matrices)
    has_deltas = bool(delta_arrays)
    n_plots = int(has_returns) + int(has_deltas)
    if n_plots == 0:
        print("Nothing to plot.")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if has_returns:
        # Mean over eval cells per run → list of [n_eval_iters] arrays
        run_means = [m.mean(axis=1) for m in return_matrices]
        _plot_mean_std(axes[idx], run_means, color="steelblue",
                       label_prefix="return", normalize=False)
        axes[idx].set_xlabel("Iteration")
        axes[idx].set_ylabel("Mean discounted return")
        axes[idx].set_title(f"Training returns — {algorithm_name}")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    if has_deltas:
        _plot_mean_std(axes[idx], delta_arrays, color="tomato",
                       label_prefix="delta", normalize=False)
        axes[idx].set_xlabel("Iteration")
        axes[idx].set_ylabel("Max |V_new - V_old|")
        axes[idx].set_title(f"Convergence — {algorithm_name}")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves across agent runs")
    parser.add_argument("algorithm", help="Algorithm name (e.g. value_iteration)")
    parser.add_argument("--save", type=Path, default=None, help="Path to save the figure")
    args = parser.parse_args()

    return_matrices, delta_arrays = load_runs(args.algorithm)
    print(f"Loaded {len(return_matrices)} return runs, {len(delta_arrays)} delta runs "
          f"for '{args.algorithm}'")

    save_path = args.save or PROJECT_ROOT / "agents" / args.algorithm / "training_curves.png"
    plot_training_curves(args.algorithm, return_matrices, delta_arrays, save_path=save_path)


if __name__ == "__main__":
    main()
