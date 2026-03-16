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

    Pads shorter arrays with their last value so all runs are aligned to the longest.
    If normalize=True, each array is min-max normalized to [0, 1] before averaging.
    """
    max_len = max(len(a) for a in arrays)

    def _pad(a: np.ndarray) -> np.ndarray:
        if len(a) == max_len:
            return a
        return np.concatenate([a, np.full(max_len - len(a), a[-1])])

    processed = [_normalize(_pad(a)) if normalize else _pad(a) for a in arrays]
    mat = np.stack(processed, axis=0)  # [n_runs, max_len]
    iters = np.arange(1, max_len + 1)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    ax.plot(iters, mean, color=color, linewidth=2,
            label=f"{label_prefix} (mean, {len(arrays)} runs)")
    ax.fill_between(iters, mean - std, mean + std, alpha=0.3, color=color, label="± 1 std")



def plot_returns_only(
    algorithm_name: str,
    return_matrices: list[np.ndarray],
    normalize: bool,
    save_path: Path,
) -> None:
    if not return_matrices:
        print("No return data to plot.")
        return

    run_means = [m.mean(axis=1) for m in return_matrices]
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_mean_std(ax, run_means, color="steelblue",
                   label_prefix="return", normalize=normalize)
    ax.set_xlabel("Iteration")
    if normalize:
        ax.set_ylabel("Normalized return [0, 1]")
        ax.set_title(f"Training returns (normalized) — {algorithm_name}")
    else:
        ax.set_ylabel("Mean discounted return")
        ax.set_title(f"Training returns — {algorithm_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {save_path}")


def plot_convergence_only(
    algorithm_name: str,
    delta_arrays: list[np.ndarray],
    save_path: Path,
) -> None:
    if not delta_arrays:
        print("No delta data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_mean_std(ax, delta_arrays, color="tomato",
                   label_prefix="delta", normalize=False)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |V_new - V_old|")
    ax.set_title(f"Convergence — {algorithm_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves across agent runs")
    parser.add_argument("algorithm", help="Algorithm name (e.g. value_iteration)")
    args = parser.parse_args()

    return_matrices, delta_arrays = load_runs(args.algorithm)
    print(f"Loaded {len(return_matrices)} return runs, {len(delta_arrays)} delta runs "
          f"for '{args.algorithm}'")

    out_dir = PROJECT_ROOT / "agents" / args.algorithm
    plot_returns_only(args.algorithm, return_matrices, normalize=False,
                      save_path=out_dir / "training_returns_absolute.png")
    plot_returns_only(args.algorithm, return_matrices, normalize=True,
                      save_path=out_dir / "training_returns_normalized.png")
    plot_convergence_only(args.algorithm, delta_arrays,
                          save_path=out_dir / "training_convergence.png")


if __name__ == "__main__":
    main()
