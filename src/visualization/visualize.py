import numpy as np
import matplotlib.pyplot as plt


def plot_height_map(height_map: np.ndarray) -> None:
    """Plot 2D heatmap of building heights.

    Args:
        height_map: 8x8 array of heights
    """
    # TODO: imshow with colorbar, annotate cells with height values
    raise NotImplementedError


def plot_value_function(V: dict, height_map: np.ndarray, hp: int) -> None:
    """Plot value function heatmap for a fixed HP level.

    Args:
        V: {(i, j, hp): float} value function
        height_map: 8x8 array for reference
        hp: HP level to visualize
    """
    # TODO: extract V values for given hp, show as heatmap
    raise NotImplementedError


def plot_policy(policy: dict, height_map: np.ndarray, hp: int) -> None:
    """Plot policy as arrows on the grid for a fixed HP level.

    Args:
        policy: {(i, j, hp): action} mapping
        height_map: 8x8 array for reference
        hp: HP level to visualize
    """
    # TODO: draw arrows for each cell based on policy action
    raise NotImplementedError


def plot_trajectory(trajectory: list, height_map: np.ndarray) -> None:
    """Plot agent's path on top of the height map.

    Args:
        trajectory: list of (state, action, reward)
        height_map: 8x8 array
    """
    # TODO: overlay path on height map heatmap
    raise NotImplementedError


def plot_convergence(info: dict) -> None:
    """Plot convergence curve (max delta per iteration).

    Args:
        info: dict from algorithm.solve() containing 'delta_history'
    """
    # TODO: line plot of delta_history
    raise NotImplementedError
