import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
from src.algorithms import ValueIteration, PolicyIteration, QLearningValueIteration
from src.utils.metrics import rollout_policy, convergence_stats
from src.visualization.visualize import (
    plot_height_map,
    plot_value_function,
    plot_policy,
    plot_trajectory,
    plot_convergence,
)

ALGORITHMS = {
    "value_iteration": (ValueIteration, "configs/value_iteration.yaml"),
    "policy_iteration": (PolicyIteration, "configs/policy_iteration.yaml"),
    "q_learning_value_iteration": (QLearningValueIteration, "configs/q_learning_value_iteration.yaml"),
}


def load_yaml(path: str) -> dict:
    with open(PROJECT_ROOT / path, "r") as f:
        return yaml.safe_load(f)


def main(algorithm_name: str):
    """Run a single experiment.

    Args:
        algorithm_name: key from ALGORITHMS dict
    """
    # 1. Load configs
    env_config = load_yaml("configs/env.yaml")
    algo_cls, algo_config_path = ALGORITHMS[algorithm_name]
    algo_config = load_yaml(algo_config_path)

    # 2. Create environment
    env = ParkourEnv(env_config)

    # Load min_hp map for this landscape
    landscape_dir = PROJECT_ROOT / "landscape" / f"landscape_{env_config['landscape_id']}"
    min_hp_map = np.load(landscape_dir / "min_hp.npy")

    # 3. Create and run algorithm
    algo = algo_cls(env, algo_config)
    info = algo.solve()

    # 4. Extract results
    policy = algo.get_policy()
    V = algo.get_value_function()

    # 5. Rollout and metrics
    rollout = rollout_policy(
        env, policy, algo.states, algo.actions, algo._state_to_id
    )
    stats = convergence_stats(info)

    print(f"Algorithm: {algorithm_name}")
    print(f"Iterations: {stats['iterations']}")
    print(f"Final delta: {stats['final_delta']}")
    print(
        f"Rollout steps: {rollout['steps']}, "
        f"total reward: {rollout['total_reward']}, "
        f"final HP: {rollout['final_hp']}, "
        f"victory: {rollout['victory']}"
    )

    # 6. Visualization — save to plots/{date_time}/
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = PROJECT_ROOT / "plots" / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {out_dir}")

    plot_height_map(env.height_map, save_path=out_dir / "height_map.png")
    plot_value_function(
        V, env.height_map, min_hp_map, algo._state_to_id,
        save_path=out_dir / "value_function.png",
    )
    plot_policy(
        policy, env.height_map, env.hp_start, algo._state_to_id, algo.actions,
        save_path=out_dir / "policy.png",
    )
    plot_trajectory(rollout["trajectory"], env.height_map, save_path=out_dir / "trajectory.png")
    plot_convergence(info, save_path=out_dir / "convergence.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parkour RL — Dynamic Programming")
    parser.add_argument(
        "algorithm",
        choices=list(ALGORITHMS.keys()),
        help="Algorithm to run",
    )
    args = parser.parse_args()
    main(args.algorithm)
