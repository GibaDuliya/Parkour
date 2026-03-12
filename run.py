import yaml

from src.environment import ParkourEnv
from src.algorithms import ValueIteration, PolicyIteration
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
}


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
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

    # 3. Create and run algorithm
    algo = algo_cls(env, algo_config)
    info = algo.solve()

    # 4. Extract results
    policy = algo.get_policy()
    #V = algo.get_value_function()

    # 5. Rollout and metrics
    # rollout = rollout_policy(env, policy)
    # stats = convergence_stats(info)

    # print(f"Algorithm: {algorithm_name}")
    # print(f"Iterations: {stats['iterations']}")
    # print(f"Rollout steps: {rollout['steps']}, total reward: {rollout['total_reward']}")
    # print(f"Victory: {rollout['victory']}, final HP: {rollout['final_hp']}")

    # 6. Visualization
    # plot_height_map(env.height_map)
    # plot_value_function(V, env.height_map, hp=env.hp_start)
    # plot_policy(policy, env.height_map, hp=env.hp_start)
    # plot_trajectory(rollout["trajectory"], env.height_map)
    # plot_convergence(info)


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
