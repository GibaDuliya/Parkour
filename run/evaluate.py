import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import ParkourEnv
from src.utils.metrics import rollout_policy
from src.visualization.visualize import (
    plot_value_function,
    plot_policy,
    plot_trajectory,
)


def load_yaml(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(agent_dir: str):
    """Evaluate a saved agent (value function + policy).

    Args:
        agent_dir: path to the agent folder (contains value_function.npy,
                   policy.npy, states.npy, meta.yaml)
    """
    agent_path = Path(agent_dir)
    if not agent_path.is_absolute():
        agent_path = PROJECT_ROOT / agent_path

    # 1. Load agent artifacts
    V = np.load(agent_path / "value_function.npy")
    policy = np.load(agent_path / "policy.npy")
    states_arr = np.load(agent_path / "states.npy")
    states = [tuple(s) for s in states_arr]
    meta = load_yaml(agent_path / "meta.yaml")

    state_to_id = {s: i for i, s in enumerate(states)}

    # 2. Reconstruct environment
    env_config = load_yaml(PROJECT_ROOT / "configs/env.yaml")
    # Override landscape_id from agent metadata
    env_config["landscape_id"] = meta["landscape_id"]
    env = ParkourEnv(env_config)

    landscape_dir = PROJECT_ROOT / "landscape" / f"landscape_{meta['landscape_id']}"
    min_hp_map = np.load(landscape_dir / "min_hp.npy")

    # actions list reconstructed from env (same order as during training)
    actions = env.get_actions()

    # 3. Rollout
    rollout = rollout_policy(env, policy, states, actions, state_to_id)

    print(f"Agent: {agent_path.name}")
    print(f"Algorithm: {meta['algorithm']}")
    print(
        f"Rollout steps: {rollout['steps']}, "
        f"total reward: {rollout['total_reward']}, "
        f"final HP: {rollout['final_hp']}, "
        f"victory: {rollout['victory']}"
    )

    # 4. Visualization — save next to agent artifacts
    out_dir = agent_path / "eval_plots"
    out_dir.mkdir(exist_ok=True)
    print(f"Eval plots saved to: {out_dir}")

    plot_value_function(
        V, env.height_map, min_hp_map, state_to_id,
        save_path=out_dir / "value_function.png",
    )
    plot_policy(
        policy, env.height_map, env.hp_start, state_to_id, actions,
        save_path=out_dir / "policy.png",
    )
    plot_trajectory(rollout["trajectory"], env.height_map, save_path=out_dir / "trajectory.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parkour RL — Evaluate saved agent")
    parser.add_argument(
        "agent_dir",
        help="Path to agent folder (e.g. agents/2024-01-01_12-00-00)",
    )
    args = parser.parse_args()
    main(args.agent_dir)
