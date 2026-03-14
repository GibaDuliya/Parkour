import numpy as np

from src.environment.parkour_env import ParkourEnv


def rollout_policy(
    env: ParkourEnv,
    policy: np.ndarray,
    states: list,
    actions: list,
    state_to_id: dict,
    max_steps: int = 10000,
) -> dict:
    """Execute a policy from start state to termination.

    Args:
        env: ParkourEnv instance
        policy: (n_states,) array of action indices
        states: list of state tuples, same order as policy indices
        actions: list of actions, same order as action indices
        state_to_id: mapping state tuple -> state index

    Returns:
        dict with keys: trajectory, total_reward, steps, final_hp, victory
    """
    state = (0, 0, getattr(env, "hp_init", env.hp_start))
    trajectory = []
    total_reward = 0.0
    steps = 0
    done = False
    dead = False

    while not done and steps < max_steps:
        s_id = state_to_id.get(state)
        if s_id is None:
            break
        a_id = int(policy[s_id])
        action = actions[a_id]
        next_state, reward, done, dead = env.step(state, action)
        trajectory.append((next_state, action, reward))
        total_reward += reward
        steps += 1
        state = next_state

    final_hp = state[2]
    victory = done and not dead
    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "steps": steps,
        "final_hp": final_hp,
        "victory": victory,
    }


def convergence_stats(info: dict) -> dict:
    """Extract summary statistics from algorithm info.

    Args:
        info: dict returned by algorithm.solve()

    Returns:
        dict with keys: 'iterations', 'final_delta', 'time'
    """
    delta_history = info.get("delta_history", [])
    iterations = info.get("iterations")
    if iterations is None:
        iterations = len(delta_history)
    final_delta = delta_history[-1] if delta_history else None
    total_time = info.get("time")

    return {
        "iterations": iterations,
        "final_delta": final_delta,
        "time": total_time,
    }
