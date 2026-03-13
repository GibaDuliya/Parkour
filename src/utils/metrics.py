from src.environment.parkour_env import ParkourEnv


def rollout_policy(env: ParkourEnv, policy: dict, max_steps: int = 10000) -> dict:
    """Execute a policy from start state to termination.

    Args:
        env: ParkourEnv instance
        policy: {state: action} mapping

    Returns:
        dict with keys:
            - trajectory: list of (state, action, reward)
            - total_reward: float
            - steps: int
            - final_hp: int
            - victory: bool
    """
    state = (0, 0, env.hp_start)
    trajectory = []
    total_reward = 0.0
    steps = 0

    done = False
    dead = False
    while not done and steps < max_steps:
        action = policy.get(state)
        if action is None:
            # No action defined — terminate rollout
            break
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
