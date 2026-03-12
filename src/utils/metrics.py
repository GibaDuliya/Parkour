from src.environment.parkour_env import ParkourEnv


def rollout_policy(env: ParkourEnv, policy: dict) -> dict:
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
    # TODO: start from (0, 0, hp_start), follow policy via env.step() until done
    raise NotImplementedError


def convergence_stats(info: dict) -> dict:
    """Extract summary statistics from algorithm info.

    Args:
        info: dict returned by algorithm.solve()

    Returns:
        dict with keys: 'iterations', 'final_delta', 'time'
    """
    # TODO: extract and return summary from info
    raise NotImplementedError
