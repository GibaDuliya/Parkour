import numpy as np

from src.environment.parkour_env import ParkourEnv


def rollout_policy(
    env: ParkourEnv,
    policy: np.ndarray,
    states: list,
    actions: list,
    state_to_id: dict,
    max_steps: int = 10000,
    start_state: tuple | None = None,
    gamma: float = 1.0,
) -> dict:
    """Execute a policy from start state to termination.

    Args:
        env: ParkourEnv instance
        policy: (n_states,) array of action indices
        states: list of state tuples, same order as policy indices
        actions: list of actions, same order as action indices
        state_to_id: mapping state tuple -> state index
        start_state: optional starting (i, j, hp); defaults to (0, 0, hp_init)
        gamma: discount factor (1.0 = undiscounted)

    Returns:
        dict with keys: trajectory, total_reward, steps, final_hp, victory
    """
    if start_state is not None:
        state = start_state
    else:
        state = (0, 0, getattr(env, "hp_init", env.hp_start))
    trajectory = []
    total_reward = 0.0
    discount = 1.0
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
        total_reward += discount * reward
        discount *= gamma
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


def sample_eval_cells(min_hp_map: np.ndarray, n: int = 10, rng: np.random.Generator | None = None) -> list:
    """Sample n valid start cells from min_hp_map once (for reuse across iterations).

    Only cells where 0 < min_hp_map[i, j] <= hp_start are included,
    i.e. cells from which the agent with the maximum available HP can reach the goal.

    Args:
        min_hp_map: (rows, cols) array; cells with value > 0 are reachable
        n: number of cells to sample
        rng: optional random generator

    Returns:
        list of (i, j) tuples
    """
    if rng is None:
        rng = np.random.default_rng()
    reachable = min_hp_map[min_hp_map > 0]
    hp_start = int(reachable.max()) if len(reachable) > 0 else 1
    rows, cols = min_hp_map.shape
    valid = [(i, j) for i in range(rows) for j in range(cols) if 0 < min_hp_map[i, j] <= hp_start]
    indices = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
    return [valid[k] for k in indices]


def eval_fixed_rollouts(
    env: ParkourEnv,
    policy: np.ndarray,
    states: list,
    actions: list,
    state_to_id: dict,
    min_hp_map: np.ndarray,
    cells: list,
    gamma: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    """Run rollouts from a fixed set of cells and return discounted return per cell.

    Args:
        cells: list of (i, j) tuples pre-sampled by sample_eval_cells
        min_hp_map: used to look up starting HP for each cell
        gamma: discount factor

    Returns:
        (n_cells,) array of discounted returns, one per cell
    """
    returns = []
    for (i, j) in cells:
        hp = int(min_hp_map[i, j])
        result = rollout_policy(env, policy, states, actions, state_to_id,
                                max_steps=max_steps, start_state=(i, j, hp), gamma=gamma)
        returns.append(result["total_reward"])
    return np.array(returns, dtype=float)


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
