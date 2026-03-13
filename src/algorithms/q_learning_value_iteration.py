import time

import numpy as np
from tqdm.auto import tqdm

from src.algorithms.base import BaseAlgorithm
from src.environment.parkour_env import ParkourEnv


class QLearningValueIteration(BaseAlgorithm):
    """Q-table Value Iteration (dynamic programming in Q-space).

    Deterministic setting:
        Q_{k+1}(s, a) = R(s, a) + gamma * max_{a'} Q_k(s', a')
        where s' is the unique next state for (s, a).
    """

    def __init__(self, env: ParkourEnv, config: dict):
        super().__init__(env, config)
        self._gamma: float = config["gamma"]
        self._theta: float = config["theta"]
        self._max_iters: int = config.get("max_iters", 1000)

        # next_state_ids from base (no T_matr)
        # Q-table: shape (n_states, n_actions)
        self.Q = np.zeros((len(self.states), len(self.actions)), dtype=float)

    def solve(self) -> dict:
        """Run Q-value iteration until convergence in sup-norm.

        Returns:
            info: dict with keys 'iterations', 'delta_history', 'time'
        """
        delta_history = []
        t0 = time.perf_counter()

        it = 0
        for _ in tqdm(range(self._max_iters), desc="Q-value iteration"):
            # Q_new[s, a] = R[s, a] + gamma * max_{a'} Q[s', a']
            # where s' = next_state_ids[s, a]
            Q_next = self.Q[self.next_state_ids]  # shape (n_states, n_actions, n_actions)
            max_Q_next = Q_next.max(axis=2)       # max over a' for each (s, a)
            Q_new = self.R + self._gamma * max_Q_next

            # Convergence check
            delta = float(np.max(np.abs(Q_new - self.Q)))
            delta_history.append(delta)
            self.Q = Q_new

            it += 1
            if delta < self._theta:
                break

        elapsed = time.perf_counter() - t0
        return {
            "iterations": it,
            "delta_history": delta_history,
            "time": elapsed,
        }

    def get_policy(self) -> np.ndarray:
        """Greedy policy as (n_states,) array of action indices."""
        return self.Q.argmax(axis=1).astype(np.int32)

    def get_value_function(self) -> np.ndarray:
        """State value function V(s) = max_a Q(s, a) as (n_states,) array."""
        return self.Q.max(axis=1)

