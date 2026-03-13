import time

import numpy as np

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

        # Derive deterministic next-state ids from transition tensor P[s, a, s']
        self.next_state_ids = self.T_matr.argmax(axis=2)  # shape (n_states, n_actions)

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
        while True:
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
            if delta < self._theta or it >= self._max_iters:
                break

        elapsed = time.perf_counter() - t0
        return {
            "iterations": it,
            "delta_history": delta_history,
            "time": elapsed,
        }

    def get_policy(self) -> dict:
        """Greedy policy π(s) = argmax_a Q(s, a) as {state: action}."""
        best_action_ids = self.Q.argmax(axis=1)  # shape (n_states,)
        policy = {}
        for s_id, a_id in enumerate(best_action_ids):
            state = self.states[s_id]
            action = self.actions[a_id]
            policy[state] = action
        return policy

    def get_value_function(self) -> dict:
        """State value function V(s) = max_a Q(s, a) as {state: float}."""
        V = self.Q.max(axis=1)
        value_dict = {}
        for s_id, v in enumerate(V):
            state = self.states[s_id]
            value_dict[state] = float(v)
        return value_dict

