import time

import numpy as np

from src.algorithms.base import BaseAlgorithm
from src.environment.parkour_env import ParkourEnv


class ValueIteration(BaseAlgorithm):
    """Value Iteration: V(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) V(s') ]."""

    def __init__(self, env: ParkourEnv, config: dict):
        super().__init__(env, config)
        self._gamma: float = config["gamma"]
        self._theta: float = config["theta"]
        self.V = np.zeros(self.n_states, dtype=float)

    def solve(self) -> dict:
        delta_history = []
        t0 = time.perf_counter()
        while True:
            # V_new[s] = max_a [ R[s,a] + gamma * V[next_state_ids[s,a]] ]
            next_V = self.V[self.next_state_ids]  # (n_states, n_actions)
            Q_sa = self.R + self._gamma * next_V
            V_new = np.max(Q_sa, axis=1)
            delta = float(np.max(np.abs(V_new - self.V)))
            delta_history.append(delta)
            self.V = V_new
            if delta < self._theta:
                break
        return {
            "iterations": len(delta_history),
            "delta_history": delta_history,
            "time": time.perf_counter() - t0,
        }

    def get_policy(self) -> np.ndarray:
        next_V = self.V[self.next_state_ids]
        Q_sa = self.R + self._gamma * next_V
        return np.argmax(Q_sa, axis=1).astype(np.int32)

    def get_value_function(self) -> np.ndarray:
        return self.V.copy()
