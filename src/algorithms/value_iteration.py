from src.algorithms.base import BaseAlgorithm
from src.environment.parkour_env import ParkourEnv
from tqdm import tqdm
import numpy as np
import time


class ValueIteration(BaseAlgorithm):
    """Value Iteration algorithm."""

    def __init__(self, env: ParkourEnv, config: dict):
        """
        Args:
            env: ParkourEnv instance
            config: parsed value_iteration.yaml (gamma, theta)
        """
        super().__init__(env, config)
        self._V: dict = {}       # {state: float}
        self._gamma: float = config["gamma"]
        self._theta: float = config["theta"]
        self._max_iters: int = config["max_iters"]

    def solve(self) -> dict:
        """Run Value Iteration until convergence.

        Iteratively applies the Bellman optimality update:
            V(s) = max_a [ r(s,a) + gamma * V(s') ]
        until max|V_new - V_old| < theta.

        Returns:
            info: dict with keys 'iterations', 'delta_history', 'time'
        """

        n_states = len(self.states)
        n_actions = len(self.actions)

        delta_history = []
        t_0 = time.perf_counter()

        for i in tqdm(range(self._max_iters)):
            V = self.V
            V_next = V[self.next_state_ids]  # (n_states, n_actions)
            self.V = np.max(self.R + self._gamma * V_next, axis=1)
            delta = np.max(np.abs(V - self.V))
            delta_history.append(delta)
            iter = i

            if delta < self._theta:
                break

        elapsed = time.perf_counter()-t_0

        return {'iterations': iter, 'delta_history': delta_history, 'time': elapsed}            


    def get_policy(self) -> dict:
        """Compute greedy policy from current V.

        For each state: pi(s) = argmax_a [ r(s,a) + gamma * V(s') ]

        Returns:
            dict: {state: action}
        """
        V_next = self.V[self.next_state_ids]  # (n_states, n_actions)
        return np.argmax(self.R + self._gamma * V_next, axis=1)


    def get_value_function(self) -> dict:
        """Return current value function.

        Returns:
            dict: {state: float}
        """
        return self.V

