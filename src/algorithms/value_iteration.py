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
        self._gamma: float = config["gamma"]
        self._theta: float = config["theta"]
        self._max_iters: int = config["max_iters"]
        self._stop_criterion: str = config.get("stop_criterion", "delta")

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
        policy_prev = None

        for i in tqdm(range(self._max_iters)):
            V = self.V
            Q = self.R + self._gamma * V[self.next_state_ids]  # (n_states, n_actions)
            self.V = np.max(Q, axis=1)
            delta = np.max(np.abs(V - self.V))
            delta_history.append(delta)

            if self._stop_criterion == "delta":
                if delta < self._theta:
                    break
            elif self._stop_criterion == "policy":
                policy_curr = np.argmax(Q, axis=1)
                if policy_prev is not None and np.array_equal(policy_curr, policy_prev):
                    break
                policy_prev = policy_curr

        elapsed = time.perf_counter() - t_0
        return {"iterations": i, "delta_history": delta_history, "time": elapsed}


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

