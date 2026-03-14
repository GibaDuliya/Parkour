import time

import numpy as np
from tqdm import tqdm

from src.algorithms.base import BaseAlgorithm
from src.environment.parkour_env import ParkourEnv


class PolicyIteration(BaseAlgorithm):
    """Policy Iteration: alternate policy evaluation and policy improvement."""

    def __init__(self, env: ParkourEnv, config: dict):
        super().__init__(env, config)
        self._gamma = config["gamma"]
        self._theta = config["theta"]
        self._max_eval_iters = config.get("max_eval_iters", 1000)
        self.V = np.zeros(self.n_states, dtype=float)
        self._policy = np.random.randint(0, self.n_actions, size=self.n_states, dtype=np.int32)

    def solve(self) -> dict:
        delta_history = []
        t0 = time.perf_counter()
        with tqdm() as pbar:
            while True:
                self._policy_evaluation()
                policy_new, delta = self._policy_improvement()
                delta_history.append(delta)
                pbar.update(1)
                pbar.set_postfix(delta=f"{delta:.2e}")
                if np.array_equal(policy_new, self._policy):
                    break
                self._policy = policy_new
        return {
            "iterations": len(delta_history),
            "delta_history": delta_history,
            "time": time.perf_counter() - t0,
        }

    def _policy_evaluation(self) -> None:
        r_pi = self.R[np.arange(self.n_states), self._policy]
        next_s = self.next_state_ids[np.arange(self.n_states), self._policy]
        for _ in range(self._max_eval_iters):
            V_new = r_pi + self._gamma * self.V[next_s]
            if np.max(np.abs(V_new - self.V)) < self._theta:
                break
            self.V = V_new.copy()
        self.V = V_new.copy()

    def _policy_improvement(self) -> tuple:
        next_V = self.V[self.next_state_ids]
        Q_sa = self.R + self._gamma * next_V
        policy_new = np.argmax(Q_sa, axis=1).astype(np.int32)
        delta = float(np.max(np.abs(Q_sa[np.arange(self.n_states), policy_new] - self.V)))
        return policy_new, delta

    def get_policy(self) -> np.ndarray:
        return self._policy.copy()

    def get_value_function(self) -> np.ndarray:
        return self.V.copy()
