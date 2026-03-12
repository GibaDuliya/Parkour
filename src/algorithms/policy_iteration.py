from src.algorithms.base import BaseAlgorithm
from src.environment.parkour_env import ParkourEnv


class PolicyIteration(BaseAlgorithm):
    """Policy Iteration algorithm."""

    def __init__(self, env: ParkourEnv, config: dict):
        """
        Args:
            env: ParkourEnv instance
            config: parsed policy_iteration.yaml (gamma, theta, max_eval_iters)
        """
        super().__init__(env, config)
        self._V: dict = {}           # {state: float}
        self._policy: dict = {}      # {state: action}
        self._gamma: float = config["gamma"]
        self._theta: float = config["theta"]
        self._max_eval_iters: int = config.get("max_eval_iters", 1000)

    def solve(self) -> dict:
        """Run Policy Iteration until policy is stable.

        Loop:
            1. Policy Evaluation  — compute V for current policy
            2. Policy Improvement — greedily update policy from V

        Returns:
            info: dict with keys 'iterations', 'delta_history', 'time'
        """
        # TODO: initialize random policy, then loop eval + improve until stable
        raise NotImplementedError

    def _policy_evaluation(self) -> None:
        """Iterative policy evaluation.

        For fixed self._policy, update self._V via:
            V(s) = r(s, pi(s)) + gamma * V(s')
        until convergence (max delta < theta) or max_eval_iters reached.
        """
        # TODO: implement iterative evaluation
        raise NotImplementedError

    def _policy_improvement(self) -> bool:
        """Greedy policy improvement.

        For each state: pi(s) = argmax_a [ r(s,a) + gamma * V(s') ]

        Returns:
            stable: True if policy did not change
        """
        # TODO: implement greedy improvement, return whether policy is stable
        raise NotImplementedError

    def get_policy(self) -> dict:
        """Return current policy.

        Returns:
            dict: {state: action}
        """
        return self._policy

    def get_value_function(self) -> dict:
        """Return current value function.

        Returns:
            dict: {state: float}
        """
        return self._V
