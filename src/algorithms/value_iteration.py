from src.algorithms.base import BaseAlgorithm
from src.environment.parkour_env import ParkourEnv


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

    def solve(self) -> dict:
        """Run Value Iteration until convergence.

        Iteratively applies the Bellman optimality update:
            V(s) = max_a [ r(s,a) + gamma * V(s') ]
        until max|V_new - V_old| < theta.

        Returns:
            info: dict with keys 'iterations', 'delta_history', 'time'
        """
        # TODO: implement VI loop
        raise NotImplementedError

    def get_policy(self) -> dict:
        """Compute greedy policy from current V.

        For each state: pi(s) = argmax_a [ r(s,a) + gamma * V(s') ]

        Returns:
            dict: {state: action}
        """
        # TODO: derive policy from self._V using argmax
        raise NotImplementedError

    def get_value_function(self) -> dict:
        """Return current value function.

        Returns:
            dict: {state: float}
        """
        return self._V
