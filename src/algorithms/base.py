from abc import ABC, abstractmethod

from src.environment.parkour_env import ParkourEnv


class BaseAlgorithm(ABC):
    """Base class for all DP-RL algorithms."""

    def __init__(self, env: ParkourEnv, config: dict):
        """
        Args:
            env: initialized ParkourEnv instance
            config: parsed algorithm config (gamma, theta, etc.)
        """
        self.env = env
        self.config = config

    @abstractmethod
    def solve(self) -> dict:
        """Run the algorithm.

        Returns:
            info: dict with convergence details (iterations, delta_history, time, ...)
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy(self) -> dict:
        """Return current policy.

        Returns:
            dict: {state: action} mapping
        """
        raise NotImplementedError

    @abstractmethod
    def get_value_function(self) -> dict:
        """Return current value function.

        Returns:
            dict: {state: float} mapping
        """
        raise NotImplementedError
