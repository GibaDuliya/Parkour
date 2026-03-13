from abc import ABC, abstractmethod
import numpy as np
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

        n_states  = len(self.env.get_all_states())
        n_actions = len(self.env.get_actions())

        self.V = np.zeros(n_states)
        self.policy = np.zeros((n_states, n_actions))
        random_actions = np.random.randint(0, n_actions, size=n_states)
        self.policy[np.arange(n_states), random_actions] = 1.0

        self.R = np.zeros((n_states, n_actions))
        self.T_matr = np.zeros((n_states, n_actions, n_states), dtype=int)
        for state_id in range(n_states):
            state = self.id2state(state_id)
            for action in range(n_actions):
                next_state, reward, _, _ = self.env.step(state, action)
                self.R[state_id, action] = reward
                self.T_matr[state_id, action, self.state2id(next_state)] = 1

        


    def state2id(self, state: tuple) -> int:
        """Map state (row, col, hp) to a flat integer index.

        Args:
            state: tuple (i, j, hp)

        Returns:
            int in [0, n_states)
        """
        i, j, hp = state
        return i * (self.env.cols * self.env.hp_start) + j * self.env.hp_start + hp

    def id2state(self, idx: int) -> tuple:
        """Map flat integer index back to state (row, col, hp).

        Args:
            idx: int in [0, n_states)

        Returns:
            tuple (i, j, hp)
        """
        i  = idx // (self.env.cols * self.env.hp_start)
        j  = (idx % (self.env.cols * self.env.hp_start)) // self.env.hp_start
        hp = idx % self.env.hp_start
        return (i, j, hp)

    @abstractmethod
    def solve(self) -> dict:
        """Run the algorithm.

        Returns:
            info: dict with convergence details (iterations, delta_history, time, ...)
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy(self) -> np.ndarray:
        """Return current policy as a (n_states, n_actions) probability matrix."""
        raise NotImplementedError

    @abstractmethod
    def get_value_function(self) -> np.ndarray:
        """Return current value function as a (n_states,) array."""
        raise NotImplementedError
