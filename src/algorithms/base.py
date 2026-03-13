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

        # Fixed ordering of states and actions (for numpy tables)
        self.states = self.env.get_all_states()
        self.actions = self.env.get_actions()

        n_states = len(self.states)
        n_actions = len(self.actions)

        # Fast mapping between state tuples and integer ids
        self._state_to_id = {s: i for i, s in enumerate(self.states)}
        self._id_to_state = list(self.states)

        self.V = np.zeros(n_states)
        self.policy = np.zeros((n_states, n_actions))
        random_actions = np.random.randint(0, n_actions, size=n_states)
        self.policy[np.arange(n_states), random_actions] = 1.0

        self.R = np.zeros((n_states, n_actions), dtype=float)                 # R[s, a]
        self.T_matr = np.zeros((n_states, n_actions, n_states), dtype=float)  # P[s, a, s']

        # Fill from environment transition table (deterministic)
        T = self.env.get_transition_table()  # dict: T[state][action] -> (next_state, reward)
        for s_id, s in enumerate(self.states):
            trans = T[s]
            for a_id, a in enumerate(self.actions):
                next_state, reward = trans[a]
                ns_id = self.state2id(next_state)
                self.R[s_id, a_id] = reward
                self.T_matr[s_id, a_id, ns_id] = 1.0

        


    def state2id(self, state: tuple) -> int:
        """Map state (row, col, hp) to a flat integer index.

        Args:
            state: tuple (i, j, hp)

        Returns:
            int in [0, n_states)
        """
        return self._state_to_id[state]

    def id2state(self, idx: int) -> tuple:
        """Map flat integer index back to state (row, col, hp).

        Args:
            idx: int in [0, n_states)

        Returns:
            tuple (i, j, hp)
        """
        return self._id_to_state[idx]

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
