import numpy as np
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ParkourEnv:
    """Deterministic Parkour grid-world environment.

    Builds a full transition table T[state][action] -> (next_state, reward)
    from the provided config (height map, hp_start, rewards).
    """

    def __init__(self, env_config: dict):
        """Initialize environment from config dict.

        Args:
            env_config: parsed env.yaml containing keys
                - height_map: list[list[int]]  (8x8)
                - hp_start: int
                - rewards: dict with keys 'victory', 'death', 'step'
        """
        self.height_map: np.ndarray = np.array(env_config["height_map"])
        self.rows, self.cols = self.height_map.shape
        self.hp_start: int = env_config["hp_start"]
        self.rewards: dict = env_config["rewards"]

        # Build transition table: T[(i, j, hp)][action] = ((i', j', hp'), reward)
        self.T: dict = {}
        self._build_transition_table()

    def _build_transition_table(self) -> None:
        """Construct the full deterministic transition table for all states and actions."""
        # TODO: iterate over all (i, j, hp) and all actions
        # For each (state, action) compute next_state and reward according to the spec:
        #   - invalid moves (dead, out of bounds, dh > 3) -> self-loop
        #   - valid moves -> apply fall damage, check death/victory
        # Store in self.T
        raise NotImplementedError

    def step(self, state: tuple, action: int) -> tuple:
        """Look up transition for (state, action).

        Args:
            state: (i, j, hp)
            action: int from Action enum

        Returns:
            (next_state, reward, done, dead)
            - next_state: (i', j', hp')
            - reward: float
            - done: bool — True if terminal (victory or death)
            - dead: bool — True if agent died
        """
        # TODO: lookup self.T[state][action], determine done/dead flags
        raise NotImplementedError

    def get_transition_table(self) -> dict:
        """Return the full transition table.

        Returns:
            dict: T[state][action] = (next_state, reward)
        """
        return self.T

    def get_all_states(self) -> list:
        """Return list of all possible states (i, j, hp).

        Returns:
            list[tuple]: all (i, j, hp) for i in [0..rows-1], j in [0..cols-1], hp in [0..hp_start]
        """
        # TODO: generate all state tuples
        raise NotImplementedError

    def get_actions(self) -> list:
        """Return list of all possible actions.

        Returns:
            list[int]: [UP, DOWN, LEFT, RIGHT]
        """
        return list(Action)
