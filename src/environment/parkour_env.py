import os
import numpy as np
import yaml
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
            env_config: either
                - landscape_id: int, hp_start, rewards — load height_map from landscape/landscape_{id}/
                - or height_map, hp_start, rewards, max_jump_up, safe_jump_down — inline (for experiments)
        """
        if "height_map" in env_config:
            self.height_map = np.array(env_config["height_map"])
            self.rows, self.cols = self.height_map.shape
            self.hp_start = int(env_config["hp_start"])
            self.hp_init = self.hp_start
            self.rewards = env_config["rewards"]
            self.max_jump_up = int(env_config["max_jump_up"])
            self.safe_jump_down = int(env_config["safe_jump_down"])
        else:
            landscape_id: int = env_config["landscape_id"]
            landscape_dir = os.path.join("landscape", f"landscape_{landscape_id}")
            with open(os.path.join(landscape_dir, "config.yaml")) as f:
                landscape_cfg = yaml.safe_load(f)
            self.height_map = np.load(os.path.join(landscape_dir, "height_map.npy"))
            self.rows, self.cols = self.height_map.shape
            min_hp_map = np.load(os.path.join(landscape_dir, "min_hp.npy"))
            self.hp_start = int(min_hp_map[min_hp_map > 0].max())  # state space bound
            self.hp_init = int(min_hp_map[0, 0])                   # rollout starting HP
            self.rewards = env_config["rewards"]
            self.max_jump_up = int(landscape_cfg["max_jump_up"])
            self.safe_jump_down = int(landscape_cfg["safe_jump_down"])

        # Build transition table: T[(i, j, hp)][action] = ((i', j', hp'), reward)
        self.T: dict = {}
        self._build_transition_table()

    def _build_transition_table(self) -> None:
        """Construct the full deterministic transition table for all states and actions."""
        rv = self.rewards["victory"]
        rd = self.rewards["death"]
        rs = self.rewards["step"]
        goal = (self.rows - 1, self.cols - 1)

        # (di, dj) for UP, DOWN, LEFT, RIGHT
        action_delta = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}

        for i in range(self.rows):
            for j in range(self.cols):
                for hp in range(self.hp_start + 1):
                    state = (i, j, hp)
                    self.T[state] = {}
                    h_ij = self.height_map[i, j]

                    for action in Action:
                        di, dj = action_delta[action]
                        i2, j2 = i + di, j + dj

                        # (0) Goal -> absorbing state, no reward
                        if (i, j) == goal and hp > 0:
                            self.T[state][action] = (state, 0)
                            continue

                        # (1) Dead -> self-loop, reward death
                        if hp <= 0:
                            self.T[state][action] = (state, rd)
                            continue

                        # (2) Out of bounds -> self-loop, step penalty
                        if not (0 <= i2 < self.rows and 0 <= j2 < self.cols):
                            self.T[state][action] = (state, rs)
                            continue

                        # (3) Jump too high -> self-loop, step penalty
                        h_i2j2 = self.height_map[i2, j2]
                        dh = h_i2j2 - h_ij
                        if dh > self.max_jump_up:
                            self.T[state][action] = (state, rs)
                            continue

                        # Valid move: apply fall damage and clamp HP to [0, hp_start]
                        damage = max(0, (h_ij - h_i2j2) - self.safe_jump_down)
                        hp_new = hp - damage
                        if hp_new < 0:
                            hp_new = 0
                        next_state = (i2, j2, hp_new)

                        # Reward: death before victory
                        if hp_new <= 0:
                            reward = rd
                        elif (i2, j2) == goal:
                            reward = rv
                        else:
                            reward = rs

                        self.T[state][action] = (next_state, reward)

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
        next_state, reward = self.T[state][action]
        i2, j2, hp2 = next_state
        goal = (self.rows - 1, self.cols - 1)
        dead = hp2 <= 0
        done = dead or ((i2, j2) == goal and hp2 > 0)
        return next_state, reward, done, dead

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
        return [
            (i, j, hp)
            for i in range(self.rows)
            for j in range(self.cols)
            for hp in range(self.hp_start + 1)
        ]

    def get_actions(self) -> list:
        """Return list of all possible actions.

        Returns:
            list[int]: [UP, DOWN, LEFT, RIGHT]
        """
        return list(Action)
