import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quarto_env.constants import BOARD_SIZE, NUM_PIECES, NUM_PLAYERS
from quarto_env.game_logic import check_win
from quarto_env.renderer import Renderer


class QuartoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_episode_steps=64):
        super().__init__()

        print("QuartoEnv initialized")

        self.action_space = spaces.Tuple((
            spaces.Discrete(NUM_PIECES),  # position
            spaces.Discrete(NUM_PIECES)   # next piece
        ))

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-1, high=NUM_PIECES - 1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8
                ),
                "current_piece": spaces.Discrete(NUM_PIECES),
                "remaining_pieces": spaces.Box(
                    low=0, high=1, shape=(NUM_PIECES,), dtype=np.int8
                ),
                "current_player": spaces.Discrete(NUM_PLAYERS),
            }
        )

        self.board = None
        self.current_piece = 0
        self.remaining_pieces = None
        self.current_player = 0
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.render_mode = render_mode
        self.renderer = Renderer() if render_mode in ("human", "rgb_array") else None

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and return an initial observation.

        Parameters:
        - seed (int, optional): Seed for random number generator.
        - options (dict, optional): Additional options for reset.

        Returns:
        - observation (dict): Initial observation of the environment.
        - info (dict): Additional info.
        """
        super().reset(seed=seed)

        self.board = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)

        self.remaining_pieces = np.ones(NUM_PIECES, np.int8)

        self.current_piece = 0  # TODO: should select a random spot to be current one
        self.remaining_pieces[self.current_piece] = 0

        self.current_player = 0
        self.step_count = 0

        info = {"action_mask": self._get_action_mask()}

        return self._get_observation(), info

    def step(self, action):
        """
        Execute one step in the environment.

        Parameters:
        - action (tuple): Action to be taken in the environment.

        Returns:
        - observation (dict): The observation after taking the action.
        - reward (float): Reward received after taking the action.
        - terminated (bool): TODO
        - truncated (bool): TODO
        - info (dict): Additional info.
        """
        position, next_piece = action
        self.step_count += 1

        row = position // 4
        col = position % 4

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.board[row, col] != -1:
            reward = -1.0
            if self.step_count >= self.max_episode_steps:
                truncated = True
            info["action_mask"] = self._get_action_mask()
            return self._get_observation(), reward, terminated, truncated, info

        self.board[row, col] = self.current_piece

        if check_win(self.board):
            reward = 1.0
            terminated = True
            info["action_mask"] = self._get_action_mask()
            return self._get_observation(), reward, terminated, truncated, info

        # Game can end in a draw after placing the final piece
        if np.all(self.board != -1):
            terminated = True
            info["action_mask"] = self._get_action_mask()
            return self._get_observation(), reward, terminated, truncated, info

        # If no piece remains, terminate without selecting next_piece
        if np.sum(self.remaining_pieces) == 0:
            terminated = True
            info["action_mask"] = self._get_action_mask()
            return self._get_observation(), reward, terminated, truncated, info

        if self.remaining_pieces[next_piece] == 0:
            reward = -1.0
            if self.step_count >= self.max_episode_steps:
                truncated = True
            info["action_mask"] = self._get_action_mask()
            return self._get_observation(), reward, terminated, truncated, info

        self.current_piece = next_piece
        self.remaining_pieces[next_piece] = 0

        self.current_player = 1 - self.current_player
        if self.step_count >= self.max_episode_steps:
            truncated = True

        info["action_mask"] = self._get_action_mask()
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """
        Get the current observation of the environment.

        Returns:
        - observation (dict): The current observation of the environment.
        """
        return {
            "board": self.board.copy(),
            "current_piece": int(self.current_piece),
            "remaining_pieces": self.remaining_pieces.copy(),
            "current_player": int(self.current_player),
        }

    def _get_action_mask(self):
        """
        Return valid-action masks for Tuple(placement_position, next_piece).
        """
        position_mask = (self.board.reshape(-1) == -1) # true only for empty cells
        next_piece_mask = (self.remaining_pieces == 1) # true only for available cells
        return {
            "position": position_mask.astype(np.bool_),
            "next_piece": next_piece_mask.astype(np.bool_),
        }

    def render(self):
        if self.renderer is None:
            return None
        return self.renderer.render(self.board)

    def close(self):
        if self.renderer is not None:
            import pygame

            pygame.quit()

