import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gym_maze.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, render_mode=None,
                 enable_render=None, zero_step_rewards=False):
        """render_mode: None, "human" or "rgb_array" (gymnasium style).
        enable_render: older switch, kept for compatibility. True with no render_mode means "human".
        """
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("render_mode must be one of %s" % self.metadata["render_modes"])
        if enable_render is None:
            enable_render = render_mode is not None
        elif enable_render and render_mode is None:
            render_mode = "human"
        elif not enable_render:
            render_mode = None

        self.render_mode = render_mode
        self.enable_render = enable_render
        self.zero_step_rewards = zero_step_rewards

        if maze_file:
            self.maze_view = MazeView2D(maze_name="Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640),
                                        enable_render=enable_render)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=np.int64)
        high = np.array(self.maze_size, dtype=np.int64) - 1
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        self.state = np.zeros(2, dtype=np.int64)
        self.done = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def _obs(self):
        return np.array(self.maze_view.robot, dtype=np.int64)

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            self.maze_view.move_robot(self.ACTION[int(action)])
        else:
            self.maze_view.move_robot(action)

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1.0
            terminated = True
        else:
            if self.zero_step_rewards:
                reward = 0.0
            else:
                reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            terminated = False

        self.state = self._obs()
        self.done = terminated

        if self.render_mode == "human":
            self.render()

        return self.state.copy(), reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        """options={"start": (x, y)} starts the episode from that cell instead of (0, 0)."""
        super().reset(seed=seed)
        self.maze_view.reset_robot()
        if options and options.get("start") is not None:
            self.maze_view.restore_robot(options["start"])
        self.state = self._obs()
        self.done = False

        if self.render_mode == "human":
            self.render()

        return self.state.copy(), {}

    def restore(self, state):
        """Put the robot at `state` (an (x, y) cell) and continue the episode from there."""
        self.maze_view.restore_robot(state)
        self.state = self._obs()
        self.done = False
        return self.state.copy()

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode=None):
        """Returns an RGB array in "rgb_array" mode, else None. `mode` overrides render_mode (older API)."""
        mode = mode or self.render_mode
        if mode is None:
            return None
        img = self.maze_view.update(mode)
        return img if mode == "rgb_array" else None


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", **kwargs)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), **kwargs)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", **kwargs)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), **kwargs)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", **kwargs)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), **kwargs)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", **kwargs)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), **kwargs)


class MazeEnvRandomCustom(MazeEnv):

    def __init__(self, maze_size, **kwargs):
        super(MazeEnvRandomCustom, self).__init__(maze_size=tuple(maze_size), **kwargs)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", **kwargs)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", **kwargs)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, **kwargs):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", **kwargs)
