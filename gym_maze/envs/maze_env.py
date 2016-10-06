import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None):

        self.viewer = None

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640))
        elif maze_size:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size,
                                        screen_size=(640, 640))
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self._seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()

    def __del__(self):
        self.maze_view.quit_game()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(action)

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            done = True
        else:
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

        self.state = self.maze_view.robot

        info = {}

        return self.state, reward, done, info

    def _reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def _render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy")


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5))


class MazeEnvSample10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy")


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10))


class MazeEnvSample3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy")


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3))


class MazeEnvSample100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy")


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100))

