"""Fast checks for the maze envs. Run: python3 -m unittest discover -s tests"""
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gym  # noqa: E402
import numpy as np  # noqa: E402

import gym_maze  # noqa: E402,F401  (registers the envs)
from gym_maze.envs import MazeEnvSample5x5, MazeEnvRandomCustom  # noqa: E402


class MazeEnvTest(unittest.TestCase):

    def test_registered_envs_step(self):
        env = gym.make("maze-sample-5x5-v0")
        env.reset()
        obs, reward, done, info = env.step(env.action_space.sample())
        self.assertEqual(obs.shape, (2,))
        env.close()

    def test_numpy_int_action(self):
        env = MazeEnvSample5x5(enable_render=False)
        env.reset()
        start = env.state.copy()
        for a in range(4):
            env.step(np.int64(a))
        env.reset()
        np.testing.assert_array_equal(env.state, start)

    def test_returned_state_is_a_copy(self):
        env = MazeEnvSample5x5(enable_render=False)
        obs0 = env.reset()
        obs1, _, _, _ = env.step(1)
        obs2, _, _, _ = env.step(2)
        np.testing.assert_array_equal(obs0, [0, 0])
        self.assertIsNot(obs1, obs2)

    def test_render_disabled(self):
        env = MazeEnvSample5x5(enable_render=False)
        env.reset()
        env.step(1)
        self.assertIsNone(env.render(mode="rgb_array"))

    def test_render_enabled_rgb_array(self):
        env = MazeEnvSample5x5()
        img = env.render(mode="rgb_array")
        self.assertEqual(img.ndim, 3)

    def test_custom_size(self):
        env = gym.make("maze-random-custom-v0", maze_size=(7, 4), enable_render=False)
        self.assertEqual(tuple(env.unwrapped.maze_size), (7, 4))

    def test_zero_step_rewards(self):
        env = MazeEnvRandomCustom(maze_size=(6, 6), enable_render=False, zero_step_rewards=True)
        env.reset()
        _, reward, done, _ = env.step(0)
        if not done:
            self.assertEqual(reward, 0)

    def test_restore(self):
        env = MazeEnvSample5x5(enable_render=False)
        env.reset()
        obs = env.restore((2, 3))
        np.testing.assert_array_equal(obs, [2, 3])
        np.testing.assert_array_equal(env.maze_view.robot, [2, 3])

    def test_episode_reaches_goal(self):
        # Walk the maze with a BFS path and check the env ends with reward 1 at the goal.
        env = MazeEnvRandomCustom(maze_size=(8, 8), enable_render=False)
        maze = env.maze_view.maze
        goal = tuple(env.maze_view.goal)
        start = (0, 0)
        prev = {start: None}
        queue = [start]
        while queue:
            cell = queue.pop(0)
            for d, (dx, dy) in maze.COMPASS.items():
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt not in prev and maze.is_open(cell, d):
                    prev[nxt] = (cell, d)
                    queue.append(nxt)
        self.assertIn(goal, prev)
        path, cell = [], goal
        while prev[cell] is not None:
            cell, d = prev[cell]
            path.append(d)
        env.reset()
        for i, d in enumerate(reversed(path)):
            obs, reward, done, _ = env.step(d)
            self.assertEqual(done, i == len(path) - 1)
        self.assertEqual(reward, 1)
        np.testing.assert_array_equal(obs, goal)


if __name__ == "__main__":
    unittest.main()
