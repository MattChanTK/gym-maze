# gym-maze

A simple 2D maze environment where an agent (blue dot) finds its way from the top left corner (blue square) to the goal at the bottom right corner (red square). 
The objective is to find the shortest path from the start to the goal.

<kbd>![Simple 2D maze environment](http://i.giphy.com/Ar3aKxkAAh3y0.gif)</kbd>

### Action space
The agent may only choose to go up, down, right, or left ("N", "S", "E", "W"). If the way is blocked, it will remain at the same the location. 

### Observation space
The observation space is the (x, y) coordinate of the agent. The top left cell is (0, 0).

### Reward
A reward of 1 is given when the agent reaches the goal. For every step in the maze, the agent recieves a reward of -0.1/(number of cells).

### End condition
The maze is reset when the agent reaches the goal. 

## Maze Versions

### Pre-generated mazes
* 3 cells x 3 cells: _maze-sample-3x3-v0_
* 5 cells x 5 cells: _maze-sample-5x5-v0_
* 10 cells x 10 cells: _maze-sample-10x10-v0_
* 100 cells x 100 cells: _maze-sample-100x100-v0_

### Randomly generated mazes (same maze every epoch)
* 3 cells x 3 cells: _maze-random-3x3-v0_
* 5 cells x 5 cells: _maze-random-5x5-v0_
* 10 cells x 10 cells: _maze-random-10x10-v0_
* 100 cells x 100 cells: _maze-random-100x100-v0_
* any size: _maze-random-custom-v0_, e.g. `gym.make("maze-random-custom-v0", maze_size=(150, 150))`

### Randomly generated mazes with portals and loops
With loops, it means that there will be more than one possible path.
The agent can also teleport from a portal to another portal of the same colour. 
* 10 cells x 10 cells: _maze-random-10x10-plus-v0_
* 20 cells x 20 cells: _maze-random-20x20-plus-v0_
* 30 cells x 30 cells: _maze-random-30x30-plus-v0_

## Installation
Requires Python 3.8+, [gymnasium](https://gymnasium.farama.org) 1.0+, pygame and numpy (1.x or 2.x).

Since v1.0 this package uses gymnasium, the maintained successor of OpenAI gym. Coming from the old gym API:
* `import gymnasium as gym` instead of `import gym` (still `import gym_maze` to register the envs).
* `obs, info = env.reset()` instead of `obs = env.reset()`.
* `obs, reward, terminated, truncated, info = env.step(a)` instead of `obs, reward, done, info`.
* Pass `render_mode="human"` or `"rgb_array"` to `gym.make`; `env.render()` takes no arguments.

Need the old gym API? Install a pre-1.0 commit with `gym<0.26` and `numpy<2` (Python 3.12 or older).

```bash
cd gym-maze
pip install .
```
## Options
* `render_mode=None` (default): headless, no pygame window; `render()` returns `None`. `"human"` opens a window, `"rgb_array"` makes `render()` return an image. The older `enable_render=True/False` still works.
* Actions: 0=N, 1=S, 2=E, 3=W (or the strings). Walking into a wall leaves the agent in place.
* `zero_step_rewards=True`: reward 0 per step instead of -0.1/(number of cells).
* `env.reset(options={"start": (x, y)})`: start an episode from a given cell. `env.restore((x, y))` moves the agent mid-episode.

## Examples
An example of finding the shortest path through the maze using Q-learning can be found here: https://github.com/tuzzer/ai-gym/blob/master/maze_2d/maze_2d_q_learning.py

![Solving 20x20 maze with loops and portals using Q-Learning](http://i.giphy.com/rfazKQngdaja8.gif)

