from gym.envs.registration import register


register(
    id='maze-v0',
    entry_point='gym_maze.envs:MazeEnvSample5x5',
    timestep_limit=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='gym_maze.envs:MazeEnvSample5x5',
    timestep_limit=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='gym_maze.envs:MazeEnvRandom5x5',
    timestep_limit=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='gym_maze.envs:MazeEnvSample10x10',
    timestep_limit=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='gym_maze.envs:MazeEnvRandom10x10',
    timestep_limit=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='gym_maze.envs:MazeEnvSample3x3',
    timestep_limit=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='gym_maze.envs:MazeEnvRandom3x3',
    timestep_limit=1000,
    nondeterministic=True,
)


register(
    id='maze-sample-100x100-v0',
    entry_point='gym_maze.envs:MazeEnvSample100x100',
    timestep_limit=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='gym_maze.envs:MazeEnvRandom100x100',
    timestep_limit=1000000,
    nondeterministic=True,
)
