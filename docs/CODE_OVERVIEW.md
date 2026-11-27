# Code overview

How it is built and where to change things. File-by-file detail in `docs/code/`. Keep updated when files change.

- `gym_maze/__init__.py`: registers the gym env ids (`maze-sample-*`, `maze-random-*`, `maze-random-custom-v0`, `*-plus-v0`).
- `gym_maze/envs/maze_env.py`: `MazeEnv` (gym.Env: step/reset/render/restore, rewards) and one subclass per preset size.
- `gym_maze/envs/maze_view_2d.py`: `MazeView2D` (pygame drawing, robot position) and `Maze` (cells, walls, portals, load/save .npy).
- `gym_maze/envs/maze_generator.py`: standalone maze generator.
- `gym_maze/envs/maze_samples/*.npy`: pre-generated mazes for the `maze-sample-*` envs.
- `docs/AUTOMATION.md`: daily maintenance bot runbook and merge guardrails.
- `tests/test_maze_env.py`: fast unittest suite (headless via `SDL_VIDEODRIVER=dummy`).
