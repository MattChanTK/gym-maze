from setuptools import setup

setup(name="gym_maze",
      version="0.7",
      url="https://github.com/Kuzat/gym-maze",
      author="Matthew T.K. Chan",
      license="MIT",
      packages=["gym_maze", "gym_maze.envs"],
      package_data = {
          "gym_maze.envs": ["maze_samples/*.npy"]
      },
      install_requires = ["gym", "pygame", "numpy"]
)
