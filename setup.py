from setuptools import setup

setup(name="gym_maze",
      version="1.0",
      url="https://github.com/tuzzer/gym-maze",
      author="Matthew T.K. Chan",
      license="MIT",
      packages=["gym_maze", "gym_maze.envs"],
      package_data = {
          "gym_maze.envs": ["maze_samples/*.npy"]
      },
      python_requires=">=3.8",
      install_requires = ["gymnasium>=1.0", "pygame", "numpy"]
)
