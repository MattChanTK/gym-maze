import os
from gym_maze.envs.maze_view_2d import Maze

if __name__ == "__main__":

    # check if the folder "maze_samples" exists in the current working directory
    dir_name = os.path.join(os.getcwd(), "maze_samples")
    if not os.path.exists(dir_name):
        # create it if it doesn't
        os.mkdir(dir_name)

    # increment number until it finds a name that is not being used already (max maze_999)
    maze_path = None
    for i in range(1, 1000):
        maze_name = "maze2d_%03d.npy" % i
        maze_path = os.path.join(dir_name, maze_name)
        if not os.path.exists(maze_path):
            break
        if i == 999:
            raise ValueError("There are already 999 mazes in the %s." % dir_name)

    maze = Maze(maze_size=(5, 5))
    maze.save_maze(maze_path)
    print("New maze generated and saved at %s." %  maze_path)

