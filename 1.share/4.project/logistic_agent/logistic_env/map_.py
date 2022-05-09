from .common import *


def create_map(num_row, num_col, start=None, goal=None, wall=None):
    if start is None:
        start = [0, 0]
    if goal is None:
        goal = [num_row, num_col]
    map_ = [[INITIAL_ENV_FLOOR for _ in range(num_col)] for _ in range(num_row)]

    for idx_row, idx_col in wall:
        map_[idx_row][idx_col] = INITIAL_ENV_WALL

    map_[start[0]][start[1]] = INITIAL_ENV_START
    map_[goal[0]][goal[1]] = INITIAL_ENV_GOAL
    return ["".join(x) for x in map_]
