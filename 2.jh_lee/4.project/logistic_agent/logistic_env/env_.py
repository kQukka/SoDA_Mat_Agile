import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

from .common import *
from .env_gym import LogisticEnv


# map_env must be list[list]
# random_map = (size_map, percentage_wall)
def create_env(id_, manual=False, map_name='4x4', map_env=None, random_map=None, is_slippery=False):
    setting = None
    if manual:
        setting = {'map_env': map_env, 'is_slippery': is_slippery}
    elif random_map:
        setting = {'random_map': (random_map[0], random_map[1]), 'is_slippery': is_slippery}
    else:
        setting = {'map_name': map_name, 'is_slippery': is_slippery}

    register(
        id=id_,
        entry_point=LogisticEnv,
        kwargs=setting
    )
    env = gym.make(id_)
    return env


def get_env_list(env):
    return env.registry.all()


def get_env_spec(env, id_):
    return env.registry.spec(id_)


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


class Env:
    def __init__(self):
        pass


