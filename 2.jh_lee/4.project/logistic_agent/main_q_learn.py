import numpy as np
import matplotlib.pyplot as plt
import random as pr
import gc
# ------------------------------------------------------------------------
# from common import *
# from func_map import *

from common.setting import *
from common.csv_ import save, load
from logistic_env.env_ import create_env, create_map
from agent.q_learning import QLearning
from agent.common import get_set_env, print_env_map, get_idx_direct, print_str_direct, get_path
from agent.common import EarlyStopping


def main():
    gc.collect()
    input_ = ['C', 'J']
    p_setting_env = list(get_set_env(input_, POINT_START, POINT_END, POINT_WALL, POINT_ITEM))

    id_env = ['LogisEnv-v1', 'LogisEnv-v2', 'LogisEnv-v3']
    path_map = ['./logistic_q_learning_map_1.csv', './logistic_q_learning_map_2.csv', './logistic_q_learning_map_3.csv']
    path = ['./logistic_q_learning_1_1.csv', './logistic_q_learning_1_2.csv', './logistic_q_learning_1_3.csv']

    list_q_map = []
    list_agent = []
    for idx, (p_start_end, p_wall) in enumerate(p_setting_env):
        map_env = create_map(NUM_ROW, NUM_COL, start=p_start_end[0], goal=p_start_end[1], wall=p_wall)
        print_env_map(map_env)
        save(path_map[idx], map_env)

        env = create_env(id_=id_env[idx], manual=True, map_env=map_env)
        agent = QLearning(env)
        # q_map, sum_reward_by_epi = agent.run(NUM_EPISODES, early_stopping=EarlyStopping(), noise=True, discount=0.9)
        q_map, sum_reward_by_epi = agent.run(NUM_EPISODES, early_stopping=EarlyStopping())
        agent.save_q_map(path[idx], q_map.tolist())
        list_q_map.append(q_map)
        list_agent.append(agent)

    path_ = []
    for idx, (p_start_end, p_wall) in enumerate(p_setting_env):
        q_map = list_q_map[idx]
        map_idx_direct = get_idx_direct(q_map, NUM_COL)
        buf_path = get_path(map_idx_direct, p_start_end[0], p_start_end[1], NUM_ROW, NUM_COL)
        path_.append(buf_path)
        print_str_direct(map_idx_direct, NUM_COL)

    gc.collect()
    pass


if __name__ == '__main__':
    main()

