import os
import time
import gc
import gym
from gym.envs.registration import register
# ------------------------------------------------------------------------------------------------------
from common.setting import *
from common.csv_ import save, load
from logistic_env.env_ import create_env, create_map
from agent.dqn import DQN
from agent.common import get_set_env, print_env_map, get_idx_direct, print_str_direct, get_path
from agent.common import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    gc.collect()
    input_ = ['C', 'J']
    p_setting_env = list(get_set_env(input_, POINT_START, POINT_END, POINT_WALL, POINT_ITEM))

    id_env = ['LogisEnv-v1', 'LogisEnv-v2', 'LogisEnv-v3']
    path_w = ['./logistic_q_net_w_1.csv', './logistic_q_net_w_2.csv', './logistic_q_net_w_3.csv']
    path_map = ['./logistic_q_net_map_1.csv', './logistic_q_net_map_2.csv', './logistic_q_net_map_3.csv']
    path = ['./logistic_q_net_1_11.csv', './logistic_q_net_1_2.csv', './logistic_q_net_1_3.csv']

    list_map = []
    list_agent = []
    # ------------------------------------------------------------------------------------------------------
    for idx, (p_start_end, p_wall) in enumerate(p_setting_env):
        # frozenlake_4x4
        # env = create_env(id_=id_env[idx])

        # project env
        map_env = create_map(NUM_ROW, NUM_COL, start=p_start_end[0], goal=p_start_end[1], wall=p_wall)
        env = create_env(id_=id_env[idx], manual=True, map_env=map_env)

        print_env_map(map_env)
        save(path_map[idx], map_env)

        agent = DQN(env)
        q_map, reward = agent.run(1000,
                                  buffer=1000, sampling=64,
                                  size_hidden=30, epoch=30, learning_rate=0.01, interval_train=10,
                                  discount=0.5, early_stopping=EarlyStopping(ratio=70))

        agent.save_q_map(path[idx], q_map)
        save(path_w[idx], q_map)
        list_map.append(q_map)
        list_agent.append(agent)
        break

    path_ = []
    for idx, (p_start_end, p_wall) in enumerate(p_setting_env):
        weight = list_map[idx]
        map_idx_direct = get_idx_direct(weight, NUM_COL)
        buf_path = get_path(map_idx_direct, p_start_end[0], p_start_end[1], NUM_ROW, NUM_COL)
        path_.append(buf_path)
        print_str_direct(map_idx_direct, NUM_COL)
    gc.collect()
    pass


if __name__ == '__main__':
    main()

