import os
import time
import gc
from datetime import datetime
import gym
from gym.envs.registration import register
# ------------------------------------------------------------------------------------------------------
import pandas as pd
from common.func_ import save, load, create_dir, edit
from agent.common import get_set_env, print_env_map, get_idx_direct, print_str_direct, get_path
from agent.common import EarlyStopping
from agent.dqn import DQN
from agent.q_network import QNetwork
from agent.q_learning import QLearning
from agent.a2c import A2C
from env_.logistic.logistic import LogisticEnv
from env_.logistic.common import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH_LOCAL = './'
PATH_INFO_BOX = PATH_LOCAL+'data/box.csv'
PATH_INFO_OBSTACLES = PATH_LOCAL+'data/obstacles.csv'
PATH_DATA_TRAIN = PATH_LOCAL+'data/factory_order_train.csv'


def run_a2c():
    gc.collect()
    order_train = pd.read_csv(PATH_DATA_TRAIN)
    env = LogisticEnv(10, 9, PATH_INFO_BOX, PATH_INFO_OBSTACLES)

    row_str = list(order_train.iloc[0])[0]
    items = list(set(env.NAME_ITEM) & set(row_str))
    items.sort()
    env.set_p_order(items)
    env.set_route((9, 4), env.get_p_item(items[0]))
    env.REWARD.MOVE = 0.001
    env.REWARD.MOVE_SUB = -0.001
    env.REWARD.GOAL = 10
    env.REWARD.NOT_MOVE = -10
    env.REWARD.OBSTACLE = -3
    env.REWARD.OUT_GIRD = -2
    env.REWARD.RETURN = 0

    agent = A2C(env, size_input=env.height * env.width, size_output=env.num_action)
    q_map, reward = agent.run(500,
                              epoch=20, learning_rate=0.01,
                              discount=0.5
                              , early_stopping=EarlyStopping(ratio=70), save_result=True)


def run_dqn():
    gc.collect()
    order_train = pd.read_csv(PATH_DATA_TRAIN)
    env = LogisticEnv(10, 9, PATH_INFO_BOX, PATH_INFO_OBSTACLES)

    row_str = list(order_train.iloc[0])[0]
    items = list(set(env.NAME_ITEM) & set(row_str))
    items.sort()
    env.set_p_order(items)
    env.set_route((9, 4), env.get_p_item(items[0]))

    agent = DQN(env, size_input=env.height * env.width, size_output=env.num_action)
    q_map, reward = agent.run(1000,
                              buffer=500, sampling=10,
                              size_hidden=40, epoch=3, learning_rate=0.001, interval_train=10,
                              discount=0.95, early_stopping=EarlyStopping(ratio=70), save_result=True, greedy=0.999)


def run_q_learning():
    gc.collect()
    order_train = pd.read_csv(PATH_DATA_TRAIN)
    env = LogisticEnv(10, 9, PATH_INFO_BOX, PATH_INFO_OBSTACLES)

    row_str = list(order_train.iloc[0])[0]
    items = list(set(env.NAME_ITEM) & set(row_str))
    items.sort()
    env.set_p_order(items)
    env.set_route((9, 4), env.get_p_item(items[0]))

    agent = QLearning(env, size_input=env.height * env.width, size_output=env.num_action)
    agent.path_result = f'./log/test4.csv'

    # print('No ======== ')
    # for _ in range(100):
    #     q_map, reward = agent.run(2000,
    #                               early_stopping=EarlyStopping(ratio=100), save_result=True)
    #
    # print('greedy ======== ')
    # for _ in range(100):
    #     q_map, reward = agent.run(2000, greedy=0.99,
    #                               early_stopping=EarlyStopping(ratio=100), save_result=True)
    #
    # print('noise ======== ')
    # for _ in range(100):
    #     q_map, reward = agent.run(2000, noise=True,
    #                               early_stopping=EarlyStopping(ratio=100), save_result=True)
    #
    # print('discount ======== ')
    # for _ in range(100):
    #     q_map, reward = agent.run(2000, discount=0.9,
    #                               early_stopping=EarlyStopping(ratio=100), save_result=True)

    # edit(agent.path_result, [['greedy 0.99 - 0.90, disc 1.0 - 0.1']])
    # for disc in range(10):
    #     for greedy in range(1, 11):
    #         for _ in range(10):
    #             q_map, reward = agent.run(2000, greedy=1-greedy/100, discount=1-disc/10,
    #                                       early_stopping=EarlyStopping(ratio=100), save_result=True)
    # edit(agent.path_result, [['']])

    # edit(agent.path_result, [['greedy 0.9 - 0.1, disc 1.0 - 0.1']])

    for disc in range(1, 11):
        for greedy in range(1, 11):
            for _ in range(100):
                q_map, reward = agent.run(1000, greedy=round(1-round(greedy/10, 1), 1),
                                          discount=round(1-round(disc/10, 1), 1),
                                          early_stopping=EarlyStopping(patience=20, ratio=80), save_result=True)

    # for disc in range(1, 9, 1):
    #     q_map, reward = agent.run(2000, greedy=0.99, noise=True, lr_action=1, discount=disc/10,
    #                               early_stopping=EarlyStopping(ratio=100), save_result=True)

    # q_map, reward = agent.run(2000, greedy=False, noise=True, discount=0.9,
    #                           early_stopping=EarlyStopping(ratio=70), save_result=True)


def run_q_net():
    gc.collect()
    order_train = pd.read_csv(PATH_DATA_TRAIN)
    env = LogisticEnv(10, 9, PATH_INFO_BOX, PATH_INFO_OBSTACLES)

    row_str = list(order_train.iloc[0])[0]
    items = list(set(env.NAME_ITEM) & set(row_str))
    items.sort()
    env.set_p_order(items)
    env.set_route((9, 4), env.get_p_item(items[0]))

    agent = QNetwork(env, size_input=env.height * env.width, size_output=env.num_action)
    q_map, reward = agent.run(2000, greedy=False, noise=True, discount=0.9,
                              early_stopping=EarlyStopping(ratio=70), save_result=True)


if __name__ == '__main__':
    # run_q_learning()
    for _ in range(20):
        run_a2c()


