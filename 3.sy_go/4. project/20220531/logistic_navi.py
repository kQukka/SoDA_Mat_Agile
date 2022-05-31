import os
import time
import gc
from datetime import datetime
import gym
from gym.envs.registration import register
# ------------------------------------------------------------------------------------------------------
import pandas as pd
from common.func_ import save, load, create_dir
from agent.common import get_set_env, print_env_map, get_idx_direct, print_str_direct, get_path
from agent.common import EarlyStopping
from agent.dqn import DQN
from env_.logistic.logistic import LogisticEnv
from env_.logistic.common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH_LOCAL = './'
PATH_INFO_BOX = PATH_LOCAL+'data/box.csv'
PATH_INFO_OBSTACLES = PATH_LOCAL+'data/obstacles.csv'
PATH_DATA_TRAIN = PATH_LOCAL+'data/factory_order_train.csv'


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
    agent.min_epsilon = 0.1
    q_map, reward = agent.run(10000,
                              buffer=10000, sampling=128,
                              size_hidden=40, epoch=10, learning_rate=0.001, interval_train=100,
                              discount=0.8, early_stopping=EarlyStopping(ratio=70), save_result=True,
                              greedy=0.999, run_time=1000, max_step=90)

if __name__ == '__main__':
    run_dqn()


