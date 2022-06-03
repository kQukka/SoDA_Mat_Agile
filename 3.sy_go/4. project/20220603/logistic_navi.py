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

    # -----------------------------------------------------------------------------------------------------------
    # [C] item Train
    # -----------------------------------------------------------------------------------------------------------
    row_str = list(order_train.iloc[0])[0]
    items = list(set(env.NAME_ITEM) & set(row_str))
    items.sort()
    env.set_p_order(items)
    #env.set_route((9, 4), env.get_p_item(items[0]), items[0])
    env.set_route(env.get_p_item('A'), env.get_p_item('B'), 'B')
    agent = DQN(env, size_input=env.height * env.width, size_output=env.num_action)
    agent.min_epsilon = 0.1
    q_map, reward = agent.run(10000,
                              buffer=10000, sampling=128,
                              size_hidden=64, epoch=10, learning_rate=0.001, interval_train=100,
                              discount=0.8, early_stopping=EarlyStopping(ratio=70), save_result=True,
                              greedy=0.999, run_time=1000, max_step=90)


    # -----------------------------------------------------------------------------------------------------------
    # order [1~3] Train
    # -----------------------------------------------------------------------------------------------------------
    '''
    agent = DQN(env, size_input=env.height * env.width, size_output=env.num_action)
    for order_idx in range(3):
        row_str = list(order_train.iloc[order_idx])[0]
        items = list(set(env.NAME_ITEM) & set(row_str))
        items.sort()

        item_cnt = len(items)
        for item_idx in range(item_cnt + 1):
            print('-----------------------------------------------')
            if item_idx == 0:
                print('---> order_idx:{}  route:[{}] -> [{}]'.format(order_idx, 'S', items[item_idx]))
                env.set_route((9, 4), env.get_p_item(items[item_idx]), items[item_idx])  ### 시작점~ 아이템첫번째 (C)-> get_p_item 함수로 좌표를 가져옴
            elif item_idx == item_cnt:  # last_item -> end_point
                print('---> order_idx:{}  route:[{}] -> [{}]'.format(order_idx, items[item_idx - 1], 'S'))
                env.set_route(env.get_p_item(items[item_idx - 1]), (9, 4), 'S')
            else:
                print('---> order_idx:{}  route:[{}] -> [{}]'.format(order_idx, items[item_idx - 1],items[item_idx]))
                env.set_route(env.get_p_item(items[item_idx - 1]), env.get_p_item(items[item_idx]), items[item_idx])
            print('-----------------------------------------------')

            agent.min_epsilon = 0.1
            agent.load_train_weights() #train인 시 가중치 활용이 필요한 경우만 주석 풀기
            q_map, reward = agent.run(10000,
                                      buffer=10000, sampling=128,
                                      size_hidden=64, epoch=10, learning_rate=0.001, interval_train=100,
                                      discount=0.8, early_stopping=EarlyStopping(ratio=70), save_result=True,
                                      greedy=0.999, run_time=1000, max_step=90)'''



# -----------------------------------------------------------------------------------------------------------
# order [1~3] test
# order 전체를 도는 것을 한 episode 단위로함.
# -----------------------------------------------------------------------------------------------------------
def play_dqn():

    gc.collect()
    order_train = pd.read_csv(PATH_DATA_TRAIN)
    env = LogisticEnv(10, 9, PATH_INFO_BOX, PATH_INFO_OBSTACLES)
    agent = DQN(env, size_input=env.height * env.width, size_output=env.num_action)

    time_steps = 0
    total_reward = 0.0
    finish_count = 0

    order_count = 3 # order_train.shape[0]
    for order_idx in range(order_count):
        row_str = list(order_train.iloc[order_idx])[0]
        items = list(set(env.NAME_ITEM) & set(row_str))
        items.sort()

        item_cnt = len(items)
        picked_count = 0
        for item_idx in range(item_cnt + 1):
            if item_idx == 0:
                env.set_route((9, 4), env.get_p_item(items[item_idx]), items[item_idx])
            elif item_idx == item_cnt:
                env.set_route(env.get_p_item(items[item_idx - 1]), (9, 4), 'S')
            else:
                env.set_route(env.get_p_item(items[item_idx - 1]), env.get_p_item(items[item_idx]), items[item_idx])
            timestep, reward, picked = agent.play_agent(size_hidden=64, max_step=90)
            time_steps += timestep
            total_reward += reward
            if picked:
                picked_count += 1

        if item_cnt+1 == picked_count:  # 모든 아이템을 가지고 온경우
            finish_count += 1

        print('Episode:{}  Timestep:{}  Avr Reward:{:2.2f}   Finish Rate:{:.4f} Goal/Item:{}/{} '.format(
            order_idx, time_steps, total_reward/(order_idx+1), finish_count/(order_idx+1), picked_count, (item_cnt+1)))

if __name__ == '__main__':
    run_dqn()
    #play_dqn()



