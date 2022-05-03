import numpy as np

from logistic_agent.common.csv_ import save, load
from logistic_agent.common.func import random_argmax


IDX_LOG_STATE = 0
IDX_LOG_Q_MAP = 1
IDX_LOG_REWARD = 2
IDX_LOG_ACTION = 3


class QLearning:
    def __init__(self, env):
        self.env = env
        # self.__log_epi = [[state_step, q_map_step, reward_step, act_step], ...]
        self.__log_epi = []

    def get_log_state(self):
        return self.__get_log_item(IDX_LOG_STATE)

    def get_log_q_map(self):
        return self.__get_log_item(IDX_LOG_Q_MAP)

    def get_log_reward(self):
        return self.__get_log_item(IDX_LOG_REWARD)

    def get_log_action(self):
        return self.__get_log_item(IDX_LOG_ACTION)

    def save_q_map(self, path, q_map):
        return save(path, q_map)

    def load_q_map(self, path):
        return load(path)

    def run(self, num_episodes, q_map=None, early_stopping=False):
        del self.__log_epi
        self.__log_epi = []
        if early_stopping:
            early_stopping.clear()

        if q_map:
            q_map = np.array(q_map)
        else:
            q_map = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        num_progress = 0
        for i in range(num_episodes):
            # __run_episodes(self, q_map, idx=0, greedy=False, noise=False, learning_rate=0, discount=1):
            # self.__run_episodes(q_map)
            self.__run_episodes(q_map, noise=True, discount=0.9)

            num_ = int((i / num_episodes) * 100)
            if num_progress != num_:
                print(f'progress = {num_} %')
                num_progress = num_
            if early_stopping:
                if early_stopping.check_stopping(self.__log_epi[-1][IDX_LOG_ACTION]):
                    print(f'progress = {num_} %  --> {i}/{num_episodes} Early Stopping')
                    break
        sum_reward_by_epi = self.__get_log_sum_reward()
        return q_map, sum_reward_by_epi

    def __run_episodes(self, q_map, idx=0, greedy=False, noise=False, learning_rate=0, discount=1.):
        log_step = [[] for _ in range(4)]
        # 초기 state 설정 (Start)
        state = self.env.reset()

        done = False
        while not done:
            act = None

            # e-greedy
            if greedy:
                e = 1. / ((idx // 1000) + 1)
                if np.random.rand(1) < e:
                    act = self.env.action_space.sample()
                else:
                    act = random_argmax(q_map[state, :])
            # noise
            elif noise:
                act = np.argmax(q_map[state, :] + np.random.randn(1, self.env.action_space.n) / (idx + 1))
            else:
                act = random_argmax(q_map[state, :])
            new_state, reward, done, _ = self.env.step(act)

            if state != new_state:
                if learning_rate > 0:
                    q_map[state, act] = (1 - learning_rate) * q_map[state, act] + learning_rate * (
                                reward + discount * np.max(q_map[new_state, :]))
                else:
                    q_map[state, act] = reward + discount * np.max(q_map[new_state, :])
            # log
            for idx, val in enumerate([state, 0, reward, act]):
                log_step[idx].append(val)
            # update state
            state = new_state
        self.__log_epi.append(log_step)
        return True

    def __get_log_sum_reward(self):
        return [sum(reward_epi) for reward_epi in self.get_log_reward()]

    def __get_log_item(self, idx):
        log_item = []
        for log_epi in self.__log_epi:
            log_item.append(log_epi[idx])
        return log_item


