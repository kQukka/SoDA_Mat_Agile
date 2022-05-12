import numpy as np

from logistic.common.csv_ import save, load
from logistic.common import random_argmax

IDX_LOG_STATE = 0
IDX_LOG_Q_MAP = 1
IDX_LOG_REWARD = 2
IDX_LOG_ACTION = 3


class Agent:
    def __init__(self, env):
        self.env = env
        self._input_size = self.env.observation_space.n
        self._output_size = self.env.action_space.n

        self._name_arg = []
        self._init_setting = []
        # self._log_epi = [[state_step, q_map_step, reward_step, act_step], ...]
        self._log_epi = []
        self.__num_progress = 0
        self._enable_log = [False, False, False, False]

    @staticmethod
    def save_q_map(path, q_map):
        return save(path, q_map)

    @staticmethod
    def load_q_map(path):
        return load(path)

    def enable_log_state(self, enable):
        self._enable_log[IDX_LOG_STATE] = enable

    def enable_log_q_map(self, enable):
        self._enable_log[IDX_LOG_Q_MAP] = enable

    def enable_log_reward(self, enable):
        self._enable_log[IDX_LOG_REWARD] = enable

    def enable_log_action(self, enable):
        self._enable_log[IDX_LOG_ACTION] = enable

    def get_log_state(self):
        return self._get_log_item(IDX_LOG_STATE)

    def get_log_q_map(self):
        return self._get_log_item(IDX_LOG_Q_MAP)

    def get_log_reward(self):
        return self._get_log_item(IDX_LOG_REWARD)

    def get_log_action(self):
        return self._get_log_item(IDX_LOG_ACTION)

    def run(self, num_episodes, q_map=None, early_stopping=False, **kwargs):
        return True

    def _run_episodes(self, q_map, **kwargs):
        return True

    def _load_matrix(self, matrix_, random=False, low=0, high=0.01):
        if matrix_:
            return np.array(matrix_)
        elif random:
            size = [self.env.observation_space.n, self.env.action_space.n]
            return np.random.uniform(size=size, low=low, high=high)
        else:
            return np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def _get_setting(self, setting):
        buf = self._init_setting.copy()
        if setting:
            for k in setting.keys():
                idx = self._name_arg.index(k)
                buf[idx] = setting[k]
        return buf

    def _print_progress(self, i, num_episodes):
        if i == 0:
            self.__num_progress = 0
        num_ = int((i / num_episodes) * 100)
        if self.__num_progress != num_:
            print(f'progress = {num_} %')
            self.__num_progress = num_
        return num_

    def _check_early_stopping(self, early_stopping):
        if early_stopping:
            if early_stopping.check_stopping(self._log_epi[-1][IDX_LOG_ACTION]):
                return True

    def _get_action_noise(self, q_value, idx=0, greedy=False, noise=False):
        if greedy:
            e = 1. / ((idx // 1000) + 1)
            if np.random.rand(1) < e:
                return self.env.action_space.sample()
            else:
                return random_argmax(q_value)
        elif noise:
            return np.argmax(q_value + np.random.randn(1, self.env.action_space.n) / (idx + 1))
        else:
            return random_argmax(q_value)

    def _get_log_sum_reward(self):
        return [sum(reward_epi) for reward_epi in self.get_log_reward()]

    def _get_log_item(self, idx):
        log_item = []
        for log_epi in self._log_epi:
            log_item.append(log_epi[idx])
        return log_item


