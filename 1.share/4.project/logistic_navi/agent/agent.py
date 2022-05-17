import numpy as np
from common.func_ import random_argmax


class Agent:
    def __init__(self, env, size_input, size_output):
        self.env = env
        self._size_input = size_input
        self._size_output = size_output

        self._name_setting = []
        self._init_setting = []
        self.log = {}

        self.__num_progress = 0

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
                idx = self._name_setting.index(k)
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

    def _get_action_noise(self, q_value, idx=0, greedy=False, noise=False):
        if greedy:
            e = 1. / ((idx // 1000) + 1)
            if np.random.rand(1) < e:
                return np.random.choice(self.env.num_action)
            else:
                return random_argmax(q_value)
        elif noise:
            return np.argmax(q_value + np.random.randn(1, self.env.action_space.n) / (idx + 1))
        else:
            return random_argmax(q_value)