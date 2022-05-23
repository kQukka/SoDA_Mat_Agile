import numpy as np
from common.func_ import random_argmax
from .common import one_hot


class Agent:
    def __init__(self, env, size_input, size_output):
        self.env = env
        self._size_input = size_input
        self._size_output = size_output
        self.min_epsilon = 0.01
        self.max_epsilon = 1.0
        self.epsilon = 1.0

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

    def _decay_epsilon(self, greedy):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * greedy

    def _get_action_noise(self, q_value, idx_epi=0, greedy=0, noise=False):
        if greedy:
            if idx_epi == 0:
                self.epsilon = self.max_epsilon
            if np.random.rand(1) < self.epsilon:
                return np.random.choice(self.env.num_action)
            else:
                return random_argmax(q_value)
        elif noise:
            return np.argmax(q_value + np.random.randn(1, self.env.num_action) / (idx_epi + 1))
        else:
            return random_argmax(q_value)

    def _one_hot(self, state):
        return one_hot(state, self._size_input)

    def _convert_p_to_idx(self, p_):
        return (p_[0] * self.env.width) + p_[1]

