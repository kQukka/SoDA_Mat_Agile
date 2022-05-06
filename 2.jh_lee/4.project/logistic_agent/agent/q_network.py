import os
import time
# ------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from logistic_agent.agent.agent import Agent


class QNetwork(Agent):
    def __init__(self, env):
        super().__init__(env)
        self._name_arg = ['greedy', 'noise', 'learning_rate', 'discount']
        self._init_setting = [False, False, 0, 1.0]
        self.__optimizer = None
        self.__input_size = self.env.observation_space.n
        self.__output_size = self.env.action_space.n

    def get_weight(self, weight):
        return np.array(weight.numpy()).tolist()

    def run(self, num_episodes, weight=None, early_stopping=False, **kwargs):
        del self._log_epi
        self._log_epi = []
        if early_stopping:
            early_stopping.clear()

        learning_rate = 0.1
        # weight
        weight_init = tf.convert_to_tensor(self._load_matrix(weight, True), dtype=tf.float32)
        weight = tf.Variable(weight_init, dtype=tf.float32)
        # optimizer
        self.__optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

        start_time = time.time()
        # w_load = self._load_matrix(weight)
        for idx in range(num_episodes):
            self._run_episodes(weight, setting=kwargs)

            num_ = self._print_progress(idx, num_episodes)
            if self._check_early_stopping(early_stopping):
                print(f'progress = {num_} %  --> {idx}/{num_episodes} Early Stopping')
                break
        sum_reward_by_epi = self._get_log_sum_reward()
        print(f'{(time.time() - start_time)} seconds')
        return weight, sum_reward_by_epi

    # def __run_episodes(self, q_map, idx=0, greedy=False, noise=False, learning_rate=0, discount=1.):
    def _run_episodes(self, weight, idx=0, setting=None):
        greedy, noise, learning_rate, discount = self._get_setting(setting)
        # 시작 state 설정
        state = self.env.reset()
        done = False
        local_loss = []
        log_step = [[] for _ in range(4)]
        while not done:
            # Choose an action by greedly (with a chance of random action)
            # from the Q-network
            q_value = self.__cal_q_value(state, weight)

            # e-greedy, noise
            # act = np.argmax(q_value)
            act = self._get_action_noise(q_value[0], idx=idx, greedy=greedy, noise=noise)
            state_next, reward, done, _ = self.env.step(act)

            if done:
                # Update Q, and no q_value+1, since it's action termial state
                q_value[0, act] = reward
            else:
                # Obtain the Q_s` values by feeding the new state through our network
                # q_score_next = tf.matmul(self.__one_hot(state_next), weight)
                q_score_next = self.__cal_q_value(state_next, weight)
                # Update Q
                q_value[0, act] = reward + discount * np.max(q_score_next)

            # def f_loss(): tf.reduce_sum(input_tensor=tf.square(q_value
            #                                                    - tf.matmul(self.__one_hot(state), weight)))
            # print(f_loss())
            loss = lambda: tf.reduce_sum(input_tensor=tf.square(q_value
                                                                - tf.matmul(self.__one_hot(state), weight)))
            self.__optimizer.minimize(loss, var_list=weight)

            # log
            # for idx, val in enumerate([state, 0, reward, act]):
            #     log_step[idx].append(val)
            # update state
            state = state_next
        # self._log_epi.append(log_step)
        return True

    def __one_hot(self, x):
        return np.identity(self.__input_size)[x:x + 1].astype(np.float32)

    def __cal_q_value(self, state, weight):
        q_value = tf.matmul(self.__one_hot(state), weight)
        return np.array(q_value.numpy())
