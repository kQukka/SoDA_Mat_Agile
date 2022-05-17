import os
import time
# ------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from .agent import Agent
from common.func_ import save, load, create_dir, make_path, edit
from env_.logistic.common import ID_GOAL, INITIAL_ACTION, STR_RESULT


class QNetwork(Agent):
    def __init__(self, env, size_input, size_output):
        super().__init__(env, size_input, size_output)
        self._name_arg = ['greedy', 'noise', 'lr_action', 'discount']
        self._init_setting = [False, False, 1, 1.0]
        self.__optimizer = None
        self.__weight = None
        self.__path_dir = None
        self.__path_log = None

    def make_dir_log(self, set_run):
        create_dir(f'./log')
        create_dir(f'./log/q_net')
        name_set = f'{set_run[0]}_{set_run[2]}_{set_run[3]}_{set_run[4]}_{set_run[6]}'
        path = f'./log/q_net/{time.strftime("%y%m%d_%H%M%S")}_{name_set}'
        try:
            create_dir(path)
            self.__path_dir = path
            self.save_info(set_run)
            return True
        except:
            return False

    def save_info(self, set_run):
        name = ['num_episodes', 'max_step', 'learning_rate', 'greedy', 'noise', 'lr_action', 'discount']
        data_ = self.get_str_setting(name, set_run)
        save(make_path(self.__path_dir, 'info', '.csv'), data_)

    def save_log_epi(self, data_):
        if not self.__path_log:
            self.__path_log = make_path(self.__path_dir, 'log_epi', '.csv')
        edit(self.__path_log, data_)

    def get_str_setting(self, name, setting):
        data_ = []
        for idx in range(len(name)):
            data_.append([f'{name[idx]} : {setting[idx]}'])
        return data_

    def get_weight(self):
        return np.array(self.__weight.numpy()).tolist()

    def run(self, num_episodes, max_step=None, learning_rate=0.1,
            early_stopping=False, save_result=False, **kwargs):
        if not max_step:
            max_step = 2 * (self._size_input * self._size_output)

        greedy, noise, lr_action, discount = self._get_setting(kwargs)
        set_run = [num_episodes, max_step, learning_rate, greedy, noise, lr_action, discount]
        self.make_dir_log(set_run)

        # weight
        weight_init = tf.convert_to_tensor(np.array(0), dtype=tf.float32)
        self.__weight = tf.Variable(weight_init, dtype=tf.float32)
        # optimizer
        self.__optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

        start_time = time.time()
        result_step = []
        for idx_epi in range(num_episodes):
            buf_result = self._run_episodes(max_step, idx_epi=idx_epi, setting=kwargs)

            num_ = self._print_progress(idx_epi, num_episodes)
            if self.__check_early_stopping(early_stopping, idx_epi, buf_result):
                print(f'progress = {num_} %  --> {idx_epi}/{num_episodes} Early Stopping')
                break

        print(f'{(time.time() - start_time)} seconds')
        return self.get_weight(), result_step

    # def __run_episodes(self, q_map, idx=0, greedy=False, noise=False, lr_action=0, discount=1.):
    def _run_episodes(self, max_step, idx_epi=0, setting=None):
        log_ = [[f'idx_epi : {idx_epi}'],
                ['time', 'action', 'p_cur', 'p_new', 'state', 'state_new', 'reward', 'done', 'result_step']]
        greedy, noise, lr_action, discount = self._get_setting(setting)

        # env 초기화, 시작 state 설정
        p_cur = self.env.reset()
        state_cur = self._convert_p_to_idx(p_cur)

        done = False
        result_step = None
        cnt_step = 0
        while not done:

            loss_pre = tf.reduce_sum(
                input_tensor=tf.square(q_value - tf.matmul(self._one_hot(state_cur), self.__weight)))
            self.loss_minimize(q_value, state_cur)
            loss_aft = tf.reduce_sum(
                input_tensor=tf.square(q_value - tf.matmul(self._one_hot(state_cur), self.__weight)))

            log_.append([time.strftime("%y%m%d_%H%M%S"), INITIAL_ACTION[action],
                         p_cur, p_new, state_cur, state_new,
                         reward, done, STR_RESULT[result_step],
                         loss_pre, loss_aft])

            state_cur = state_new
            cnt_step += 1
            if cnt_step > max_step:
                break
        self.save_log_epi(log_)
        del log_
        return result_step

    def __train(self, state_cur, idx_epi, greedy, noise, discount):
        q_value = self.__get_q_value(state_cur)

        # apply e-greedy, noise
        action = self._get_action_noise(q_value[0], idx=idx_epi, greedy=greedy, noise=noise)
        p_new, reward, done, result_step = self.env.step(action)
        state_new = self._convert_p_to_idx(p_new)

        if done:
            # update q-value
            q_value[0, action] = reward
        else:
            # next state q_value
            q_value_next = self.__get_q_value(state_new)
            # update q-value
            q_value[0, action] = reward + discount * np.max(q_value_next)
        return


    def loss_minimize(self, q_value, state_cur):
        loss = lambda: tf.reduce_sum(
            input_tensor=tf.square(q_value - tf.matmul(self._one_hot(state_cur), self.__weight)))
        self.__optimizer.minimize(loss, var_list=self.__weight)

    def __get_q_value(self, state):
        q_value = tf.matmul(self._one_hot(state), self.__weight)
        return np.array(q_value.numpy())

    def __check_early_stopping(self, early_stopping, idx_epi, buf_result):
        if early_stopping:
            if idx_epi == 0:
                early_stopping.clear()
            flg = idx_epi
            if buf_result == ID_GOAL:
                flg = True
            if early_stopping.check_stopping(flg):
                return False
        return True

