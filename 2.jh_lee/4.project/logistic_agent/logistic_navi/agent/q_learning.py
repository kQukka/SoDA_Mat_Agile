import time
import numpy as np
from .agent import Agent
from common.func_ import save, load, create_dir, make_path, edit
from env_.logistic.common import ID_GOAL, INITIAL_ACTION, STR_RESULT, IDX_ACTION_UP


class QLearning(Agent):
    def __init__(self, env, size_input, size_output):
        super().__init__(env, size_input, size_output)
        self._name_setting = ['greedy', 'noise', 'learning_rate', 'discount']
        self._init_setting = [0, False, 0, 1.0]
        self.sum_step = 0
        self.q_map = None
        self.path_result = None

        self.__path_dir = None
        self.__path_log = None

    def make_dir_log(self, set_run):
        self.__path_dir = None

        create_dir(f'./log')
        create_dir(f'./log/q_learning')
        name_set = f'{set_run[0]}_{set_run[2]}_{set_run[3]}_{set_run[1]}'
        path = make_path(f'./log/q_learning', f'{time.strftime("%y%m%d_%H%M%S")}_{name_set}', '')
        try:
            create_dir(path)
            self.__path_dir = path
            self.save_info(set_run)
            self.__path_log = make_path(self.__path_dir, 'log_epi', '.csv')
            return True
        except Exception as e:
            return False

    def save_info(self, set_run):
        name = ['num_episodes', 'max_step', 'greedy', 'noise', 'lr_action', 'discount']
        data_ = self.get_str_setting(name, set_run)
        save(make_path(self.__path_dir, 'info', '.csv'), data_)

    def save_result(self, set_run):
        data_ = [[time.strftime("%y%m%d_%H:%M:%S")]]
        name = ['num_episodes', 'max_step', 'greedy', 'noise', 'lr_action', 'discount', 'end_epi', 'step', 'score']
        data_.extend(self.get_str_setting(name, set_run))
        data_.extend(self.env.get_result())
        data_.append([])
        data_.append(['q-map'])
        data_.extend(self.q_map.copy())
        save(make_path(self.__path_dir, 'result', '.csv'), data_)

        data_result = set_run.copy()
        result_env = self.env.get_result()
        data_result.append(result_env[-1])
        edit(self.path_result, [data_result])

    def save_log_epi(self, data_):
        edit(self.__path_log, data_)

    def get_str_setting(self, name, setting):
        data_ = []
        for idx in range(len(name)):
            data_.append([f'{name[idx]} : {setting[idx]}'])
        return data_

    # -----------------------------------------------------------------------------------------------------------
    def run(self, num_episodes, max_step=None, early_stopping=False, save_result=False, **kwargs):
        if not max_step:
            max_step = 2 * (self._size_input * self._size_output)
        self.sum_step = 0
        greedy, noise, lr_action, discount = self._get_setting(kwargs)
        set_run = [num_episodes, max_step, greedy, noise, lr_action, discount]
        if not self.make_dir_log(set_run):
            return False

        result_step = []
        self.q_map = np.zeros([self.env.height*self.env.width, self.env.num_action])
        idx_epi = 0
        for idx_epi in range(num_episodes):
            buf_result = self._run_episodes(max_step, idx_epi=idx_epi, setting=kwargs)
            result_step.append(buf_result)

            num_ = self._print_progress(idx_epi, num_episodes)
            if not self.__check_early_stopping(early_stopping, idx_epi, buf_result):
                print(f'progress = {num_} %  --> {idx_epi}/{num_episodes} Early Stopping : {self.sum_step/idx_epi}')
                break
        if save_result:
            set_run.append(idx_epi)
            set_run.append(self.sum_step)
            set_run.append(self.sum_step/idx_epi)
            self.save_result(set_run)
        return self.q_map.copy(), result_step

    # def __run_episodes(self, q_map, idx=0, greedy=False, noise=False, learning_rate=0, discount=1.):
    def _run_episodes(self, max_step, idx_epi=0, setting=None):
        greedy, noise, lr_action, discount = self._get_setting(setting)
        log_ = [[f'idx_epi : {idx_epi}'],
                ['time', 'action', 'p_cur', 'p_new', 'state_cur', 'state_new', 'reward', 'done', 'result_step']]
        # 시작 state 설정
        p_cur = self.env.reset()
        state_cur = self._convert_p_to_idx(p_cur)
        done = False
        result_step = False
        while not done:
            self.sum_step += 1
            # e-greedy, noise
            action = self._get_action_noise(self.q_map[state_cur, :], idx_epi=idx_epi, greedy=greedy, noise=noise)
            p_new, reward, done, result_step = self.env.step(action)
            state_new = self._convert_p_to_idx(p_new)

            if done:
                self.q_map[state_cur, action] = reward
            else:
                if state_cur != state_new:
                    if lr_action > 0:
                        self.q_map[state_cur, action] = \
                            (1 - lr_action) * self.q_map[state_cur, action] + \
                            lr_action * (reward + discount * np.max(self.q_map[state_new, :]))
                    else:
                        self.q_map[state_cur, action] = reward + discount * np.max(self.q_map[state_new, :])
            log_.append([time.strftime("%y%m%d_%H%M%S"), INITIAL_ACTION[action],
                         p_cur, p_new, state_cur, state_new,
                         reward, done, STR_RESULT[result_step]])

            # update state
            p_cur = p_new
            state_cur = state_new
        log_.append([])
        # self.save_log_epi(log_)
        del log_
        return result_step

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

