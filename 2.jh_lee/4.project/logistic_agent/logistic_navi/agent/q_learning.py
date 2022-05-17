import numpy as np
from .agent import Agent
from env_.logistic.common import ID_GOAL


class QLearning(Agent):
    def __init__(self, env, size_input, size_output):
        super().__init__(env, size_input, size_output)
        self._name_arg = ['greedy', 'noise', 'learning_rate', 'discount']
        self._init_setting = [False, False, 0, 1.0]
        self.q_map = None

    def run(self, num_episodes, max_step=None, early_stopping=False, save_result=False, **kwargs):
        if not max_step:
            max_step = 2 * (self._size_input * self._size_output)

        result_step = []
        self.q_map = np.zeros([self.env.height*self.env.width, self.env.num_action])
        for idx_epi in range(num_episodes):
            buf_result = self._run_episodes(max_step, idx_epi=idx_epi, setting=kwargs)
            result_step.append(buf_result)

            num_ = self._print_progress(idx_epi, num_episodes)
            if self.__check_early_stopping(early_stopping, idx_epi, buf_result):
                print(f'progress = {num_} %  --> {idx_epi}/{num_episodes} Early Stopping')
                break
        return self.q_map.copy(), result_step

    # def __run_episodes(self, q_map, idx=0, greedy=False, noise=False, learning_rate=0, discount=1.):
    def _run_episodes(self, max_step, idx_epi=0, setting=None):
        greedy, noise, lr_action, discount = self._get_setting(setting)
        # 시작 state 설정
        p_cur = self.env.reset()
        state_cur = self._convert_p_to_idx(p_cur)

        done = False
        while not done:
            # e-greedy, noise
            act = self._get_action_noise(self.q_map[state_cur, :], idx=idx_epi, greedy=greedy, noise=noise)
            p_new, reward, done, result_step = self.env.step(act)
            state_new = self._convert_p_to_idx(p_new)

            if state_cur != state_new:
                if lr_action > 0:
                    self.q_map[state_cur, act] = (1 - lr_action) * self.q_map[state_cur, act] + lr_action * (
                                reward + discount * np.max(self.q_map[state_new, :]))
                else:
                    self.q_map[state_cur, act] = reward + discount * np.max(self.q_map[state_new, :])
            # update state
            state_cur = state_new
        return True

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

