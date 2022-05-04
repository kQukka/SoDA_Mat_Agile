import numpy as np

from logistic_agent.agent.agent import Agent


class QLearning(Agent):
    def __init__(self, env):
        super().__init__(env)
        self._name_arg = ['greedy', 'noise', 'learning_rate', 'discount']
        self._init_setting = [False, False, 0, 1.0]

    def run(self, num_episodes, q_map=None, early_stopping=False, **kwargs):
        del self._log_epi
        self._log_epi = []
        if early_stopping:
            early_stopping.clear()

        q_map = self._load_map(q_map)
        for idx in range(num_episodes):
            self._run_episodes(q_map, setting=kwargs)

            num_ = self._print_progress(idx, num_episodes)
            if self._check_early_stopping(early_stopping):
                print(f'progress = {num_} %  --> {idx}/{num_episodes} Early Stopping')
                break
        sum_reward_by_epi = self._get_log_sum_reward()
        return q_map, sum_reward_by_epi

    # def __run_episodes(self, q_map, idx=0, greedy=False, noise=False, learning_rate=0, discount=1.):
    def _run_episodes(self, q_map, idx=0, setting=None):
        greedy, noise, learning_rate, discount = self._get_setting(setting)
        # 시작 state 설정
        state = self.env.reset()
        done = False
        log_step = [[] for _ in range(4)]
        while not done:
            # e-greedy, noise
            act = self._get_action_noise(q_map[state, :], idx=idx, greedy=greedy, noise=noise)
            state_next, reward, done, _ = self.env.step(act)

            if state != state_next:
                if learning_rate > 0:
                    q_map[state, act] = (1 - learning_rate) * q_map[state, act] + learning_rate * (
                                reward + discount * np.max(q_map[state_next, :]))
                else:
                    q_map[state, act] = reward + discount * np.max(q_map[state_next, :])
            # log
            for idx, val in enumerate([state, 0, reward, act]):
                log_step[idx].append(val)
            # update state
            state = state_next
        self._log_epi.append(log_step)
        return True

