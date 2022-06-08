import sys
import gym
import time
import keras.initializers.initializers_v2
import pylab
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers.initializers_v2 import RandomUniform
# -----------------------------------------------------------------------------------------------------------
from .agent import Agent
from common.func_ import save, load, create_dir, make_path, edit
from env_.logistic.common import INITIAL_ACTION, STR_RESULT, IDX_ACTION_UP, IDX_ACTION_DOWN
from env_.logistic.common import ID_GOAL, ID_OBSTACLE, ID_OUT_GRID, ID_GENERAL_MOVE


size_network_policy = 24
size_network_val = 24


# 정책 신경망과 가치 신경망 생성
class Network(tf.keras.Model):
    def __init__(self, action_size):
        super(Network, self).__init__()
        # police network
        self.actor_fc = Dense(size_network_policy, activation='tanh')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=RandomUniform(-1e-3, 1e-3))
        # value network
        self.critic_fc1 = Dense(size_network_val, activation='tanh')
        self.critic_fc2 = Dense(size_network_val, activation='tanh')
        self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    # def call(self, x):
    #     actor_x = self.actor_fc(x)
    #     policy = self.actor_out(actor_x)
    #
    #     critic_x = self.critic_fc1(x)
    #     critic_x = self.critic_fc2(critic_x)
    #     value = self.critic_out(critic_x)
    #     return policy, value

    def get_policy(self, x):
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x)
        return policy

    def get_value(self, x):
        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2C(Agent):
    def __init__(self, env, size_input, size_output):
        super(A2C, self).__init__(env, size_input, size_output)
        self._name_setting = ['greedy', 'noise', 'lr_action', 'discount']
        self._init_setting = [False, False, 1, 0.99]

        self.start_time = None
        self.__path_dir = None
        self.__path_log = None

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.sum_step = 0

        # 정책신경망과 가치신경망 생성
        self.model = Network(self._size_output)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=5.0)

    # -----------------------------------------------------------------------------------------------------------
    # region SAVE
    def make_dir_log(self, set_run, name):
        create_dir(f'./log')
        create_dir(f'./log/A2C')
        name_set = f'{set_run[0]}_{set_run[1]}_{set_run[3]}_{set_run[4]}'
        path = f'./log/A2C/{time.strftime("%y%m%d_%H%M%S")}_{name_set}'
        try:
            create_dir(path)
            self.__path_dir = path
            self.save_info(set_run, name)
            return True
        except:
            return False

    def save_info(self, set_run, name):
        data_ = self.get_str_setting(name, set_run)
        data_.extend(self.env.get_result())
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

    # endregion SAVE

    def run(self, num_episodes, max_step=None,
            epoch=1, learning_rate=0.001,
            early_stopping=False, save_result=False, **kwargs):
        greedy, noise, lr_action, discount = self._get_setting(kwargs)

        self.sum_step = 0
        self.discount_factor = discount
        self.learning_rate = learning_rate
        if not max_step:
            max_step = self._size_input

        name = ['num_episodes', 'max_step', 'epoch', 'learning_rate', 'discount']
        set_run = [num_episodes, max_step, epoch, learning_rate, discount]
        self.make_dir_log(set_run, name)

        # debug
        self.start_time = time.time()
        for idx_epi in range(num_episodes):
            start_time = time.time()
            self._run_episodes(max_step, idx_epi, epoch)
            # print(f'{(time.time() - start_time)} seconds')
        return True, True

    def _run_episodes(self, max_step, idx_epi=0, epoch=1, **kwargs):
        log_ = [[f'idx_epi : {idx_epi}'],
                ['cnt_step', 'time', 'action', 'p_cur', 'p_new', 'state_cur', 'state_new',
                 'reward', 'done', 'result_step', 'loss']]

        p_cur = self.env.reset()
        state_cur = self._convert_p_to_idx(p_cur)

        log_step, result_step = self.__run_loop_step(idx_epi, p_cur, state_cur, max_step, epoch)

        # done = False
        # cnt_step = 0
        # result_step = None
        # score_400 = 0
        # time_400 = 0
        # while not done:
        #     self.sum_step += 1
        #     action = self.__get_action(self._one_hot(state_cur))
        #     if cnt_step == 0:
        #         action = 3
        #
        #     p_new, reward, done, result_step = self.env.step(action)
        #     state_new = self._convert_p_to_idx(p_new)
        #
        #     reward, done = self.__check_reward(cnt_step, max_step, result_step, done, (2 / 3))
        #     loss = self.__train_done(epoch, result_step, state_cur, action, reward, state_new, done)
        #
        #     # debug
        #     time_400, score_400 = self.__print_log(done, idx_epi, loss, result_step, cnt_step, time_400, score_400)
        #     log_.append([cnt_step, time.strftime("%y%m%d_%H%M%S"), INITIAL_ACTION[action],
        #                  p_cur, p_new, state_cur, state_new,
        #                  reward, done, STR_RESULT[result_step], loss])
        #     p_cur = p_new
        #     state_cur = state_new
        #     cnt_step += 1
        # log_.append(['score_400 : ', score_400, 'time-400 : ', time_400,
        #              'score : ', round(self.sum_step / (idx_epi+1), 3),
        #              'time : ', round(time.time() - self.start_time, 3)])
        log_.extend(log_step)
        log_.append([])
        self.save_log_epi(log_)
        del log_
        del log_step
        if idx_epi % 100 == 0:
            self.model.save_weights(f"./save_model/model_{idx_epi}", save_format="tf")
        return result_step

    def __run_loop_step(self, idx_epi, p_cur, state_cur, max_step, epoch):
        log_ = []
        num_match = 5
        buffer_state = []
        done = False
        cnt_step = 0
        result_step = None
        score_400 = 0
        time_400 = 0
        while not done:
            self.sum_step += 1
            action = self.__get_action(self._one_hot(state_cur))
            if cnt_step == 0:
                action = 3

            action = self.__check_action(action, num_match, buffer_state, state_cur)
            p_new, reward, done, result_step = self.env.step(action)
            state_new = self._convert_p_to_idx(p_new)

            reward, done = self.__check_reward(reward, cnt_step, max_step, result_step, done, 2/3)
            loss = self.__train_done(epoch, result_step, state_cur, action, reward, state_new, done)

            # debug
            time_400, score_400 = self.__print_log(done, idx_epi, loss, result_step, cnt_step, time_400, score_400)
            log_.append([cnt_step, time.strftime("%y%m%d_%H%M%S"), INITIAL_ACTION[action],
                         p_cur, p_new, state_cur, state_new,
                         reward, done, STR_RESULT[result_step], loss])

            # update p, state, step
            p_cur = p_new
            state_cur = state_new
            cnt_step += 1
        log_.append(['score_400 : ', score_400, 'time-400 : ', time_400,
                     'score : ', round(self.sum_step / (idx_epi+1), 3),
                     'time : ', round(time.time() - self.start_time, 3)])
        return log_, result_step

    def __check_action(self, action, num_match, buffer_state, state_cur):
        p_new, reward, done, result_step = self.env.pred_step(action)
        state_new = self._convert_p_to_idx(p_new)
        if self.__check_loop_state(num_match, buffer_state, state_new):
            buf_ = action
            action = self.__get_action(self._one_hot(state_cur), idx_ignore=action)
            print(f'state_cur : {state_cur}, buf_ : {buf_}, action : {action}')
        return action

    def __check_loop_state(self, num_match, buffer_state, state_new):
        num_buf = 2 * num_match
        buffer_state.append(state_new)
        if len(buffer_state) > num_buf:
            del buffer_state[0]
        len_buf = len(buffer_state)
        if len_buf == num_buf:
            state_1 = buffer_state[-2]
            state_2 = buffer_state[-1]
            cnt_match = 0
            for idx in range(num_match):
                if state_1 == buffer_state[2*idx]:
                    cnt_match += 1
                if state_2 == buffer_state[1+(2*idx)]:
                    cnt_match += 1
            if cnt_match == len_buf:
                buffer_state.clear()
                return True
        return False

    def __print_log(self, done, idx_epi, loss, result_step, cnt_step, time_400, score_400):
        if done:
            print(f'[LOG] epi: {idx_epi}, loss: {loss}, '
                  f'result_step : {STR_RESULT[result_step]}, cnt_step : {cnt_step}, '
                  f'score : {self.sum_step / (idx_epi + 1)}')
        if idx_epi == 400:
            time_400 = round(time.time() - self.start_time, 3)
            score_400 = round(self.sum_step / (idx_epi + 1), 3)
        return time_400, score_400

    def __check_reward(self, reward, cnt_step, max_step, result_step, done, rate):
        # check over step
        if cnt_step >= max_step:
            reward = self.env.REWARD.NOT_MOVE
            done = True
        # check reference to apply reward(move_sub)
        elif result_step == ID_GENERAL_MOVE:
            if cnt_step >= (max_step * rate):
                reward = self.env.REWARD.MOVE_SUB
        return reward, done

    def __train_done(self, epoch, result_step, state_cur, action, reward, state_new, done):
        loss = 0
        num_loop = epoch
        if done:
            if (result_step == ID_GOAL) or (result_step == ID_OBSTACLE):
                num_loop = epoch * 10
            for _ in range(num_loop):
                loss = self.__train(state_cur, action, reward, state_new, done)
                if loss < 1:
                    break
        else:
            loss = self.__train(state_cur, action, reward, state_new, done)
        return loss

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def __train(self, state_cur, action, reward, state_new, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            policy = self.model.get_policy(self._one_hot(state_cur))
            value = self.model.get_value(self._one_hot(state_cur))
            next_value = self.model.get_value(self._one_hot(state_new))
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            one_hot_action = tf.one_hot([action], self._size_output)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            advantage = tf.stop_gradient(target - value[0])
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def __get_action(self, state, idx_ignore=None):
        if idx_ignore is not None:
            # policy = self.model.get_policy(state)
            # policy = np.array(policy[0])
            if idx_ignore < self._size_output:
                # a = policy[idx_ignore]
                # b = round(a/(self._size_output-1), 7)
                # c = round(a-(b*(self._size_output-2)), 7)
                # d = [b for _ in range(self._size_output-2)]
                # d.append(c)
                # cnt = 0
                # for idx in range(self._size_output):
                #     if idx == idx_ignore:
                #         policy[idx] = 0
                #         continue
                #     policy[idx] += d[cnt]
                #     cnt += 1
                # # action = np.random.choice(self._size_output, 1, p=policy)[0]
                action = np.random.choice(self._size_output, 1)[0]
                return action
        policy = self.model.get_policy(state)
        policy = np.array(policy[0])
        return np.random.choice(self._size_output, 1, p=policy)[0]

    # -----------------------------------------------------------------------------------------------------------
    def __check_early_stopping(self, early_stopping, idx_epi, buf_result):
        # if early_stopping:
        #     if idx_epi == 0:
        #         early_stopping.clear()
        #     if not self.dqn_update.loss == 0:
        #         flg = idx_epi
        #         if buf_result == ID_GOAL:
        #             flg = True
        #         # if early_stopping.check_stopping(round(float(self.dqn_update.loss), 5)):
        #         if early_stopping.check_stopping(flg):
        #             return False
        return True


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    #
    # # 액터-크리틱(A2C) 에이전트 생성
    # agent = A2C(action_size)
    #
    # scores, episodes = [], []
    # score_avg = 0
    #
    # num_episode = 1000
    # for e in range(num_episode):
    #     done = False
    #     score = 0
    #     loss_list = []
    #     state = env.reset()
    #     state = np.reshape(state, [1, state_size])
    #
    #     while not done:
    #         if agent.render:
    #             env.render()
    #
    #         action = agent.get_action(state)
    #         next_state, reward, done, info = env.step(action)
    #         next_state = np.reshape(next_state, [1, state_size])
    #
    #         # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
    #         score += reward
    #         reward = 0.1 if not done or score == 500 else -1
    #
    #         # 매 타임스텝마다 학습
    #         loss = agent.train_model(state, action, reward, next_state, done)
    #         loss_list.append(loss)
    #         state = next_state
    #
    #         if done:
    #             # 에피소드마다 학습 결과 출력
    #             score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
    #             print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f}".format(
    #                   e, score_avg, np.mean(loss_list)))
    #
    #             # 에피소드마다 학습 결과 그래프로 저장
    #             scores.append(score_avg)
    #             episodes.append(e)
    #             pylab.plot(episodes, scores, 'b')
    #             pylab.xlabel("episode")
    #             pylab.ylabel("average score")
    #             pylab.savefig("./save_graph/graph.png")
    #
    #             # 이동 평균이 400 이상일 때 종료
    #             if score_avg > 400:
    #                 agent.model.save_weights("./save_model/model", save_format="tf")
    #                 sys.exit()