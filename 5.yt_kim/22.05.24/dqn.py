import time
import random
import gc
from collections import deque
# ------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from .agent import Agent
from .common import one_hot
from common.func_ import save, load, create_dir, make_path, edit
from env_.logistic.common import ID_GOAL, INITIAL_ACTION, STR_RESULT, IDX_ACTION_UP, IDX_ACTION_DOWN


class Network:
    def __init__(self, input_size, output_size, h_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.h_size = h_size
        self.loss = 0

        # W = tf.Variable(tf.initializers.GlorotUniform()([input_size, h_size]), dtype=tf.float32)
        self.w_1 = tf.Variable(tf.random.uniform([self.input_size, self.h_size], 0, 0.1), dtype=tf.float32)
        self.w_2 = tf.Variable(tf.random.uniform([self.h_size, self.output_size], 0, 0.1), dtype=tf.float32)
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def predict(self, state):
        activation = tf.nn.tanh(tf.matmul(state, self.w_1))
        q_value = tf.matmul(activation, self.w_2)
        return np.array(q_value.numpy())[0]

    def update(self, x_stack, y_stack):
        loss = lambda: tf.reduce_mean(
            input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2)))
        self.optimizer.minimize(loss, var_list=[self.w_1, self.w_2])

    def get_q_map(self):
        q_map = []
        for state in range(self.input_size):
            q_map.append(self.predict(one_hot(state, self.input_size)))
        return q_map

    def get_loss(self, x_stack, y_stack):
        self.loss = tf.reduce_mean(
            input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2)))
        return float(self.loss)


class DQN(Agent):
    def __init__(self, env, size_input, size_output):
        super().__init__(env, size_input, size_output)
        self._name_setting = ['greedy', 'noise', 'lr_action', 'discount']
        self._init_setting = [False, False, 1, 1.0]

        self.__replay_buffer = deque()
        self.__path_dir = None
        self.__path_log = None
        # self.__path_batch = None

        self.dqn_update = None
        self.dqn_target = None

        # ksy
        self.start_time = 0
        self.lastest_reward = deque(maxlen=100)

    # -----------------------------------------------------------------------------------------------------------
    # region SAVE
    def make_dir_log(self, set_run):
        create_dir(f'./log')
        create_dir(f'./log/dqn')
        name_set = f'{set_run[0]}_{set_run[3]}_{set_run[4]}_{set_run[5]}'
        path = f'./log/dqn/{time.strftime("%y%m%d_%H%M%S")}_{name_set}'
        try:
            create_dir(path)
            self.__path_dir = path
            self.save_info(set_run)
            return True
        except:
            return False

    def save_info(self, set_run):
        name = ['num_episodes', 'max_step', 'buffer', 'sampling',
                'size_hidden', 'epoch', 'learning_rate', 'interval_train', 'greedy', 'noise', 'lr_action', 'discount']
        data_ = self.get_str_setting(name, set_run)
        save(make_path(self.__path_dir, 'info', '.csv'), data_)

    def save_result(self, set_run):
        data_ = [[time.strftime("%y%m%d_%H:%M:%S")]]
        time_taken = time.time() - self.start_time
        mins = int(time_taken / 60)
        secondes = int(time_taken % 60)
        data_.append([f'taken time : {mins}m {secondes}s'])
        name = ['num_episodes', 'max_step', 'buffer', 'sampling',
                'size_hidden', 'epoch', 'learning_rate', 'interval_train', 'greedy', 'noise', 'lr_action', 'discount']
        data_.extend(self.get_str_setting(name, set_run))
        data_.extend(self.env.get_result())
        data_.append([])
        data_.append(['q-map'])
        data_.extend(self.dqn_update.get_q_map())
        save(make_path(self.__path_dir, 'result', '.csv'), data_)

    def save_log_epi(self, data_):
        if not self.__path_log:
            self.__path_log = make_path(self.__path_dir, 'log_epi', '.csv')
        edit(self.__path_log, data_)

    #원래코드
    def save_log_batch(self, idx_epi, idx_epoch, loss_pre, loss_aft, minibatch):
        path = self.__path_dir+f'/log_batch_epi_{idx_epi}.csv'
        data_ = [[f'idx_epoch : {idx_epoch}, loss_pre : {loss_pre}, loss_aft : {loss_aft}, dif : {loss_aft-loss_pre}']]
        buf = []
        for idx, sample in enumerate(minibatch):
            buf.append(sample)
            if idx % 10 == 0:
                data_.extend(buf)
                buf.clear()
        data_.append([])
        edit(path, data_)
        
    #수정 - 임시
    # def save_log_batch(self, idx_epi, idx_epoch, loss_pre, loss_aft, minibatch):
    #     path = self.__path_dir+f'/log_batch_epi_{idx_epi}.csv'
    #     data_ = [[f'idx_epi : {idx_epi}, idx_epoch : {idx_epoch}, loss_pre : {loss_pre}, loss_aft : {loss_aft}, dif : {loss_aft-loss_pre}']]
    #     buf = []
    #     for idx, sample in enumerate(minibatch):
    #         buf.append(sample)
    #         if idx % 10 == 0:
    #             data_.extend(buf)
    #             buf.clear()
    #     data_.append([])
    #     edit(path, data_)

    #추가. 추후 지울것
    def save_buffer_log(self, idx_epi, minibatch):
        path = self.__path_dir+f'/log_buffer_epi_{idx_epi}.csv'
        data_ = [[f'idx_epi : {idx_epi}']]
        buf = []
        for idx, sample in enumerate(minibatch):
            buf.append(sample)
            data_.extend(buf)
            buf.clear()
        data_.append([])
        edit(path, data_)

    def save_log_train(self, idx_epi, loss_pre, loss_aft, q_map):
        path = self.__path_dir+f'/log_train_epi_{idx_epi}.csv'
        data_ = [[f'loss_pre : {loss_pre}, loss_aft : {loss_aft}, dif : {loss_aft-loss_pre}']]
        data_.extend(q_map)
        edit(path, data_)

    def get_str_setting(self, name, setting):
        data_ = []
        for idx in range(len(name)):
            data_.append([f'{name[idx]} : {setting[idx]}'])
        return data_

    # endregion SAVE

    # -----------------------------------------------------------------------------------------------------------
    def run(self, num_episodes, max_step=None, buffer=2500, sampling=32,
            size_hidden=1, epoch=50, learning_rate=0.1, interval_train=10,
            early_stopping=False, save_result=False, run_time=2000, **kwargs):
        greedy, noise, lr_action, discount = self._get_setting(kwargs)
        max_step = self.__init_run(max_step, size_hidden, buffer)
        set_run = [num_episodes, max_step, buffer, sampling,
                   size_hidden, epoch, learning_rate, interval_train, greedy, noise, lr_action, discount]
        self.make_dir_log(set_run)

        # debug
        self.start_time = time.time()

        result_step = []
        for idx_epi in range(num_episodes):
            start_time = time.time()
            buf_result = self._run_episodes(max_step, idx_epi=idx_epi, setting=[greedy, noise])
            result_step.append(buf_result)

            # train every interval
            if idx_epi % interval_train == 0:
                #용석님이 권유한 코드
                # if len(self.__replay_buffer) >= run_time:
                
                #원래코드
                if len(self.__replay_buffer) >= buffer:
                    
                # 리플레이버퍼용
                #if len(self.__replay_buffer) >= sampling:
                    loss_pre, loss_aft = self.__train(idx_epi, epoch, sampling, discount)
                    self.save_log_train(idx_epi, loss_pre, loss_aft, self.dqn_update.get_q_map())

                    print(f'{(time.time() - start_time)} seconds')
                    #리플레이버퍼용
                    #self.__replay_buffer.clear()
            num_ = self._print_progress(idx_epi, num_episodes)

            if not self.__check_early_stopping(early_stopping, idx_epi, buf_result):
                print(f'progress = {num_} %  --> {idx_epi}/{num_episodes} Early Stopping')
                break

        if save_result:
            self.save_result(set_run)
        gc.collect()
        return self.dqn_target.get_q_map(), result_step

    # -----------------------------------------------------------------------------------------------------------
    def __init_run(self, max_step, size_hidden, buffer):
        if not max_step:
            max_step = 2 * (self._size_input * self._size_output)
        self.__replay_buffer = deque(maxlen=buffer)

        self.dqn_update = Network(self._size_input, self._size_output, size_hidden)
        self.dqn_target = Network(self._size_input, self._size_output, size_hidden)

        self.dqn_target.w_1 = tf.Variable(tf.identity(self.dqn_update.w_1), dtype=tf.float32)
        self.dqn_target.w_2 = tf.Variable(tf.identity(self.dqn_update.w_2), dtype=tf.float32)
        return max_step

    def _run_episodes(self, max_step, idx_epi=0, setting=None):
        greedy, noise = setting
        log_ = [[f'idx_epi : {idx_epi}'],
                ['time', 'action', 'p_cur', 'p_new', 'state_cur', 'state_new', 'reward', 'done', 'result_step']]

        # env 초기화, 시작 state_cur 설정
        p_cur = self.env.reset()
        state_cur = self._convert_p_to_idx(p_cur)

        done = False
        cnt_step = 0
        result_step = None

        while not done:
            q_value = self.dqn_update.predict(self._one_hot(state_cur))
            action = 0
            # 첫스텝은 action 고정
            if cnt_step == 0:
                action = IDX_ACTION_UP
            else:
                action = self._get_action_noise(q_value, idx_epi=idx_epi, greedy=greedy, noise=noise)

            # Get new state_cur and reward from environment
            p_new, reward, done, result_step = self.env.step(action)
            state_new = self._convert_p_to_idx(p_new)

            self.__replay_buffer.append((state_cur, action, reward, state_new, done))

            #버퍼 들어오는 크기 확인
            print(len(self.__replay_buffer))

            log_.append([time.strftime("%y%m%d_%H%M%S"), INITIAL_ACTION[action],
                         p_cur, p_new, state_cur, state_new,
                         reward, done, STR_RESULT[result_step]])

            if done:
                self._decay_epsilon(greedy)
                # debug
                elapsed_time = time.time() - self.start_time
                if reward == 10:
                    self.lastest_reward.append(1)
                else:
                    self.lastest_reward.append(0)
                lastest_score = sum(self.lastest_reward)

                #추가코드
                self.save_buffer_log(idx_epi, self.__replay_buffer)

                self.__report(idx_epi, cnt_step, state_cur, elapsed_time, lastest_score, reward)

            p_cur = p_new
            state_cur = state_new
            cnt_step += 1
            if cnt_step > max_step:
                break
        log_.append([])
        self.save_log_epi(log_)
        del log_
        return result_step

    # -----------------------------------------------------------------------------------------------------------
    def __train(self, idx_epi, epoch, sampling, discount):
        # Get a random batch of experiences
        x_stack = None
        y_stack = None
        loss_pre = None
        loss_aft = None
        for idx_epoch in range(epoch):
            minibatch = random.sample(self.__replay_buffer, sampling)
            x_stack, y_stack = self.__make_target(minibatch, discount)
            # Train our network using target and predicted Q values on each episode
            loss_pre = float(self.dqn_update.get_loss(x_stack, y_stack))
            self.dqn_update.update(x_stack, y_stack)
            loss_aft = float(self.dqn_update.get_loss(x_stack, y_stack))
            self.save_log_batch(idx_epi, idx_epoch, loss_pre, loss_aft, minibatch)
        print('[LOG] epi:', idx_epi, 'loss:', loss_pre, '=>', loss_aft)

        # 일정 epi 횟수마다 q_pred의 W로 업데이트 (W2_1, W2_2 = W1_1, W1_2)
        loss_pre = float(self.dqn_target.get_loss(x_stack, y_stack))
        
        #용석님 시도했을때 잘 됐던 것
        if idx_epi % 1000 == 0:
             self.dqn_target.w_1 = tf.Variable(tf.identity(self.dqn_update.w_1), dtype=tf.float32)
             self.dqn_target.w_2 = tf.Variable(tf.identity(self.dqn_update.w_2), dtype=tf.float32)

        loss_aft = float(self.dqn_target.get_loss(x_stack, y_stack))

        #원본
        # self.dqn_target.w_1 = tf.Variable(tf.identity(self.dqn_update.w_1), dtype=tf.float32)
        # self.dqn_target.w_2 = tf.Variable(tf.identity(self.dqn_update.w_2), dtype=tf.float32)
        # loss_aft = float(self.dqn_target.get_loss(x_stack, y_stack))
        return loss_pre, loss_aft

    def __make_target(self, minibatch, discount):
        x_stack = np.empty(0, dtype=np.float32).reshape(0, self._size_input)
        y_stack = np.empty(0, dtype=np.float32).reshape(0, self._size_output)
        for state_cur, action, reward, state_next, done in minibatch:
            # Get stored information from the buffer
            q_update = self.dqn_update.predict(self._one_hot(state_cur))
            # debug
            q_map_update = self.dqn_update.get_q_map()

            if done:
                q_update[action] = reward
            else:
                q_target = self.dqn_target.predict(self._one_hot(state_next))
                q_update[action] = reward + discount * np.max(q_target)

            # Q map 전체 optimize
            q_map_update[state_cur] = q_update
            input_state = [self._one_hot(idx) for idx in range(self._size_input)]
            for idx in range(self._size_input):
                x_stack = np.vstack([x_stack, input_state[idx]])
                y_stack = np.vstack([y_stack, q_map_update[idx]])

            # Q value optimize
            #x_stack = np.vstack([x_stack, self._one_hot(state_cur)])
            #y_stack = np.vstack([y_stack, q_update])
        return x_stack, y_stack

    def __check_early_stopping(self, early_stopping, idx_epi, buf_result):
        if early_stopping:
            if idx_epi == 0:
                early_stopping.clear()
            if not self.dqn_update.loss == 0:
                flg = idx_epi
                if buf_result == ID_GOAL:
                    flg = True
                # if early_stopping.check_stopping(round(float(self.dqn_update.loss), 5)):
                if early_stopping.check_stopping(flg):
                    return False
        return True

    def __report(self, episode, steps, state_cur, elapsed_time, lastest_score, reward):
        mins = int(elapsed_time / 60)
        secondes = int(elapsed_time % 60)
        colour = '\033[92m' if reward > 0 else '\033[91m'
        print("episode: " + str(episode).rjust(4)
              #+ ' ε: {:.3f}'.format(self.dqn_update.e)
              + " steps: " + str(steps).rjust(3)
              + " state_cur: [" + str(state_cur).rjust(2) + "]"
              + ' time: {:02d}'.format(mins)
              + ':{:02d}'.format(secondes)
              + " score: " + str(lastest_score).rjust(2)
              + ' memory:' + str(len(self.__replay_buffer)).rjust(4)
              + f' reward: {colour}' + f"{reward:+.1f}" + '\033[0m')
