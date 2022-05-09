import os
import time
import random
from collections import deque
# ------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from logistic_agent.agent.agent import Agent
from .common import one_hot


class Network:
    def __init__(self, input_size, output_size, h_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.h_size = h_size

        # W = tf.Variable(tf.initializers.GlorotUniform()([input_size, h_size]), dtype=tf.float32)
        self.w_1 = tf.Variable(tf.random.uniform([self.input_size, self.h_size], 0, 0.1), dtype=tf.float32)
        self.w_2 = tf.Variable(tf.random.uniform([self.h_size, self.output_size], 0, 0.1), dtype=tf.float32)
        self.b = tf.Variable(tf.random.uniform([self.input_size*32, self.output_size], 0, 0.1), dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # @tf.function
    def predict(self, state):
        activation = tf.nn.tanh(tf.matmul(state, self.w_1))
        q_value = tf.matmul(activation, self.w_2)
        return np.array(q_value.numpy())[0]

    # @tf.function
    def update(self, x_stack, y_stack):
        # print(tf.reduce_mean(
        #     input_tensor=(tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2)))))
        # loss = lambda: tf.reduce_mean(
        #     input_tensor=tf.square(y_stack - tf.matmul(tf.nn.relu(tf.matmul(x_stack, self.w_1)), self.w_2)))

        # dd = self.w_1.numpy().tolist()
        # ee = self.w_2.numpy().tolist()
        #
        # xx = x_stack.tolist()
        # a = tf.nn.relu(tf.matmul(x_stack, self.w_1))
        # aa = a.numpy().tolist()
        # b = tf.matmul(a, self.w_2)
        # bb = b.numpy().tolist()
        # c = tf.square(y_stack - b)
        # cc = c.numpy().tolist()
        # yy = y_stack.tolist()
        # loss = lambda: tf.reduce_mean(input_tensor=c)

        # print(tf.reduce_mean(
        #     input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2) + self.b)))
        # loss = lambda: tf.reduce_mean(
        #     input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2) + self.b))
        # self.optimizer.minimize(loss, var_list=[self.w_1, self.w_2, self.b])

        # print(tf.reduce_mean(
        #     input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2))))
        loss = lambda: tf.reduce_mean(
            input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2)))
        self.optimizer.minimize(loss, var_list=[self.w_1, self.w_2])
        # print('after')
        # print(tf.reduce_mean(
        #     input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2))))

    # @tf.function
    def l_q_map(self):
        return 1

    def get_q_map(self):
        q_map = []
        for state in range(self.input_size):
            q_map.append(self.predict(one_hot(state, self.input_size)))
        return q_map


class DQN(Agent):
    def __init__(self, env):
        super().__init__(env)
        self._name_arg = ['greedy', 'noise', 'lr_action', 'discount']
        self._init_setting = [False, False, 1, 1.0]
        self.__replay_buffer = deque()

        self.dqn_update = None
        self.dqn_target = None

    def run(self, num_episodes, max_step=None, buffer=2500, sampling=32,
            size_hidden=1, epoch=50, learning_rate=0.1, interval_train=10,
            early_stopping=False, **kwargs):
        greedy, noise, lr_action, discount = self._get_setting(kwargs)

        if early_stopping:
            early_stopping.clear()

        if not max_step:
            max_step = 2 * (self._input_size * self._output_size)
        self.__replay_buffer = deque(maxlen=buffer)

        self.dqn_update = Network(self._input_size, self._output_size, sampling)
        self.dqn_target = Network(self._input_size, self._output_size, sampling)

        # rewards per episode
        reward = []
        q_map = None
        for idx_epi in range(num_episodes):
            start_time = time.time()
            buf_reward = self._run_episodes(max_step, idx_epi=idx_epi, setting=kwargs)
            reward.append(buf_reward)
            # Reset environment and get first new observation

            if idx_epi % interval_train == 1:  # train every interval
                if len(self.__replay_buffer) >= sampling:
                    q_map = self.__train(epoch, sampling, discount)
                    # self.__replay_buffer.clear()
                    print(f'{(time.time() - start_time)} seconds')

            num_ = self._print_progress(idx_epi, num_episodes)

            if early_stopping:
                if buf_reward <= 0:
                    buf_reward = idx_epi
                if early_stopping.check_stopping(buf_reward):
                    print(f'progress = {num_} %  --> {idx_epi}/{num_episodes} Early Stopping')
        return self.dqn_target.get_q_map(), reward

    def _run_episodes(self, max_step, idx_epi=0, setting=None):
        state = self.env.reset()

        done = False
        cnt_step = 0
        reward = 0
        while not done:
            # q_value
            q_value = self.dqn_target.predict(one_hot(state, self._input_size))
            action = self._get_action_noise(q_value, idx=idx_epi, greedy=True)

            # Get new state and reward from environment
            state_next, reward, done, _ = self.env.step(action)

            if (state == state_next):
                reward = -10
                done = True
            if reward == -1:
                reward = -10
            if reward == 1:
                reward = 100

            self.__replay_buffer.append((state, action, reward, state_next, done))

            if cnt_step > max_step:
                break
            state = state_next
            cnt_step += 1
        return reward

    def __train(self, epoch, sampling, discount):
        # Get a random batch of experiences
        for _ in range(epoch):
            minibatch = random.sample(self.__replay_buffer, sampling)
            x_stack = np.empty(0, dtype=np.float32).reshape(0, self._input_size)
            y_stack = np.empty(0, dtype=np.float32).reshape(0, self._output_size)
            for state, action, reward, state_next, done in minibatch:
                # Get stored information from the buffer

                q_update = self.dqn_update.predict(one_hot(state, self._input_size))
                # debug
                q_map_update = self.dqn_update.get_q_map()

                if done:
                    q_update[action] = reward
                else:
                    q_target = self.dqn_target.predict(one_hot(state, self._input_size))

                    # original : "Q[0, action] = reward
                    # + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]"
                    q_pred = self.dqn_update.predict(one_hot(state, self._input_size))
                    q_update[action] = reward + discount * q_target[np.argmax(q_pred)]
                    # q_update[0, action] = reward + dis * np.max(q_target)
                # if (state == 0):
                #     b = 3
                # if (state == 0) and (state_next == 0) and (action == 0):
                #     b = 3
                # if (state == 0) and (state_next == 0) and (action == 3):
                #     b = 3
                q_map_update[state] = q_update
                input_state = [one_hot(idx, self._input_size) for idx in range(self._input_size)]
                # q_map_update = np.array(q_map_update)
                # input_state = np.array(input_state)

                for idx in range(self._input_size):
                    x_stack = np.vstack([x_stack, input_state[idx]])
                    y_stack = np.vstack([y_stack, q_map_update[idx]])

            # Train our network using target and predicted Q values on each episode
            self.dqn_update.update(x_stack, y_stack)

        # 일정 epi 횟수마다 q_pred의 W로 업데이트 (W2_1, W2_2 = W1_1, W1_2)
        self.dqn_target.w_1 = tf.Variable(tf.identity(self.dqn_update.w_1), dtype=tf.float32)
        self.dqn_target.w_2 = tf.Variable(tf.identity(self.dqn_update.w_2), dtype=tf.float32)
        return self.dqn_target.get_q_map()

    def test(self):
        pass

