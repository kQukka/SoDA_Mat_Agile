import os
import time
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Set Q-learning parameters
num_episodes = 5000
learning_rate = 0.1
dis = 0.9
REPLAY_MEMORY = 2500
MAX_STEP = 10000
NUM_SAMPLING = 100
NUM_EPOCH = 50


def one_hot(x):
    return np.identity(16)[x:x + 1].astype(np.float32)


def get_q_map(dqn):
    q_map = []
    for state in range(dqn.input_size):
        q_map.append(dqn.predict(state))
    return q_map


class DQN:
    def __init__(self, input_size, output_size, h_size):
        self.input_size = input_size
        self.output_size = output_size
        self.h_size = h_size

        # W = tf.Variable(tf.initializers.GlorotUniform()([input_size, h_size]), dtype=tf.float32)
        self.w_1 = tf.Variable(tf.random.uniform([self.input_size, self.h_size], 0, 0.01), dtype=tf.float32)
        self.w_2 = tf.Variable(tf.random.uniform([self.h_size, self.output_size], 0, 0.01), dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # @tf.function
    def predict(self, state):
        activation = tf.nn.tanh(tf.matmul(one_hot(state), self.w_1))
        q_value = tf.matmul(activation, self.w_2)
        return np.array(q_value.numpy())

    # @tf.function
    def update(self, x_stack, y_stack):
        print(tf.reduce_mean(
            input_tensor=(tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2)))))
        loss = lambda: tf.reduce_mean(
            input_tensor=tf.square(y_stack - tf.matmul(tf.nn.tanh(tf.matmul(x_stack, self.w_1)), self.w_2)))
        self.optimizer.minimize(loss, var_list=[self.w_1, self.w_2])


def main():
    # Input and output size based on the Env
    register(
        id='LakeEnv-v1',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
    )
    env = gym.make('LakeEnv-v1')

    input_size = env.observation_space.n
    output_size = env.action_space.n
    h_size = NUM_SAMPLING

    dqn_update = DQN(input_size, output_size, h_size)
    dqn_target = DQN(input_size, output_size, h_size)


    # rewards per episode
    rList = []
    replay_buffer = deque()

    start_time = time.time()
    for idx_epi in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        rAll = 0
        done = False
        cnt_step = 0
        e = 1. / ((idx_epi / 10) + 1)

        # The Q-Table learning algorithm
        while not done:
            # q_value
            q_value = dqn_target.predict(state)

            # Choose an action by greedly (with a chance of random action)
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_value)

            # Get new state and reward from environment
            state_next, reward, done, _ = env.step(action)

            # Penalty
            if done:
                rAll += reward
            replay_buffer.append((state, action, reward, state_next, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()
            state = state_next

            cnt_step += 1
        rList.append(rAll)

        print(f"Episode: {idx_epi} steps: {cnt_step}")
        if len(replay_buffer) < NUM_SAMPLING:
            continue
            # break

        if idx_epi % 10 == 1:  # train every 10 episode
            # Get a random batch of experiences
            for _ in range(50):
                minibatch = random.sample(replay_buffer, NUM_SAMPLING)
                # Get stored information from the buffer
                x_stack = np.empty(0, dtype=np.float32).reshape(0, input_size)
                y_stack = np.empty(0, dtype=np.float32).reshape(0, output_size)
                for state, action, reward, state_next, done in minibatch:
                    q_update = dqn_update.predict(state)

                    if done:
                        q_update[0, action] = reward
                    else:
                        q_target = dqn_target.predict(state_next)

                        # original : "Q[0, action] = reward
                        # + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]"
                        q_pred = dqn_update.predict(state_next)
                        q_update[0, action] = reward + dis * q_target[0, np.argmax(q_pred)]
                        # q_update[0, action] = reward + dis * np.max(q_target)

                    x_stack = np.vstack([x_stack, one_hot(state)])
                    y_stack = np.vstack([y_stack, q_update])

                # Train our network using target and predicted Q values on each episode
                dqn_update.update(x_stack, y_stack)

            # 일정 epi 횟수마다 q_pred의 W로 업데이트 (W2_1, W2_2 = W1_1, W1_2)
            dqn_target.w_1 = tf.Variable(tf.identity(dqn_update.w_1), dtype=tf.float32)
            dqn_target.w_2 = tf.Variable(tf.identity(dqn_update.w_2), dtype=tf.float32)

            # for debug
            q_map = get_q_map(dqn_target)
            a = 2

    # for debug
    q_map = get_q_map(dqn_target)
    a = 2

    # test
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    act_ = []
    state_ = []
    rList = []
    for _ in range(10000):
        buf_act = []
        buf_state = []
        while True:
            # env.render()
            q_update = dqn_target.predict(state)

            action = np.argmax(q_update)
            state, reward, done, _ = env.step(action)
            buf_act.append(action)
            buf_state.append(state)
            reward_sum += reward
            if done:
                if reward < 0:
                    reward = 0
                rList.append(reward)
                break
        act_.append(buf_act)
        state_.append(buf_state)


    # a = tf.identity(W_target_1)
    # b = tf.identity(W_target_2)
    print(f'{(time.time() - start_time)} seconds')
    print("Success rate: " + str(sum(rList) / 10000))
    # plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
    # plt.show()

    k = 3


if __name__ == "__main__":
    main()

