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


# Input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
h_size = NUM_SAMPLING

# weight
# q_pred(예측 q_value)을 위한 W
# W1_1 = tf.Variable(tf.random.uniform([input_size, h_size], 0, 0.01), dtype=tf.float32)
W_update_1 = tf.Variable(tf.random.uniform([input_size, h_size], 0, 0.01), dtype=tf.float32)
W_update_2 = tf.Variable(tf.random.uniform([h_size, output_size], 0, 0.01), dtype=tf.float32)
# W_update_1 = tf.Variable(tf.initializers.GlorotUniform()([input_size, h_size]), dtype=tf.float32)
# W_update_2 = tf.Variable(tf.initializers.GlorotUniform()([h_size, output_size]), dtype=tf.float32)

# target(실제 계산된 q_value)을 위한 W, 스스로 업데이트 되지 않음
# 학습 시작과 일정 epi 횟수마다 q_pred의 W로 업데이트 (W2_1, W2_2 = W1_1, W1_2)
W_target_1 = tf.Variable(tf.identity(W_update_1), dtype=tf.float32)
W_target_2 = tf.Variable(tf.identity(W_update_2), dtype=tf.float32)

# optimizer_1
optimizer_update = tf.optimizers.Adam(learning_rate=learning_rate)
# optimizer_target = tf.optimizers.SGD(learning_rate=learning_rate)


start_time = time.time()
# rewards per episode
rList = []
replay_buffer = deque()
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
        activation = tf.nn.tanh(tf.matmul(one_hot(state), W_target_1))
        q_value = tf.matmul(activation, W_target_2)
        q_value = np.array(q_value.numpy())

        # Choose an action by greedly (with a chance of random action)
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_value)
        # action = np.argmax(q_value)

        # Get new state and reward from environment
        state_next, reward, done, _ = env.step(action)

        # Penalty
        if done:
            rAll += reward
            if reward >= 1.:
                reward = 10
            # reward = -100
        replay_buffer.append((state, action, reward, state_next, done))
        if len(replay_buffer) > REPLAY_MEMORY:
            replay_buffer.popleft()
        state = state_next

        cnt_step += 1
        if cnt_step > MAX_STEP:   # Good enough. Let's move on
            break
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
                activation = tf.nn.tanh(tf.matmul(one_hot(state), W_update_1))
                q_update = tf.matmul(activation, W_update_2)
                q_update = np.array(q_update.numpy())

                if done:
                    q_update[0, action] = reward
                    if reward:
                        pass
                else:
                    activation = tf.nn.tanh(tf.matmul(one_hot(state_next), W_target_1))
                    q_target = tf.matmul(activation, W_target_2)
                    q_target = np.array(q_target.numpy())

                    # activation = tf.nn.tanh(tf.matmul(one_hot(state_next), W_update_1))
                    # q_pred = tf.matmul(activation, W_update_2)
                    # q_pred = np.array(q_pred.numpy())

                    # 두개 머가 다르지
                    q_update[0, action] = reward + dis * np.max(q_target)
                    # a = reward + dis * np.max(q_target)
                    # q_update[0, action] = reward + dis * q_target[0, np.argmax(q_pred)]

                x_stack = np.vstack([x_stack, one_hot(state)])
                y_stack = np.vstack([y_stack, q_update])

            # Train our network using target and predicted Q values on each episode

            # val_loss = tf.reduce_mean(input_tensor=q_value - tf.matmul(one_hot(state), W))
            # print(val_loss)
            # a = tf.identity(W_update_1)
            # b = tf.identity(W_update_2)

            # loss = lambda: tf.reduce_mean(
            #     input_tensor=tf.square(q_update -
            #                            tf.matmul(tf.nn.tanh(tf.matmul(one_hot(state), W_update_1)), W_update_2)))

            print(tf.reduce_mean(input_tensor =
                                 (tf.square(y_stack -
                                            tf.matmul(tf.nn.tanh(tf.matmul(x_stack, W_update_1)), W_update_2)))))
            loss = lambda: tf.reduce_mean(
                input_tensor=tf.square(y_stack -
                                       tf.matmul(tf.nn.tanh(tf.matmul(x_stack, W_update_1)), W_update_2)))
            optimizer_update.minimize(loss, var_list=[W_update_1, W_update_2])
            # optimizer_update.minimize(loss)
        # print("Loss: ", loss)
        # copy q_net -> target_net
        # 일정 epi 횟수마다 q_pred의 W로 업데이트 (W2_1, W2_2 = W1_1, W1_2)
        # a = tf.identity(W_update_1).numpy()
        # b = tf.identity(W_update_2).numpy()
        # activation = tf.nn.tanh(tf.matmul(one_hot(0), W_target_1))
        # c = tf.matmul(activation, W_target_2)
        W_target_1 = tf.Variable(tf.identity(W_update_1), dtype=tf.float32)
        W_target_2 = tf.Variable(tf.identity(W_update_2), dtype=tf.float32)

        q_map = []
        for idx in range(input_size):
            activation = tf.nn.tanh(tf.matmul(one_hot(idx), W_target_1))
            q_target = tf.matmul(activation, W_target_2)
            q_target = np.array(q_target.numpy())
            q_map.append(q_target)
        a=2


q_map = []
for idx in range(16):
    activation = tf.nn.tanh(tf.matmul(one_hot(idx), W_target_1))
    q_target = tf.matmul(activation, W_target_2)
    q_target = np.array(q_target.numpy())
    q_map.append(q_target)

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
        activation = tf.nn.tanh(tf.matmul(one_hot(state), W_update_1))
        q_update = tf.matmul(activation, W_update_2)
        q_update = np.array(q_update.numpy())

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