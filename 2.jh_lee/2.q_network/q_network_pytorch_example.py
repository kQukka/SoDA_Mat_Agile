import os
import time
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# define actions
actions = ('left', 'down', 'right', 'up')


# --------------------------------------------------------------------------------------------------------
def print_policy(num_state, max_row, max_col):
    policy = [agent(s).argmax(1)[0].detach().item() for s in range(num_state)]
    policy = np.asarray([actions[act] for act in policy])
    policy = policy.reshape((max_row, max_col))
    print("\n\n".join('\t'.join(line) for line in policy) + "\n")


def one_hot_encoding(x, num_state):
    out_tensor = torch.zeros([1, num_state])
    out_tensor[0][x] = 1
    return out_tensor
    # return np.identity(16)[x:x + 1].astype(np.float32)


class QNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        self.state_space = state_space
        self.hidden_size = state_space

        self.l1 = nn.Linear(in_features=self.state_space, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space)

    def forward(self, x):
        x = one_hot_encoding(x, self.state_space)
        out1 = torch.sigmoid(self.l1(x))
        return self.l2(out1)


# --------------------------------------------------------------------------------------------------------
env = gym.make('FrozenLake-v1')

# Set Q-learning parameters
lr = .03
num_episodes = 2000
e = 0.1
dis = .99

# Input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n

# weight
W = tf.Variable(tf.random.uniform([input_size, output_size], 0, 0.01), dtype=tf.float32)

# Make use of cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# init Q-Network
agent = QNetwork(input_size, output_size).to(device)
# optimizer
optimizer = optim.SGD(params=agent.parameters(), lr=lr)
# loss
criterion = nn.SmoothL1Loss()


# --------------------------------------------------------------------------------------------------------
start_time = time.time()
# rewards per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    local_loss = []

    e = 1. / ((i / 50) + 10)
    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily (with e chance of random action) from the Q-network
        with torch.no_grad():
            # Do a feedforward pass for the current state s to get predicted Q-values
            # for all actions (=> agent(s)) and use the max as action a: max_a Q(s, a)
            action = agent(state).max(1)[1].view(1, 1)  # max(1)[1] returns index of highest value

        q_value = agent(state).max(1)[0].view(1, 1)
        # if np.random.rand(1) < e:
        #     action = env.action_space.sample()
        # else:
        #     action = np.argmax(q_value)

        # Get new state and reward from environment
        # perform action to get reward r, next state s1 and game_over flag
        # calculate maximum overall network outputs: max_a’ Q(s1, a’).
        state_next, reward, done, _ = env.step(action.tolist()[0][0])

        if done:
            # Update Q, and no q_value+1, since it's action termial state
            # q_value[0, action] = reward
            q_value_next = reward
        else:
            # Obtain the Q_s` values by feeding the new state through our network
            q_value_next = agent(state_next).max(1)[0].view(1, 1)
            with torch.no_grad():
                # Update Q
                q_value_next = reward + dis * q_value_next

        # print(q, target_q)
        # Calculate loss
        loss = criterion(q_value, q_value_next)
        if i % 100 == 0:
            print("loss and reward: ", i, loss, reward)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward
        state = state_next
    rList.append(rAll)


# print("\Average steps per episode: " + str(sum(jList) / num_episodes))
print("\nScore over time: " + str(sum(rList) / num_episodes))
print("\nFinal Q-Network Policy:\n")
print_policy(input_size, 4, 4)
# plt.plot(jList)
# plt.plot(rList)
plt.savefig("j_q_network.png")
plt.show()

k = 3