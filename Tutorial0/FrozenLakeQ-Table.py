"""
Created on Wed Aug  5 11:01:53 2020

@author: polfr

Simple example of Q-Table learning using the Bellman Equation and the
FrozenLake example from the OpenAI gym.

Bellman eq: Q(s,a) = r + gamma(max(Q(s', a')))

s = state, a = action, r = current reward, gamma = maximum discounted future reward
"""
import sys
sys.path.append('/home/polfr/.local/lib/python3.8/site-packages')

import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
# Init table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = 0.8
y = 0.95
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

# Iterate through the episodes
for i in range(num_episodes):
    # Reset env and get first new observation
    s = env.reset()  # State
    rAll = 0
    d = False
    j = 0

    # Implement Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i+1)))

        # Get new state and reward from env
        s1, r, d, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr*(r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1

        if d:
            break
    rList.append(rAll)

# Display results
print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
