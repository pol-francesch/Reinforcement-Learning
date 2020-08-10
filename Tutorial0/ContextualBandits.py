"""
@author: polfr

How to build policy-gradient based agent that can solve contextual bandit problem

"""
import sys

# sys.path.append('/home/polfr/.local/lib/python3.8/site-packages')
# sys.path.append('/usr/lib/python3/dist-packages')

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# tf.disable_v2_behavior()


# The Contextual Bandits
# In this example we use 3 four armed bandits (so we have 4 bandits with 3 arms each).
# Each arm on each bandit has a different probability of giving a success, and we want our agent to learn
# which arm on each bandit is best to pull.
# The pullBandit function generates random number from normal distr. with mean of 0. The lower the number,
# the more likely a positive reward

class contextual_bandit:
    def __init__(self):
        self.state = 0
        # List out the bandits. Currently the best arms are: 4, 2, 1
        self.bandits = np.array([[0.2, 0, -0.0, -5],
                                 [0.1, -5, 1, 0.25],
                                 [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    # End __init__

    def getBandit(self):
        # Returns random state for each episode
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    # End getBandit

    def pullArm(self, action):
        # Get random number
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            # Return positive reward
            return 1
        else:
            # Return negative reward
            return -1
    # End pullArm


# End contextual_bandit class definition

# The Policy Based Agent
# Establish simple neural network. Takes the current state as input, and gives the best action as output.
# This makes it so that the agent takes actions based on the environment which is non-constant.
# A set of weights are used to determine the best action for each state, and these are updated with a
# policy-based gradient method.
class agent:
    def __init__(self, lr, s_size, a_size):
        # Establish feed-forward part of network. Agent takes a state, and produces an action
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(state_in_OH, a_size, biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # Establish training procedure for the network. Feed reward and action into network, compute loss
        # and update network accordingly.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)
    # End __init__


# End agent class definition

# Training the Agent
# Train agent by getting state from environment, taking an action and receiving a reward.
# Using these three things, we can properly update our network
tf.reset_default_graph()  # Clear tensorflow graph

cBandit = contextual_bandit()  # load the bandits
myAgent = agent(lr=0.001, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)  # load agent
weights = tf.trainable_variables()[0]  # weights to be evaluated to look into the network

total_episodes = 10000
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])  # set scoreboard to 0
e = 0.1  # Set chance of taking random action

init = tf.initialize_all_variables()

# Launch tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        # Get state from environment
        s = cBandit.getBandit()

        # Choose either random action or one from network
        if np.random.rand(1) < e:
            action = np.random.randint(cBandit.num_actions)
        else:
            action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})

        # Get reward for taking the action
        reward = cBandit.pullArm(action)

        # Update the network
        feed_dict = {myAgent.reward_holder: [reward], myAgent.action_holder: [action], myAgent.state_in: [s]}
        _, ww = sess.run([myAgent.update, weights], feed_dict=feed_dict)

        # Update running tally of scores
        total_reward[s, action] += reward

        if i % 500 == 0:
            print("Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " +
                  str(np.mean(total_reward, axis=1)))
        i += 1
    # End while
# End with

# Print results
for a in range(cBandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a]) + 1) + " for bandit " + str(a + 1)
          + " is the most promising...")
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")
# End for

# End script
