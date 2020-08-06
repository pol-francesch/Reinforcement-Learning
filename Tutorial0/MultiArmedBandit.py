"""
@author: polfr

How to build policy-gradient based agent that can solve multi-armed bandit problem

"""
import sys
sys.path.append('/home/polfr/.local/lib/python3.8/site-packages')
sys.path.append('/usr/lib/python3/dist-packages')

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# The bandits
# In this section we define our bandits. We choose to have 4 bandits, but more could be added
# pullBandit generates random number from normal distribution with 0 mean. The lower the number,
# the more likely a positive reward.
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)
def pullBandit(bandit) :
    # Get random num
    result = np.random.randn(1)
    if result > bandit:
        # return positive reward
        return 1
    else:
        # return negative reward
        return -1

# The agent
# In this section, we establish a simple neural network. It consists of a set of values for each bandit.
# Each value is an estimate of the return from choosing the bandit. Use policy-gradient method to
# update agent by moving value for selected action towards the reward.
tf.reset_default_graph()

# Establish the feed-forward part of network. This does the choosing
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

# Establish the training procedure. Feed reward and chosen action into the network to compute loss,
# and use it to update network
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# Training agent
# Train the agent by taking actions in the environment and getting rewards. From there, we update network.
total_episodes = 1000  # total number of episodes used to train agent
total_reward = np.zeros(num_bandits)  # set scoreboard for bandits to 0
e = 0.1  # chance of taking random action

init = tf.initialize_all_variables()

# Launch tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        # Choose either random action or one from network
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)

        # Get our reward from our action
        reward = pullBandit(bandits[action])

        # Update network
        _, resp, ww = sess.run([update, responsible_weight, weights],
                               feed_dict={reward_holder: [reward], action_holder: [action]})

        # Update running tally of scores
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
        i += 1
    # End while
# End with

print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising...")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")


























