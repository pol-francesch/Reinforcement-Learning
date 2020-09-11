"""
@author: polfr

Simple example of Q-Network learning using the
FrozenLake example from the OpenAI gym.


"""
import sys
sys.path.append('/home/polfr/.local/lib/python3.8/site-packages')
sys.path.append('/usr/lib/python3/dist-packages')

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# Load environment
env = gym.make('FrozenLake-v0')

# Q-Network approach
# Implementing the network
tf.reset_default_graph()

# Establish feed-forward part of the network to choose actions
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Obtain loss by taking sum of squares difference between target and prediction Q values
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# Training the network
init = tf.initialize_all_variables()

# Set learning params
y = 0.99
e = 0.1
num_episodes = 2000

# Lists to store total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset env and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0

        # Q-Network
        while j < 99:
            j += 1
            # Choose action by either greedy or e chance or random action from Q-Network
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get new state and reward for env
            s1, r, d, _ = env.step(a[0])
            # Obtain Q' values by feeding new state through network
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1+1]})
            # Obtain maxQ' and set target value for chosen action
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y*maxQ1
            # Train network using target and predicted values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s+1], nextQ: targetQ})
            rAll += r
            s = s1
            if d:
                # Reduce chance of random action as we train the model
                e = 1./((i/50) + 10)
                break
            # End of while
        jList.append(j)
        rList.append(rAll)
        # End of for

    # End of with

print("Percent of successful episodes: " + str(sum(rList)/num_episodes*100) + "%")
plt.plot(rList)
plt.show()

plt.plot(jList)
plt.show()
























