"""
@author: polfr

Implement Deep Q-Network using both Double DQN and Dueling DQN. The agent learns to solve a navigation task in basic
grid world

"""
from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

# Load the game env
# We can easily adjust the size of gridWorld to make it harder or easier for our DQN
from gridworld import gameEnv
env = gameEnv(partial=False, size=5)

# Implementing the Q Network
class Qnetwork:
    def __init__(self, h_size):
        # Network receives frame from the game, flattened into an array. Resizes it and then processes it through
        # four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                                 padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
                                 padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
                                 padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1],
                                 padding='VALID', biases_initializer=None)

        # Take output from final convolution layer and split it into separate advantage and value streams.
        # This is dueling dqn behaviour
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Combine the streams together to get the final Q-values
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Obtain loss by taking sum of square difference between target and prediction Q values
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
    # End __init__ method definition
# End Qnetwork class definition


# Experience Replay
# Store experiences and sample them randomly to train network
class experience_buffer:
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0: (len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
# End experience_buffer class definition


# Simple function to resize game frames
def processState(states):
    return np.reshape(states, [21168])

# Functions to update parameters of target network with those of the primary network
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0: total_vars//2]):
        op_holder.append(tfVars[idx + total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# Training the Network
# Set all training parameters
batch_size = 32             # How many experiences to use for each training step
update_freq = 4             # How often to perform training step
y = 0.99                    # Discount factor on target Q-values
startE = 1                  # Starting chance for random action
endE = 0.1                  # Final chance for random action
annealing_steps = 10000     # How many steps of training to reduce startE to endE
num_episodes = 10000        # How many episodes of game env to train network with
pre_train_steps = 10000     # How many steps of random action before training begins
max_epLength = 50           # The max allowed length of our episode
load_model = False          # Whether to load a saved model
path = "./dqn"              # The path to save our model to
h_size = 512                # Size of final convolutional layer before splitting into Advantage and Value streams]
tau = 0.001                 # Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease
e = startE
steDrop = (startE - endE) / annealing_steps

# Create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make path for our model to be saved in
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episode_buffer = experience_buffer()
        # Reset env and get first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0

        # The Q-Network
        # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
        while j < max_epLength:
            j += 1
            # Choose either greedy action (from Q-network) or random action
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]

            # Take the action and get new state
            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1

            # Save the experience to the episode buffer
            episode_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

            # Train the network if we have completed the pre work
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= steDrop

                if total_steps % update_freq == 0:
                    # Get random batch of experiences
                    trainBatch = myBuffer.sample(batch_size)

                    # Perform the double DQN update to target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y*doubleQ*end_multiplier)

                    # Update the network with target values
                    _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                                mainQN.targetQ: targetQ,
                                                                mainQN.actions: trainBatch[:, 1]})
                    rAll += r
                    s = s1

                    if d == True:
                        break
                # End update sequence
            # End train sequence
        # End while loop

        myBuffer.add(episode_buffer.buffer)
        jList.append(j)
        rList.append(rAll)

        # Periodically save the model
        if i % 1000 == 0:
            saver.save(sess, path+'/model-'+str(i)+'.ckpt')
            print("Saved model")
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)
    # End for loop
    saver.save(sess, path+'/model-'+str(i)+'.ckpt')
# End with

print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")

# Checking Network Learning
# Mean reward over time
rMat = np.resize(np.array(rList), [len(rList)//100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)

plt.show()































