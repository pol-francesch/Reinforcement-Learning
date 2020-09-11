"""
@author: polfr

How to build policy-gradient based agent that can solve CartPole problem

"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

try:
    xrange = xrange
except:
    xrange = range

env = gym.make('CartPole-v0')

# The Policy Based Agent
gamma = 0.99

def discount_rewards(r):
    # Take 1D float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r
# End discounted_rewards method definition

class agent:
    def __init__(self, lr, s_size, a_size, h_size):
        # Produce feed-forward part of network. Take a state and produce an action
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None,
                                      activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax,
                                           biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # Establish training procedure. Feed reward and chosen action into network to compute loss, and update.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
    # End __init__ method definition
# End agent class definition


# Training the agent
tf.reset_default_graph()  # Clear Tensorflow graph

myAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8)  # Load agent

total_episodes = 5000  # max number of episodes
max_ep = 999  # max runs per episode
update_frequency = 5

init = tf.global_variables_initializer()

# Launch tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # Probabilistically pick action given network outputs
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            # Get our reward for taking an action
            s1, r, d, _ = env.step(a)
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r

            if d == True:
                # Update the network
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)

                for idx, grad in enumerate(grads):
                    if i > 1:
                        print("i: " + str(i) + " j: " + str(j) + " gradBuffer: " + str(gradBuffer[idx].__sizeof__())
                              + " idx: " + str(idx.__sizeof__()))
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)

                    for idx, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad*0
                # End i % update_frequency if

                total_reward.append(running_reward)
                total_length.append(j)
                break
            # End d if
        # End for loop

        # Update our running tally of scores
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1
    # End while
# End with

# End script





























