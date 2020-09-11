"""
@author: polfr

Creating a test-bed for the 10-armed bandit.

Generate the plots and figures found in chp 2 of the book

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')

class Bandit:
    # num_arm: Number of arms for the bandit
    # epsilon: probability of exploration in epsilon-greedy algorithm
    # initial: initial estimation of each action
    # step_size: constant step size for updating estimations
    # sample_avgs: if true, use sample averages to update estimations instead of step size
    # UCB_param: if not None, use UCB algorithm to select action
    # gradient: if True, use gradient bases bandit problem
    # gradient_baseline: if true, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, num_arm=10, epsilon=0, initial=0., step_size=0.1, sample_avgs=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0):
        self.k = num_arm
        self.step_size = step_size
        self.sample_averages = sample_avgs
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # Real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # Estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # Number of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # Get action for bandit
    def act(self):
        # Get random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        # Get UCB action
        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation +\
                             self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        # Get gradient based action
        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        # Take best action from table
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # Take action and update estimate for the action
    def step(self, action):
        # Generate reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # Update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            # Update estimation using gradient method
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0

            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # Update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])

        return reward
# End of Bandit class definition


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)

    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward

                if action == bandit.best_action:
                    best_action_counts[i, r, t] += 1

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('images/figure_2_1.png')
    plt.close()

def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_avgs=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % eps)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % eps)
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('images/figure_2_2.png')
    plt.close()

def figure_2_3(runs=2000, time=1000):
    bandits = [Bandit(epsilon=0, initial=5, step_size=0.1), Bandit(epsilon=0.1, initial=0, step_size=0.1)]
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='epsilon = 0, q = 5')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('images/figure_2_3.png')
    plt.close()

def figure_2_4(runs=2000, time=1000):
    bandits = [Bandit(epsilon=0, UCB_param=2, sample_avgs=True), Bandit(epsilon=0.1, sample_avgs=True)]
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB; c = 2')
    plt.plot(average_rewards[1], label='epsilon greedy; epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('images/figure_2_4.png')
    plt.close()

def figure_2_5(runs=2000, time=1000):
    bandits = [Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4),
               Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4),
               Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4),
               Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4)]
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])

    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('images/figure_2_5.png')
    plt.close()

def figure_2_6(runs=2000, time=1000):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_avgs=True),
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_avgs=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i: i+l], label=label)
        i += 1
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('images/figure_2_6.png')
    plt.close()


if __name__ == '__main__':
    # figure_2_1()
    # figure_2_2()
    # figure_2_3()
    # figure_2_4()
    # figure_2_5()
    figure_2_6()




































