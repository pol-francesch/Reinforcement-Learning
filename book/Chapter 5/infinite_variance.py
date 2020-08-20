"""
@author: polfr

The estimates of ordinary importance sampling will
typically have infinite variance, and thus unsatisfactory convergence properties, whenever
the scaled returns have infinite variance—and this can easily happen in o↵-policy learning
when trajectories contain loops. A simple example is shown inset in Figure 5.4. There is
only one nonterminal state s and two actions, right and left. The right action causes a
deterministic transition to termination, whereas the left action transitions, with probability
0.9, back to s or, with probability 0.1, on to termination. The rewards are +1 on the
latter transition and otherwise zero. Consider the target policy that always selects left.
All episodes under this policy consist of some number (possibly zero) of transitions back
to s followed by termination with a reward and return of +1. Thus the value of s under
the target policy is 1 ( = 1). Suppose we are estimating this value from o↵-policy data
using the behavior policy that selects right and left with equal probability

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

ACTION_BACK = 0
ACTION_END = 1

# Behaviour policy
def behaviour_policy():
    return np.random.binomial(1, 0.5)

# Target policy
def target_policy():
    return ACTION_BACK

# One turn
def play():
    # Track the action for importance ratio
    trajectory = []
    while True:
        action = behaviour_policy()
        trajectory.append(action)
        if action == ACTION_END:
            return 0, trajectory
        if np.random.binomial(1, 0.9) == 0:
            return 1, trajectory

def figure_5_4():
    runs = 10
    episodes = 100000
    for run in range(runs):
        rewards = []
        for episode in range(0, episodes):
            reward, trajectory = play()
            if trajectory[-1] == ACTION_END:
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(trajectory))
            rewards.append(rho * reward)
        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, episodes + 1)
        plt.plot(estimations)

    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.savefig('images/figure_5_4.png')
    plt.close()


if __name__ == '__main__':
    figure_5_4()


































