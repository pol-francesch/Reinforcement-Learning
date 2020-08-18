"""
@author: polfr

A gambler has the opportunity to make bets on the
outcomes of a sequence of coin flips. If the coin comes up heads, he wins as many dollars as
he has staked on that flip; if it is tails, he loses his stake. The game ends when the gambler
wins by reaching his goal of $100, or loses by running out of money. On each flip, the gambler
must decide what portion of his capital to stake, in integer numbers of dollars. This problem
can be formulated as an undiscounted, episodic, finite MDP. The state is the gambler’s capital,
s 2 {1, 2,..., 99} and the actions are stakes, a 2 {0, 1,..., min(s, 100  s)}.

The reward is zero on all transitions except those on which the gambler reaches his goal, when
it is +1. The state-value function then gives the probability of winning from each state. A
policy is a mapping from levels of capital to stakes. The optimal policy maximizes the probability
of reaching the goal. Let ph denote the probability of the coin coming up heads. If ph is known,
then the entire problem is known and it can be solved, for instance, by value iteration.
Figure 4.3 shows the change in the value function over successive sweeps of value iteration, and
the final policy found, for the case of ph = 0.4. This policy is optimal, but not unique. In fact,
there is a whole family of optimal policies, all corresponding to ties for the argmax action
selection with respect to the optimal value function. Can you guess what the entire family looks like?

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# Define constants
GOAL = 100
STATES = np.arange(GOAL + 1)
HEAD_PROB = 0.4                     # Probability of getting heads

def figure_4_3():
    # State value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    # Value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1: GOAL]:
            # Get possible actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(HEAD_PROB * state_value[state + action] +
                                      (1 - HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break
    # End while loop

    # Compute the optimal policy
    policy = np.zeros(GOAL + 1)
    for state in STATES[1: GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(HEAD_PROB * state_value[state + action] +
                                  (1 - HEAD_PROB) * state_value[state - action])

        # Round to resemble book figure
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
    # End for loop

    # Plot
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stakes)')

    plt.savefig('images/figure_4_3.png')
    plt.close()


if __name__ == '__main__':
    figure_4_3()













































