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

import numpy as np
import matplotlib.pyplot as plt

def train(ph=0.4, Theta=0.000001):
    V = [0]*100
    for i in range(0, 100):
        V[i] = np.random.random() * 1000
    V[0] = 0
    pi = [0]*100
    counter = 1
    while True:
        Delta = 0
        for s in range(1, 100):  # for each state
            old_v = V[s]
            v = [0] * 51
            for a in range(1, min(s, 100 - s) + 1):
                v[a] = 0
                if a + s < 100:
                    v[a] += ph * (0 + V[s + a])
                    v[a] += (1 - ph) * (0 + V[s - a])
                elif a + s == 100:
                    v[a] += ph
                    v[a] += (1 - ph) * (0 + V[s - a])
            op_a = np.argmax(v)
            pi[s] = op_a
            V[s] = v[op_a]
            Delta = max(Delta, abs(old_v - V[s]))
        counter += 1
        if counter % 1000 == 0:
            print("train loop" + str(counter))
            print("Delta =" + str(Delta))
        if Delta < Theta:
            break
    return [V[1:100], pi[1:100]]
# End train method definition


if __name__ == '__main__':
    [V1, pi1] = train(ph=0.4)
    [V2, pi2] = train(ph=0.25)
    [V3, pi3] = train(ph=0.55)
    S = np.linspace(1, 99, num=99, endpoint=True)

    # Plot everything
    plt.figure()
    plt.plot(S, V1)
    plt.plot(S, V2)
    plt.plot(S, V3)
    plt.show()
    plt.figure()
    plt.bar(S, pi1)
    plt.show()
    plt.figure()
    plt.bar(S, pi2)
    plt.show()
    plt.figure()
    plt.bar(S, pi3)
    plt.show()





























