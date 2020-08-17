"""
@author: polfr

Exercise 2.5:

Design and conduct an experiment to demonstrate the
difficulties that sample-average methods have for non-stationary problems. Use a modified
version of the 10-armed testbed in which all the q⇤(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean 0
and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2
for an action-value method using sample averages, incrementally computed, and another
action-value method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 and
longer runs, say of 10,000 steps.

"""

import numpy as np
import matplotlib.pyplot as plt

# x is a vector.
# Each element takes random walk independently
# Returns vector where each element takes a step by the rule of random walk
def RandomWalk(x):
    dim = np.size(x)
    walk_set = [-1, 1, 0]
    for i in range(dim):
        x[i] = x[i] + np.random.choice(walk_set)
    return x


# Q = current action value estimate
# Returns epsilon-greedy action
def eps_greedy(epsilon, Q):
    i = np.argmax(Q)
    dim = np.size(Q)
    action_space = range(0, dim, 1)
    sample = np.random.uniform(0, 1)

    if sample <= 1 - epsilon:
        return i
    else:
        np.delete(action_space, i)
        return np.random.choice(action_space)


# Main work function
def multi_task(max_iter, task_number, epsilon, arm_number, step_size):
    rows, cols = task_number, arm_number
    my_matrix = np.array([([0]*cols) for i in range(rows)])
    constQ = np.array([([0]*cols) for i in range(rows)])
    variaQ = np.array([([0]*cols) for i in range(rows)])
    q = np.array([([0]*cols) for i in range(rows)])
    constN = np.array([([0]*cols) for i in range(rows)])
    variaN = np.array([([0]*cols) for i in range(rows)])
    constR = np.zeros(max_iter)
    variaR = np.zeros(max_iter)

    for i in range(max_iter):
        for j in range(task_number):
            # Random walk of each arm
            task_q = q[j, :]
            task_q = RandomWalk(task_q)
            q[j, :] = task_q

            # Constant step size
            task_constQ = constQ[j, :]
            task_constN = constN[j, :]
            action_const = eps_greedy(epsilon, task_constQ)

            rewardConst = task_q[action_const]
            constR[i] = constR[i] + rewardConst
            task_constN[action_const] = task_constN[action_const] + 1
            alpha = step_size
            difference = rewardConst - task_constQ[action_const] + 1
            # NewEstimate <- OldEstimate + StepSize * [Target - OldEstimate]
            task_constQ[action_const] = task_constQ[action_const] + alpha * difference
            constQ[j, :] = task_constQ
            constN[j, :] = task_constN

            # Changing step size
            task_variaQ = variaQ[j, :]
            task_variaN = variaN[j, :]
            action_varia = eps_greedy(epsilon, task_variaQ)
            reward_varia = task_q[action_varia]
            task_variaN[action_varia] = task_variaN[action_varia] + 1

            if i == 0:
                beta = 1
            else:
                beta = 1/task_variaN[action_varia]
                # print("Beta is weird. i: " + str(i) + "\tj: " + str(j))
            '''if i == 1:
                print("task_variaQ: " + str(task_variaQ[action_varia]) + "\taction_varia: " + str(action_varia) +
                      "\tbeta: " + str(beta) + "\treward_varia: " + str(reward_varia) +
                      "\ttask_variaN: " + str(task_variaN[action_varia]))'''
            # NewEstimate <- OldEstimate + StepSize * [Target - OldEstimate]
            task_variaQ[action_varia] = task_variaQ[action_varia] + beta*(reward_varia - task_variaQ[action_varia])

            variaN[j, :] = task_variaN
            variaQ[j, :] = task_variaQ
            variaR[i] = variaR[i] + reward_varia
            # End j-loop

        variaR[i] = variaR[i]/task_number
        constR[i] = constR[i]/task_number
        # End i-loop

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.plot(variaR, color='r')
    plt.plot(constR, color='b')
    plt.xticks(np.arange(0, max_iter + 1, 100))
    plt.show()
    plt.close()
    print(q)
    print(constQ)
    print(variaQ)


# Run everything, set variables
max_iter = 1000
task_number = 500
epsilon = 0.1
arm_number = 10
step_size = 0.1
multi_task(max_iter, task_number, epsilon, arm_number, step_size)

































