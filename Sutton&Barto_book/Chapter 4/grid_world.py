"""
@author: polfr

The nonterminal states are S = {1, 2,..., 14}. There are four actions possible in each
state, A = {up, down, right, left}, which deterministically cause the corresponding
state transitions, except that actions that would take the agent o↵ the grid in fact leave
the state unchanged. Thus, for instance, p(6, 1|5, right) = 1, p(7, 1|7, right) = 1,
and p(10, r|5, right) = 0 for all r 2 R. This is an undiscounted, episodic task. The
reward is 1 on all transitions until the terminal state is reached. The terminal state is
shaded in the figure (although it is shown in two places, it is formally one state). The
expected reward function is thus r(s, a, s0) = 1 for all states s, s0 and actions a. Suppose
the agent follows the equiprobable random policy (all actions equally likely). The left side
of Figure 4.1 shows the sequence of value functions {vk} computed by iterative policy
evaluation. The final estimate is in fact v⇡, which in this case gives for each state the
negation of the expected number of steps from that state until termination.

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

# Define constants
WORLD_SIZE = 4
# Left, up, right and down
ACTIONS = [np.array([0, -1]),
           np.array([1, 0]),
           np.array([0, 1]),
           np.array([-1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']
ACTION_PROB = 0.25

def is_terminal(state):
    x, y = state
    return(x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)

def step(state, action):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward
# End state function definition

def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # Row and column labels
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)
# End draw_image function definition

def compute_state_value(in_place=True, discount=1.0):
    new_state_value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0

    while True:
        if in_place:
            state_values = new_state_value
        else:
            state_values = new_state_value.copy()
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                new_state_value[i, j] = value

        max_delta_value = abs(old_state_values - new_state_value).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1
    return new_state_value, iteration
# End compute_state_value function definition

def figure_4_1():
    # Book suggests using in-place iterative policy evaluation, the figure in the book uses out-og-place version
    _, async_iteration = compute_state_value(in_place=True)
    values, sync_iteration = compute_state_value(in_place=False)
    draw_image(np.round(values, decimals=2))
    print('In-place: {} iterations'.format(async_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))

    plt.savefig('images/figure_4_1.png')
    plt.close()
# End figure_4_1 function definition


if __name__ == '__main__':
    figure_4_1()








































