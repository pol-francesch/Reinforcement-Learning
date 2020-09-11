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

import numpy as np
from tabulate import tabulate

# Class for each unit on the game
class State:
    def __init__(self, _id):
        if _id != 0:
            self.value = 0
        elif _id == 0:
            self.value = 0
        self.id = _id
        self.left_bound = max(1, (self.id//4) * 4)          # save the left border
        self.right_bound = min(14, (self.id//4) * 4 + 3)    # save the right border
        self.nextS = [self.move('L'), self.move('R'), self.move('U'), self.move('D')]

    # Get next state (nextS)
    def move(self, u):
        if u == 'L':        # move left
            if self.id - 1 >= self.left_bound:
                return self.id - 1
            elif self.id - 1 == 0:
                return 0
            else:
                return self.id
        if u == 'R':        # move right
            if self.id + 1 >= self.right_bound:
                return self.id + 1
            elif self.id - 1 == 15:
                return 0
            else:
                return self.id
        if u == 'U':        # move up
            if self.id - 4 >= 1:
                return self.id - 4
            elif self.id - 4 == 0:
                return 0
            else:
                return self.id
        if u == 'D':        # move down
            if self.id + 4 >= 14:
                return self.id + 4
            elif self.id + 4 == 15:
                return 0
            else:
                return self.id

    # S is whole set of the States
    def update(self, S):
        V = 0
        for i in range(0, 4):
            V += S[self.nextS[i]].value

        self.value = -1 + 0.25*V
# End of State class definition


def train(k=10):
    V = []
    S_T = State(0)
    S = {0: S_T}

    for j in range(k):
        S[j] = State(j)

    for loop in range(k):
        if loop >= 1000 and loop % 1000 == 0:
            print("Training " + str(loop) + "'s loop.........Remaining: " + str(k-loop) + " loops")
        n = np.random.random()
        if n > 0.5:
            for j in range(1, 15):
                S[j].update(S)
        else:
            for j in range(14, 0, -1):
                S[j].update(S)

    for t in range(0, 16):
        if t == 0 or t == 15:
            V.append("0")
        else:
            V.append(S[t].value)
    draw(V)
# End of train function definition

def draw(valueArray):
    for i in range(4):
        print("----------------------")
        print("| "+str(int(valueArray[i*4]))+" | "+str(int(valueArray[i*4+1])) +" | "+str(int(valueArray[i*4+2])) +" | "+str(int(valueArray[i*4+3])) +" |")
    print("----------------------")
    print("Accurate State Values List:")
    for i in range(1,8):
        print("State "+str(2*i-1)+": "+str(valueArray[2*i-1])+ "          State "+str(2*i)+": "+str(valueArray[2*i]))
# End of draw function definition


if __name__ == '__main__':
    k = input("Specify the desired training loop count(0-10000):")
    train(int(k))










































