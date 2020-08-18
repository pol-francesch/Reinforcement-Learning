"""
@author: polfr

Jack manages two locations for a nationwide car
rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. Cars become available for
renting the day after they are returned. To help ensure that cars are available where
they are needed, Jack can move them between the two locations overnight, at a cost of
$2 per car moved. We assume that the number of cars requested and returned at each
location are Poisson random variables, meaning that the probability that the number is
n is n
n! e, where  is the expected number. Suppose  is 3 and 4 for rental requests at
the first and second locations and 3 and 2 for returns. To simplify the problem slightly,
we assume that there can be no more than 20 cars at each location (any additional cars
are returned to the nationwide company, and thus disappear from the problem) and a
maximum of five cars can be moved from one location to the other in one night. We take
the discount rate to be  = 0.9 and formulate this as a continuing finite MDP, where
the time steps are days, the state is the number of cars at each location at the end of
the day, and the actions are the net numbers of cars moved between the two locations
overnight. Figure 4.2 shows the sequence of policies found by policy iteration starting
from the policy that never moves any cars

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

# Lambda: expected number of poisson random distribution
# Return dictionary made up of: {value: possibility}
def poisson_calculator(Lambda=3):
    result = {}
    for i in range(0, 21):
        result[i] = max(np.finfo(float).eps, abs((np.power(Lambda, i) / (np.math.factorial(i))) * np.exp(-Lambda)))
    return result

# all_possibility: dictionary made up of: {(customer_A, returned_car_A, customer_B, returned_car_B): Joint Possibility}
# Return dictionary made up of: {((state_value_A, state_value_B), a): reward_dict}
# where reward_dict = {reward: Joint Possibility}
# Note that: S_T --> (day)sell --> (night)returned --> policy action --> S_{T+1}
def P_calculate(all_possibility):
    for state_value_A in range(21):             # car left at end of day at A
        print("State " + str(state_value_A))
        for state_value_B in range(21):         # car left at end of day at B
            P = {}
            for action in range(-5, 6):         # action range is from -5 to 5
                temp = {}
                # problem: action=-5 A=1 B=10
                if action <= state_value_A and -action <= state_value_B and action + state_value_B <= 20 and -action + state_value_A <= 20:
                    for customerA in range(21):                     # Total customers come at the end of day at A
                        for customerB in range(21):                 # Total customers come at the end of day at B
                            for returned_car_A in range(21):        # Total cars come at the end of day at A
                                for returned_car_B in range(21):    # Total cars come at the end of day at B
                                    value_A_Changed = min(20, state_value_A + returned_car_A - action -
                                                          min(customerA, state_value_A - action))
                                    value_B_Changed = min(20, state_value_B + returned_car_B - action -
                                                          min(customerB, state_value_B - action))
                                    reward = 10 * min(customerA, state_value_A - action) +\
                                        10 * min(customerB, state_value_B + action) - abs(action) * 2
                                    temp[((value_A_Changed, value_B_Changed), reward)] = temp.get(
                                        (value_A_Changed, value_B_Changed),
                                        0)
                                    temp[((value_A_Changed, value_B_Changed), reward)] += all_possibility[
                                        (customerA, returned_car_A, customerB, returned_car_B)]
                    P[action] = temp
            with open('4_2_data/P' + str(state_value_A) + str('_') + str(state_value_B), 'wb') as f:
                pickle.dump(P, f, protocol=-1)
# End of P_calculate method definition


def policy_evaluation(V, pi, Theta):
    counter = 1
    while True:
        Delta = 0
        print("Calculating loop " + str(counter))
        for i in range(21):
            print("----Calculating " + str(i))
            for j in range(21):
                with open('4_2_data/P' + str(i) + str('_') + str(j), 'rb') as f:
                    p = pickle.load(f)
                a = pi[(i, j)]
                p = p[a]
                old_value = V[(i, j)]
                for keys, values in p.items():
                    (states, reward) = keys
                    possibility = values
                    V[(i, j)] += (reward + 0.9 * V[states]) * possibility
                Delta = max(Delta, abs(V[(i, j)] - old_value))
        print("Delta = " + str(Delta))
        if Delta < Theta:
            return V
        counter += 1
# End of policy_evaluation function definition


def policy_improvement(V, pi={}):
    counter = 1
    while True:
        print("Calculating policy loop " + str(counter))
        policy_stable = True
        for keys, old_action in pi.items():
            with open('4_2_data/P' + str(keys[0])+str('_')+str(keys[1]), 'rb') as f:
                p = pickle.load(f)
            possible_q = [0]*11
            [state_value_A, state_value_B] = keys
            for possible_action in range(-5, 5):
                index = possible_action + 5
                if possible_action <= state_value_A and -possible_action <= state_value_B and\
                    possible_action + state_value_B <= 20 and -possible_action + state_value_A <= 20:
                    p_a = p[possible_action]
                    for p_keys, values in p_a.items():
                        [states, reward] = p_keys
                        possibility = values
                        possible_q[index] += (reward + 0.9 * V[states]) * possibility
                else:
                    possible_q[index] = -999
            pi[keys] = np.argmax(possible_q) - 5
            if pi[keys] != old_action:
                policy_stable = False
        if policy_stable:
            return pi
        counter += 1
# End policy_improvement method definition

def init():
    customer_A = poisson_calculator(3)      # Possible customers from location A and corresponding possibility
    customer_B = poisson_calculator(4)      # Possible customers from location A and corresponding possibility
    return_A = poisson_calculator(3)        # Possible cars returned from location A and corresponding possibility
    return_B = poisson_calculator(2)        # Possible cars returned from location B and corresponding possibility
    all_possibility_A = {}
    all_possibility_B = {}
    all_possibility = {}

    for i in range(21):
        for j in range(21):
            all_possibility_A[(i, j)] = max(np.finfo(float).eps, abs(np.multiply(customer_A[i], return_A[j])))
            all_possibility_B[(i, j)] = max(np.finfo(float).eps, abs(np.multiply(customer_B[i], return_B[j])))
            # min here is to prevent underflow. np.finfo(float).eps is exactly EPS of this machine
            # they are joint possibility that customers and returned cars both happen
    for i in range(21):
        for j in range(21):
            for m in range(21):
                for n in range(21):
                    all_possibility[(i, j, m, n)] = max(np.finfo(float).eps,
                                                        abs(all_possibility_A[i, j] * all_possibility_B[i, j]))
    with open('4_2_data/all_possibility', 'wb') as f:
        pickle.dump(all_possibility, f, protocol=-1)

    P_calculate(all_possibility)
# End of init method definition


def train():
    V = {}
    for i in range(21):
        for j in range(21):
            V[(i, j)] = 10 * np.random.random()

    pi = {}
    for i in range(21):
        for j in range(21):
            pi[(i, j)] = 0
    for q in range(5):
        print("Big loop " + str(q))
        V = policy_evaluation(V, pi, Theta=0.01)
        pi = policy_improvement(V, pi)
        with open('4_2_data/pi' + str(q), 'wb') as f:
            pickle.dump(pi, f, protocol=-1)
        with open('4_2_data/V' + str(q), 'wb') as f:
            pickle.dump(V, f, protocol=-1)
        print("================")
        for i in range(21):
            print("i = " + str(i))
            for j in range(21):
                print("  " + str(pi[i, j]))
# End of train method definition

def main():
    init()
    train()


if __name__ == '__main__':
    main()






































