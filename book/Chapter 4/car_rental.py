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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use('Agg')

# Define constants
MAX_CARS = 20                       # max number of cars per location
MAX_MOVE_OF_CARS = 5                # max number of cars to move during the night
RENTAL_REQUEST_FIRST_LOC = 3        # expectation of rental requests first location
RENTAL_REQUEST_SECOND_LOC = 4       # expectation of rental requests second location
RETURNS_FIRST_LOC = 3               # expectation of rental returns first location
RETURNS_SECOND_LOC = 3              # expectation of rental returns second location
RENTAL_CREDIT = 10                  # credit earned by a car
MOVE_CAR_COST = 2                   # cost of moving a car

DISCOUNT = 0.9

# All possible actions
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# Set upper bound for poisson distribution.
# If n is greater than this bound, then probability of getting n is truncated to 0.
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
poisson_cache = dict()

# lam: lambda should be < 10
def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

# state: [number of cars first location, number of cars second location]
# action: positive means move car from first to second location. Negative means the opposite
# stateValue: state value matrix
# constant_returned_cars: if true, model is simplified by having a constant amount of cars being returned each day
def expected_return(state, action, state_value, constant_returned_cars):
    # Init total return (credit)
    returns = 0

    # Cost of moving cars
    returns -= MOVE_CAR_COST * abs(action)

    # Moving cars
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    # Go through all possible rental requests
    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            # Probability of current combination of rental requests
            prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) *\
                poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            num_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_cars_second_loc = NUM_OF_CARS_SECOND_LOC

            # Valid rental requests should be less than number of cars
            valid_rental_first_loc = min(num_cars_first_loc, rental_request_first_loc)
            valid_rental_second_loc = min(num_cars_second_loc, rental_request_second_loc)

            # Get paid for renting (credits)
            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
            num_cars_first_loc -= valid_rental_first_loc
            num_cars_second_loc -= valid_rental_second_loc

            if constant_returned_cars:
                # Get returned cars, which can be used for renting tomorrow
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC
                num_cars_first_loc = min(num_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                num_cars_second_loc = min(num_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                returns += prob * (reward + DISCOUNT * state_value[num_cars_first_loc, num_cars_second_loc])
            else:
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(returned_cars_first_loc, RETURNS_FIRST_LOC) *\
                            poisson_probability(returned_cars_second_loc, RETURNS_SECOND_LOC)
                        num_cars_first_loc = min(num_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_cars_second_loc = min(num_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + DISCOUNT * state_value[num_cars_first_loc, num_cars_second_loc])
    return returns
# End of expected_return function definition

def figure_4_2(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # Policy evaluation in-place
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            print('max value change{}'.format(max_value_change))
            if max_value_change < 1e-4:
                break

        # Policy improvement
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in actions:
                    if(0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('policy {}'.format(iterations), fontsize=30)
            break

        iterations += 1

    plt.savefig('images/figure_4_2.png')
    plt.close()
# End figure_4_2 function definition


if __name__ == '__main__':
    figure_4_2()











































