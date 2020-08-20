"""
@author: polfr

Here we go through three ways of determining optimal policies with Monte Carlo methods for Blackjack.
See figure functions for descriptions of each specific methods

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

matplotlib.use('Agg')

# Actions: hit or stick
ACTION_HIT = 0
ACTION_STICK = 1
ACTIONS = [ACTION_HIT, ACTION_STICK]

# Policy for player
POLICY_PLAYER = np.zeros(22, dtype=int)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STICK
POLICY_PLAYER[21] = ACTION_STICK

# Function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]

# Function form of behaviour policy of player
def behaviour_policy_player(usable_ace_player, player_sum, dealer_card):
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STICK
    return ACTION_HIT


# Policy for the dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STICK

# Get a new card; game is played with infinite deck
def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# Get value of a card (11 for ace)
def card_value(card_id):
    return 11 if card_id == 1 else card_id

# Play a game of blackjack
# policy_player: specify the policy for the player
# initial_state: [usable ace, sum of player's cards, one card of dealer]
# initial_action: first action
def play(policy_player, initial_state=None, initial_action=None):
    # Get player status
    player_sum = 0
    player_trajectory = []
    usable_ace_player = False       # whether the player has a usable ace

    # Dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        # Generate random initial state
        while player_sum < 12:
            # If the player_sum is less than 12, we always hit
            card = get_card()
            player_sum += card_value(card)

            # If the sum is > 21, they may hold one or more aces
            if player_sum > 21:
                assert player_sum == 22  # Program will raise assertion error if this condition is not true
                # Last card must be an ace
                player_sum -= 10
            else:
                usable_ace_player |= (1 == card)

        # Initialize cards of dealer; dealer will show their first card
        dealer_card1 = get_card()
        dealer_card2 = get_card()
    else:
        # Use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # Initial game state
    state = [usable_ace_player, player_sum, dealer_card1]

    # Initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)

    # If the dealer's sum is larger than 21, he must hold two aces
    if dealer_sum > 21:
        assert dealer_sum == 22
        # Use one Ace as 1 rather than 11
        dealer_sum -= 10
    assert dealer_sum <= 21
    assert player_sum <= 21

    # Start the game
    # Player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # Get the action based on the current sum
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # Track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])

        if action == ACTION_STICK:
            break

        # If hit, get new card
        card = get_card()

        # Keep track of ace count. Usable_ace_player flag is insufficient alone as it cannot distinguish between
        # one or two aces
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)

        # If the player has a usable ace, use it as a 1 to avoid busting and continue
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1

        # Player busts
        if player_sum > 21:
            return state, -1, player_trajectory
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1)
    # End player's turn

    # Dealer's turn
    while True:
        # Get action based on current sum
        action = POLICY_DEALER[dealer_sum]

        if action == ACTION_STICK:
            break

        # If hit, get new card
        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)

        # If the dealer has a usable ace, use it as a 1 to avoid busting and continue
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1

        # Dealer busts
        if dealer_sum > 21:
            return state, 1, player_trajectory
        usable_ace_dealer = (ace_count == 1)
    # End dealer's turn

    # Compare sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory
# End play function definition


# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    states_usable_ace = np.zeros((10, 10))
    states_no_usable_ace = np.zeros((10, 10))

    # Initialize counts to 1 to avoid 0 division
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for i in tqdm(range(0, episodes)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward

    return states_usable_ace / states_no_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count
# End On-Policy function definition

# Monte Carlo with Exploring Starts
def monte_carlo_es(episodes):
    state_action_values = np.zeros((10, 10, 2, 2))      # (playerSum, dealerCard, usableAce, action)

    # Initialize counts to 1 to avoid 0 division
    state_action_pair_count = np.ones((10, 10, 2, 2))

    # Behaviour policy is greedy
    def behaviour_policy(usable_ace, player_sum, dealer_card):
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1

        # Get argmax of the average returns (s, a)
        values_ = state_action_values[player_sum, dealer_card, usable_ace, :] / \
                  state_action_pair_count[player_sum, dealer_card, usable_ace, :]

        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # Play for several episodes
    for episode in tqdm(range(episodes)):
        # For each episode, randomly initialize state and action
        initial_state = [bool(np.random.choice(([0, 1]))),
                         np.random.choice(range(12, 22)),
                         np.random.choice((range(1, 11)))]
        initial_action = np.random.choice(ACTIONS)
        current_policy = behaviour_policy if episode else target_policy_player
        _, reward, trajectory = play(current_policy, initial_state, initial_action)
        first_visit_check = set()

        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)

            # Update values of state-action pairs
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

    return state_action_values / state_action_pair_count
# End Exploring Starts function definition

# Monte Carlo Sample with Off-Policy
def monte_carlo_off_policy(episodes):
    initial_state = [True, 13, 2]

    rhos = []
    returns = []

    for i in range(0, episodes):
        _, reward, player_trajectory = play(behaviour_policy_player, initial_state=initial_state)

        # Get the importance ratio
        numerator = 1.0
        denominator = 1.0
        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            if action == target_policy_player(usable_ace, player_sum, dealer_card):
                denominator *= 0.5
            else:
                numerator = 0.0
                break
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)

    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns

    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)

    ordinary_sampling = weighted_returns / np.arange(1, episodes + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)

    return ordinary_sampling, weighted_sampling
# End Off-Policy function definition


def figure_5_1():
    states_usable_ace_1, states_no_usable_ace1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1, states_usable_ace_2,
              states_no_usable_ace1, states_no_usable_ace2]

    titles = ['Usable Ace, 10000 Episodes', 'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes', 'No Usable Ace, 500000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('Player sum', fontsize=30)
        fig.set_xlabel('Dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('images/figure_5_1.png')
    plt.close()
# End figure_5_1 function definition

def figure_5_2():
    state_action_values = monte_carlo_es(500000)
    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=1)

    # Get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    images = [action_usable_ace, state_value_usable_ace,
              action_no_usable_ace, state_value_no_usable_ace]
    titles = ['Optimal policy with usable Ace', 'Optimal value with usable Ace',
              'Optimal policy without usable Ace', 'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('Player sum', fontsize=30)
        fig.set_xlabel('Dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('images/figure_5_2.png')
    plt.close()
# End figure_5_2 function definition

def figure_5_3():
    true_value = -0.27726
    episodes = 10000
    runs = 100
    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)

    for i in tqdm(range(0, runs)):
        ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)

        # Get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)

    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(error_ordinary, label='Ordinary Importance Sampling')
    plt.plot(error_weighted, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend()

    plt.savefig('images/figure_5_3.png')
    plt.close()
# End figure_5_3 function definition


if __name__ == '__main__':
    # figure_5_1()
    # figure_5_2()
    figure_5_3()










































