import numpy as np
import matplotlib.pyplot as plt

# Implements the Monte Carlo Off-Policy Control algorithm
class Monte_Carlo_Control:
    # Helpful functions
    def evaluate_target_policy(self, env, agent):
        env.reset()
        state = env.start()
        self.data.episode['S'].append(state)
        rew = -1

        for t in range(100000):
            action = self.generate_behavioural_policy_action(state, agent.possible_actions(state))
            rew, state = env.step(state, action)

            if rew is None:
                break

        self.data.rewards.append(sum(self.data.episode['R'][1:]))

    def plot_rewards(self, image_name):
        ax, fig = plt.subplots(figsize=(30, 15))
        x = np.arange(1, len(self.data.rewards) + 1)
        plt.plot(x * 10, self.data.rewards, linewidth=0.5, color='#BB8FCE')
        plt.xlabel('Episode number', size=20)
        plt.ylabel('Reward', size=20)
        plt.title('Plot of Reward vs Episode Number', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig('images/' + image_name + '.png')
        plt.close()

    def save_your_work(self, directory='Racetrack_Data'):
        self.data.save_Q_vals(directory)
        self.data.save_C_vals(directory)
        self.data.save_pi(directory)
        self.data.save_rewards(directory)

    def determine_probability_behaviour(self, state, action, possible_actions):
        best_action = self.data.pi[tuple(state)]
        num_actions = len(possible_actions)

        if best_action in possible_actions:
            if action == best_action:
                prob = 1 - self.data.epsilon + self.data.epsilon / num_actions
            else:
                prob = self.data.epsilon / num_actions
        else:
            prob = 1 / num_actions

        self.data.episode['probs'].append(prob)

    # Takes state and possible actions, and determines best action using target policy
    def generate_target_policy_action(self, state, possible_actions):
        if self.data.pi[tuple(state)] in possible_actions:
            action = self.data.pi[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        return action

    # Takes state and possible actions, and determines best action using behaviour policy
    # Note: behavioural policy is epsilon-greedy pi policy
    def generate_behavioural_policy_action(self, state, possible_actions):
        if np.random.rand() > self.data.epsilon and self.data.pi[tuple(state)] in possible_actions:
            action = self.data.pi[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        self.determine_probability_behaviour(state, action, possible_actions)

        return action

    # Constructor
    # Initialize for all s in S and a in A(s):
    # data.Q(s,a) <- arbitrary (done in Data)
    # data.C(s,a) <- 0 (done in Data)
    # pi(s) <= argmax_a Q(s,a)
    # (with ties broken consistently)
    def __init__(self, data, ROWS, COLS):
        self.data = data
        self.ROWS = ROWS
        self.COLS = COLS
        # We want to create a Pi matrix which has values for every possible state.
        # That means that for every combination of position, velocity and angle, the Pi matrix has a slot.
        for i in range(self.ROWS):
            for j in range(self.COLS):
                if self.data.racetrack[i, j] != 1:
                    for k in range(-4, 5):
                        for l in range(-4, 5):
                            for t in range(8):
                                for n in range(6):
                                    self.data.pi[i, j, k, l, t, n] = np.argmax(self.data.Q_vals[i, j, k, l, t, n])

    # Performs Monte Carlo control using episode list [ S0 , A0 , R1, . . . , ST −1 , AT −1, RT , ST ]
    # G <- 0
    # W <- 1
    # For t = T-1, T - 2, ... down to 0:
    #       G <- y*G + R_T + 1
    #       C(St, At) <- C(St, At) + W
    #       Q(St, At) <- Q(St, At) + (W / C(St, At)) * [G - Q(St, At)]
    #       pi(St) <- argmax_a Q(St, At) (with ties broken consistently)
    #       If At != pi(St) then exit for loop
    #       W <- W * (1 / b * (At | St))
    def control(self, env, agent):
        env.reset()
        state = env.start()
        self.data.episode['S'].append(state)
        rew = -1

        for t in range(100000):
            action = self.generate_behavioural_policy_action(state, agent.possible_actions(state))
            rew, state = env.step(state, action)

            if rew is None:
                break

        G = 0
        W = 1
        T = env.step_count

        for t in range(T - 1, -1, -1):
            G = self.data.gamma * G + self.data.episode['R'][t+1]
            S_t = tuple(self.data.episode['S'][t])
            A_t = self.data.episode['A'][t]

            S_list = list(S_t)
            S_list.append(A_t)
            SA = tuple(S_list)

            self.data.C_vals[SA] += W
            self.data.Q_vals[SA] += (W * (G - self.data.Q_vals[SA])) / (self.data.C_vals[SA])
            self.data.pi[S_t] = np.argmax(self.data.Q_vals[S_t])
            if A_t != self.data.pi[S_t]:
                break
            W /= self.data.episode['probs'][t]
# End Monte Carlo class definition
