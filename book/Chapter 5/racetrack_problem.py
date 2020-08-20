"""
@author: polfr

Solving the racetrack in reinforcement learning using Monte Carlo Off-Policy control.
Following the medium post here:
https://towardsdatascience.com/solving-racetrack-in-reinforcement-learning-using-monte-carlo-control-bdee2aa4f04e
" Solving Racetrack in Reinforcement Learning using Monte Carlo Control " , by Aditya Rastogi

Equivalent to exercise 5.12 in Sutton & Barto book

"""

import numpy as np
import pygame
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

ROWS = 200
COLS = 100

# Class to get random racetracks
class Generator:
    # Helpful functions
    def widen_hole_transformation(self, racetrack, start_cell, end_cell):

        delta = 1
        while True:
            if (start_cell[1] < delta) or (start_cell[0] < delta):
                racetrack[0: end_cell[0], 0: end_cell[1]] = -1
                break

            if (end_cell[1] + delta > COLS) or (end_cell[0] + delta > ROWS):
                racetrack[start_cell[0]: ROWS, start_cell[1]: COLS] = -1
                break

            delta += 1

        return racetrack

    # Returns fraction of valid cells in racetrack
    def calculate_valid_fraction(self, racetrack):
        return len(racetrack[racetrack == 0])/(ROWS * COLS)

    # Returns racetrack with marked finish states
    def mark_finish_states(self, racetrack):
        last_col = racetrack[0:ROWS, COLS - 1]
        last_col[last_col == 0] = 2
        return racetrack

    # Returns racetrack with marked start states
    def mark_start_states(self, racetrack):
        last_row = racetrack[ROWS - 1, 0: COLS]
        last_row[last_row == 0] = 1
        return racetrack

    # Constructor
    def __init__(self):
        pass

    # Racetrack is 2D numpy array coded as:
    # 0, 1, 2: valid racetrack cells
    # -1: invalid racetrack cell
    # 1: start line cells
    # 2: finish line cells
    # Method returns randomly generated racetrack
    def generate_racetrack(self):
        racetrack = np.zeros((ROWS, COLS), dtype='int')

        frac = 1
        while frac > 0.5:
            # Transformation
            random_cell = np.random.randint((ROWS, COLS))
            random_hole_dims = np.random.randint((ROWS//4, COLS//4))
            start_cell = np.array([max(0, x - y//2) for x, y in zip(random_cell, random_hole_dims)])
            end_cell = np.array([min(z, x+y) for x, y, z in zip(start_cell, random_hole_dims, [ROWS, COLS])])

            # Apply transformation
            racetrack = self.widen_hole_transformation(racetrack, start_cell, end_cell)
            frac = self.calculate_valid_fraction(racetrack)

        racetrack = self.mark_start_states(racetrack)
        racetrack = self.mark_finish_states(racetrack)

        return racetrack
# End of Generator class definition


class Data:
    # Helpful functions
    def get_start_line(self):
        self.start_line = np.array([np.array([ROWS - 1, j]) for j in range(COLS) if self.racetrack[ROWS - 1, j] == 1])

    def get_finish_line(self):
        self.finish_line = np.array([np.array([i, COLS - 1]) for i in range(ROWS) if self.racetrack[i, COLS - 1] == 2])

    # Constructor
    def __init__(self):
        # racetrack: 2D numpy array
        # Q(s, a): 5D numpy array
        # C(s, a): 5D numpy array
        # pi: target policy
        # start_line: set of start states
        # finish_line: set of finish states
        self.load_racetrack()
        self.get_start_line()
        self.get_finish_line()
        self.load_Q_vals()
        self.load_C_vals()
        self.load_pi()
        self.load_rewards()
        self.epsilon = 0.1
        self.gamma = 1
        self.episode = dict({'S': [], 'A': [], 'probs': [], 'R': [None]})

    # Methods to get and save to files
    def save_rewards(self, directory='Racetrack_Data'):
        self.rewards = np.array(self.rewards)
        filename = directory + '/rewards.npy'
        np.save(filename, self.rewards)
        self.rewards = list(self.rewards)

    def load_rewards(self, directory='Racetrack_Data'):
        filename = directory + '/rewards.npy'
        self.rewards = list(np.load(filename))

    def save_pi(self, directory='Racetrack_Data'):
        filename = directory + '/pi.npy'
        np.save(filename, self.pi)

    def load_pi(self, directory='Racetrack_Data'):
        filename = directory + '/pi.npy'
        self.pi = np.load(filename)

    def save_C_vals(self, directory='Racetrack_Data'):
        filename = directory + '/C_vals.npy'
        np.save(filename, self.C_vals)

    def load_C_vals(self, directory='Racetrack_Data'):
        filename = directory + '/C_vals.npy'
        self.C_vals = np.load(filename)

    def save_Q_vals(self, directory='Racetrack_Data'):
        filename = directory + '/Q_vals.npy'
        np.save(filename, self.Q_vals)

    def load_Q_vals(self, directory='Racetrack_Data'):
        filename = directory + '/Q_vals.npy'
        self.Q_vals = np.load(filename)

    def save_racetrack(self, directory='Racetrack_Data'):
        filename = directory + '/racetrack.npy'
        np.save(filename, self.racetrack)

    def load_racetrack(self, directory='Racetrack_Data'):
        filename = directory + '/racetrack.npy'
        self.racetrack = np.load(filename)


class Environment:
    # Helpful functions
    # Get new state after applying the action on the given state
    # Assume that the car keeps on moving with the current velocity, and then action is applied to change the velocity
    def get_new_state(self, state, action):
        new_state = state.copy()
        new_state[0] = state[0] - state[2]
        new_state[1] = state[1] + state[3]
        new_state[2] = state[2] + action[0]
        new_state[3] = state[3] + action[1]
        return new_state

    # Returns value uniform randomly from NUMPY_ARR
    # NUMPY_ARR should be 1D
    def select_randomly(self, NUMPY_ARR):
        return np.random.choice(NUMPY_ARR)

    # Returns NUMPY_ARR after making all elements 0
    # Class to create the environment
    def set_zero(NUMPY_ARR):
        NUMPY_ARR[:] = 0
        return NUMPY_ARR

    # Return true if car crosses the finish line, false otherwise
    def is_finish_line_crossed(self, state, action):
        new_state = self.get_new_state(state, action)
        old_cell, new_cell = state[0:2], new_state[0:2]

        # New cell's row index will be less
        rows = np.array(range(new_cell[0], old_cell[0] + 1))
        cols = np.array(range(old_cell[1], new_cell[1] + 1))
        fin = set([tuple(x) for x in self.data.finish_line])
        row_col_matrix = [(x, y) for x in rows for y in cols]
        intersect = [x for x in row_col_matrix if x in fin]

        return len(intersect) > 0

    # Returns true if car goes out of track or if action is taken on state, false otherwise
    def is_out_of_track(self, state, action):
        new_state = self.get_new_state(state, action)
        old_cell, new_cell = state[0:2], new_state[0:2]

        if new_cell[0] < 0 or new_cell[0] >= ROWS or new_cell[1] < 0 or new_cell[1] >= COLS:
            return True
        else:
            return self.data.racetrack[tuple(new_cell)] == -1

    # Constructor
    # Initialize step count to be 0
    def __init__(self, data, gen):
        self.data = data
        self.gen = gen
        self.step_count = 0

    # Member functions

    def reset(self):
        self.data.episode = dict({'S': [], 'A': [], 'probs': [], 'R': [None]})
        self.step_count = 0

    # Makes velocity of car 0
    # Return randomly selected start state
    def start(self):
        state = np.zeros(4, dtype='int')
        state[0] = ROWS - 1
        state[1] = self.select_randomly(self.data.start_line[:, 1])
        # state 2 and 3 are already 0
        return state

    # Returns reward and new state when action is taken on state
    # Check 2 cases maintaining the order:
    # 1. car finished by crossing the finish line
    # 2. car goes out of track
    # Ends episode by returning the reward as None and state as usual
    def step(self, state, action):
        self.data.episode['A'].append(action)
        reward = -1

        if self.is_finish_line_crossed(state, action):
            new_state = self.get_new_state(state, action)

            self.data.episode['R'].append(reward)
            self.data.episode['S'].append(new_state)
            self.step_count += 1

            return None, new_state
        elif self.is_out_of_track(state, action):
            new_state = self.start()
        else:
            new_state = self.get_new_state(state, action)

        self.data.episode['R'].append(reward)
        self.data.episode['S'].append(new_state)
        self.step_count += 1

        return reward, new_state
# End of Environment class definition


class Agent:
    # Helpful functions

    # Performs two task which can be split up
    # Universe of actions: alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
    #
    # Use constraints to filter out invalid actions given the velocity
    # 0 <= v_x < 5
    # 0 <= v_y < 5
    # v_x and v_y cannot both be made 0 (you cannot take an action which would make them zero simultaneously
    # Returns list of possible actions given the velocity
    def possible_actions(self, velocity):
        # List of possible actions
        alpha = [(-1, -1), (-1, 0), (0, -1), (-1, 1), (0, 0), (1, -1), (0, 1), (1, 0), (1, 1)]
        alpha = [np.array(x) for x in alpha]

        # List of possible actions without going out of bounds of velocity
        beta = []
        for i, x in zip(range(9), alpha):
            new_vel = np.add(velocity, x)
            if (new_vel[0] < 5) and (new_vel[0] >= 0) and (new_vel[1] < 5) and (new_vel[1] >= 0) and ~(new_vel[0] == 0 and new_vel[1] == 0):
                beta.append(i)
        beta = np.array(beta)

        return beta

    # Returns location of chosen action in alpha
    def map_to_1D(self, action):
        alpha = [(-1, -1), (-1, 0), (0, -1), (-1, 1), (0, 0), (1, -1), (0, 1), (1, 0), (1, 1)]
        for i, x in zip(range(9), alpha):
            if action[0] == x[0] and action[1] == x[1]:
                return i

    # Returns value of action from location in alpha
    def map_to_2D(self, action):
        alpha = [(-1, -1), (-1, 0), (0, -1), (-1, 1), (0, 0), (1, -1), (0, 1), (1, 0), (1, 1)]
        return alpha[action]

    # Constructor
    def __init__(self):
        pass

    # Returns action given state using the policy
    def get_action(self, state, policy):
        return self.map_to_2D(policy(state, self.possible_actions(state[2:4])))
# End of Agent class definition


# Visualizes racetrack and the agent using a window
class Visualizer:
    # Helpful functions
    def visualize_episode(data):
        for i in range(data.episode['S']):
            vis.visualize_racetrack(i)

    # Creates window and assigns self.display variable
    def create_window(self):
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racetrack")

    # Only runs at the beginning
    def setup(self):
        self.width = COLS * self.cell_edge
        self.height = ROWS * self.cell_edge
        self.create_window()
        self.window = True

    def close_window(self):
        self.window = False
        # pygame.quit()

    def draw(self, state=np.array([])):
        self.display.fill(0)
        for i in range(ROWS):
            for j in range(COLS):
                if self.data.racetrack[i, j] != -1:
                    if self.data.racetrack[i, j] == 0:
                        color = (255, 0, 0)
                    elif self.data.racetrack[i, j] == 1:
                        color = (255, 255, 0)
                    elif self.data.racetrack[i, j] == 2:
                        color = (0, 255, 0)
                    pygame.draw.rect(self.display, color, ((j * self.cell_edge, i * self.cell_edge),
                                                           (self.cell_edge, self.cell_edge)), 1)
        if len(state) > 0:
            pygame.draw.rect(self.display, (0, 0, 255), ((state[1]*self.cell_edge, state[0]*self.cell_edge),
                                                         (self.cell_edge, self.cell_edge)), 1)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
                self.close_window()
                return 'stop'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.loop = False

        return None

    # Draws racetrack in pygame window
    def visualize_racetrack(self, state=np.array([])):
        if not self.window:
            self.setup()
        self.loop = True
        while self.loop:
            ret = self.draw(state)
            if ret is not None:
                return ret

    def single_frame(self, state=np.array([])):
        self.display.fill(0)
        for i in range(ROWS):
            for j in range(COLS):
                if self.data.racetrack[i, j] != -1:
                    if self.data.racetrack[i, j] == 0:
                        color = (255, 0, 0)
                    elif self.data.racetrack[i, j] == 1:
                        color = (255, 255, 0)
                    elif self.data.racetrack[i, j] == 2:
                        color = (0, 255, 0)
                    pygame.draw.rect(self.display, color, ((j * self.cell_edge, i * self.cell_edge),
                                                           (self.cell_edge, self.cell_edge)), 1)
        if len(state) > 0:
            pygame.draw.rect(self.display, (0, 0, 255), ((state[1]*self.cell_edge, state[0]*self.cell_edge),
                                                         (self.cell_edge, self.cell_edge)), 1)
        pygame.display.update()

    # Saves racetrack in file
    def save_racetrack(self, filename, state=np.array([])):
        self.single_frame(state)
        pygame.image.save(pygame.display.get_surface(), filename)


    # Constructor
    def __init__(self, data, cell_edge):
        self.data = data
        self.cell_edge = cell_edge
        self.window = False
# End of Visualizer class definition


# Implements the Monte Carlo Off-Policy Control algorithm
class Monte_Carlo_Control:
    # Helpful functions
    def evaluate_target_policy(self):
        env.reset()
        state = env.start()
        self.data.episode['S'].append(state)
        rew = -1
        while rew is not None:
            action = agent.get_action(state, self.generate_target_policy_action)
            rew, state = env.step(state, action)

        self.data.rewards.append(sum(self.data.episode['R'][1:]))

    def plot_rewards(self):
        ax, fig = plt.subplots(figsize=(30, 15))
        x = np.arange(1, len(self.data.rewards) + 1)
        plt.plot(x * 10, self.data.rewards, linewidth=0.5, color='#BB8FCE')
        plt.xlabel('Episode number', size=20)
        plt.ylabel('Reward', size=20)
        plt.title('Plot of Reward vs Episode Number', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig('images/reward_graph.png')
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
    def __init__(self, data):
        self.data = data
        for i in range(ROWS):
            for j in range(COLS):
                if self.data.racetrack[i, j] != 1:
                    for k in range(5):
                        for l in range(5):
                            self.data.pi[i, j, k, l] = np.argmax(self.data.Q_vals[i, j, k, l])

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
        while rew is not None:
            action = agent.get_action(state, self.generate_behavioural_policy_action)
            rew, state = env.step(state, action)

        G = 0
        W = 1
        T = env.step_count

        for t in range(T - 1, -1, -1):
            G = self.data.gamma * G + self.data.episode['R'][t+1]
            S_t = tuple(self.data.episode['S'][t])
            A_t = agent.map_to_1D(self.data.episode['A'][t])

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


# Helpful Functions
def train_agent(plot=False, episodes=50000):
    for i in tqdm(range(0, episodes)):
        mcc.control(env, agent)

        if i % 10 == 9:
            mcc.evaluate_target_policy()
        if i % 100 == 99:
            mcc.save_your_work()
            if plot:
                mcc.plot_rewards()

# You probably shouldn't run see_path and get_gif within the same compilation
def see_path():
    for state in data.episode['S']:
        if vis.visualize_racetrack(state) == 'stop':
            break
    vis.close_window()

def get_gif(gif_name):
    filenames = []
    for i in range(len(data.episode['S'])):
        filename = 'images_racetrack/' + str(i) + '.png'
        filenames.append(filename)
        vis.save_racetrack(filename=filename, state=data.episode['S'][i])

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_name + '.gif', images, duration=0.25)
    vis.close_window()

# Different ways to run this code. The first uses a racetrack with a nice solution. The second uses a random one
def run_old_data(plot=False):
    vis.visualize_racetrack()
    train_agent(plot)

    ch = 50
    S = sum(data.rewards[: ch]) / ch

    R = []

    for i in tqdm(range(ch, len(data.rewards))):
        R.append(S)
        S *= ch
        S += data.rewards[i]
        S -= data.rewards[i - ch]
        S /= ch

    if plot:
        ax, fig = plt.subplots(figsize=(60, 30))
        x = np.arange(1, len(R) + 1)
        plt.plot(x * 10, R, linewidth=1, color='#BB8FCE')
        plt.xlabel('Episode number', size=40)
        plt.ylabel('Reward', size=40)
        plt.title('Plot of Reward vs Episode Number', size=40)
        plt.xticks(size=40)
        plt.yticks(size=40)
        plt.savefig('images/reward_graph_2.png')
        plt.close()

# TODO:: Racetrack selection must be better. We should be able to loop until the user says they like the racetrack.
def run_random_racetrack(plot=False, use_old_racetrack=True):
    # Get a new racetrack if we want
    if use_old_racetrack:
        data.load_racetrack(directory='Random_Racetrack_Data')
    else:
        data.racetrack = gen.generate_racetrack()

    # Visualize racetrack
    vis.visualize_racetrack()
    data.save_racetrack(directory='Random_Racetrack_Data')

    # Set new algorithm arrays
    data.Q_vals = np.random.rand(ROWS, COLS, 5, 5, 9) * 400 - 500
    data.rewards = []
    data.C_vals = np.zeros((ROWS, COLS, 5, 5, 9))
    data.pi = np.zeros((ROWS, COLS, 5, 5), dtype='int')

    # Save arrays
    data.save_Q_vals(directory='Random_Racetrack_Data')
    data.save_C_vals(directory='Random_Racetrack_Data')
    data.save_rewards(directory='Random_Racetrack_Data')
    data.save_pi(directory='Random_Racetrack_Data')

    # Train agent
    train_agent(plot=plot)


if __name__ == '__main__':
    # Set up our RL
    data = Data()
    gen = Generator()
    env = Environment(data, gen)
    mcc = Monte_Carlo_Control(data)
    vis = Visualizer(data, cell_edge=6)         # If you cannot see the whole "game" lower cell_edge value.
    agent = Agent()

    # Run
    # run_old_data()
    run_random_racetrack()

    get_gif(gif_name='images/test_old')

'''

count = 0

rew = -1
while rew is not None:
    action = agent.get_action(state, mcc.generate_target_policy_action)
    rew, state = env.step(state, action)
    count += 1

print('Done training agent, ran for ' + str(count) + ' loops.')

'''









































