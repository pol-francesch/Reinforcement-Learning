import numpy as np

# Used to hold data types in the functions.
# Used to write and read from files to store progress.
class Data:
    # Helpful functions
    def get_start_line(self):
        for i in range(self.ROWS):
            for j in range(self.COLS):
                if self.racetrack[i][j] == 1:
                    self.start_line = [i, j]
                    break

    def get_finish_line(self):
        self.finish_line = []
        for i in range(self.ROWS):
            for j in range(self.COLS):
                if self.racetrack[i][j] == 2:
                    self.finish_line.append([i, j])

    # Constructor
    def __init__(self, ROWS, COLS, load_data=False, epsilon=0.1):
        # map: 2D numpy array
        # Q(s, a): 5D numpy array
        # C(s, a): 5D numpy array
        # pi: target policy
        # start_line: set of start states
        # finish_line: set of finish states
        self.ROWS = ROWS
        self.COLS = COLS
        self.epsilon = epsilon
        self.gamma = 1
        self.episode = dict({'S': [], 'A': [], 'probs': [], 'R': [None]})

        if load_data:
            self.load_old_data()
        else:
            self.reset_data()

    def load_old_data(self):
        self.load_racetrack()
        self.get_start_line()
        self.get_finish_line()
        self.load_Q_vals()
        self.load_C_vals()
        self.load_pi()
        self.load_rewards()

    def reset_data(self):
        self.Q_vals = np.random.rand(self.ROWS, self.COLS, 9, 9, 8, 6, 8) * 400 - 500
        self.rewards = []
        self.C_vals = np.zeros((self.ROWS, self.COLS, 9, 9, 8, 6, 8))
        self.pi = np.zeros((self.ROWS, self.COLS, 9, 9, 8, 6), dtype='int')
        self.racetrack = []

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
        filename = directory + '/map.npy'
        np.save(filename, self.racetrack)

    def load_racetrack(self, directory='Racetrack_Data'):
        filename = directory + '/map.npy'
        self.racetrack = np.load(filename)
