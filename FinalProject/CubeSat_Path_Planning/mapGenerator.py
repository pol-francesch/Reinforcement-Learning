import numpy as np

# Class to get random racetracks
class Generator:
    # Constructor
    def __init__(self, ROWS, COLS, resolution, size):
        self.ROWS = ROWS
        self.COLS = COLS
        self.resolution = resolution
        self.size = size

    # Map is 2D numpy array coded as:
    # 0, 1, 2: valid map cells
    # -1: invalid map cell
    # 1: start area cells
    # 2: finish area cells
    # Method returns randomly generated map
    def generate_map(self):
        map = np.zeros((self.ROWS, self.COLS), dtype='int')

        # Set up start and finish zones.
        # These are 10 cm from top and bottom edge, and then randomly located on the x-axis (with a margin of 10 cm again)
        # Finish zone is 30x30 cm
        edge = int(0.1 / self.resolution)           # 10 cm in grid space
        random_start = np.random.randint(edge, self.COLS - edge)
        map[edge, random_start] = 1

        edge = int(3*edge)                          # 30 cm in grid space
        random_finish = np.random.randint(edge, self.COLS - edge)
        for i in range(self.ROWS - int(edge * 1.5), self.ROWS - edge//2, 1):
            for j in range(random_finish - edge//2, random_finish + edge//2, 1):
                map[i, j] = 2

        # Add 5-12 square obstacles of size 10 - 50 cm
        num_obs = np.random.randint(5, 13)
        count = 0
        min_size = int(0.1 / self.resolution)
        max_size = int(5*edge)

        while count < num_obs:
            random_size = np.random.randint(min_size, max_size)
            random_x = np.random.randint(random_size//2, self.COLS - random_size//2)
            random_y = np.random.randint(int(min_size*2), self.ROWS - max_size)     # Use max_size and min_size as convenient measurements.

            for i in range(random_y - random_size//2, random_y + random_size//2, 1):
                for j in range(random_x - random_size//2, random_x + random_size//2, 1):
                    # Make sure section is not already part of the final or start state
                    if map[i, j] == 0:
                        map[i, j] = -1

            count += 1

        return map
# End of Generator class definition
