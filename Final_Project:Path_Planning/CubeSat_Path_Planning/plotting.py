import numpy as np
import pygame

# Plots racetrack and the agent using a window
class Plotter:
    # Helpful functions
    def visualize_episode(data, vis):
        for i in range(data.episode['S']):
            vis.visualize_racetrack(i)

    # Creates window and assigns self.display variable
    def create_window(self):
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Map")

    # Only runs at the beginning
    def setup(self):
        self.width = self.COLS * self.cell_edge
        self.height = self.ROWS * self.cell_edge
        self.create_window()
        self.window = True

    def close_window(self):
        self.window = False
        # pygame.quit()

    def draw(self, state=np.array([])):
        self.display.fill(0)
        for i in range(self.ROWS):
            for j in range(self.COLS):
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
        for i in range(self.ROWS):
            for j in range(self.COLS):
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
    def __init__(self, data, cell_edge, ROWS, COLS):
        self.data = data
        self.cell_edge = cell_edge
        self.ROWS = ROWS
        self.COLS = COLS
        self.window = False
# End of Plotter class definition
