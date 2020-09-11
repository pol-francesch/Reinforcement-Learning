import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

from mapGenerator import Generator
from data import Data
from environment import Environment
from agent import Agent
from plotting import Plotter
from MonteCarlo import Monte_Carlo_Control

# Helpful Functions
def train_agent(mcc, env, agent, data, plot=False, episodes=50000, image_name='reward_graph'):
    print("Starting training...")
    for i in tqdm(range(0, episodes)):
        mcc.control(env, agent)

        if i % 10 == 9:
            mcc.evaluate_target_policy(env, agent)
        if i % 10000 == 9999:
            mcc.save_your_work()
        if plot and i % 100 == 99:
            mcc.plot_rewards(image_name=image_name)

    print("Creating smooth plot...")
    # We can also plot a cleaned up version of the plot that is more smoothed out
    if plot:
        ch = 50
        S = sum(data.rewards[: ch]) / ch

        R = []

        for i in tqdm(range(ch, len(data.rewards))):
            R.append(S)
            S *= ch
            S += data.rewards[i]
            S -= data.rewards[i - ch]
            S /= ch

        ax, fig = plt.subplots(figsize=(60, 30))
        x = np.arange(1, len(R) + 1)
        plt.plot(x * 10, R, linewidth=1, color='#BB8FCE')
        plt.xlabel('Episode number', size=40)
        plt.ylabel('Reward', size=40)
        plt.title('Plot of Reward vs Episode Number', size=40)
        plt.xticks(size=40)
        plt.yticks(size=40)
        plt.savefig('images/' + image_name + '_clean_plot.png')
        plt.close()

# You probably shouldn't run see_path and get_gif within the same compilation
def see_path(data, vis):
    for state in data.episode['S']:
        if vis.visualize_racetrack(state) == 'stop':
            break
    vis.close_window()

def get_gif(gif_name, data, vis):
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

# Functions used to test functionality
def test_map_making():
    data = Data(ROWS, COLS, load_data=False)
    gen = Generator(ROWS, COLS, resolution, size)
    vis = Plotter(data, cell_edge, ROWS, COLS)

    data.racetrack = gen.generate_map()

    vis.visualize_racetrack()

def sensitivity_analysis():
    # Sensitivity analysis settings for epsilon=0.1
    directory = 'Sensitivity_Analysis'
    image_name = 'epsilon=0.1'
    epsilon = 0.1

    # Set up RL
    data = Data(ROWS, COLS, load_data=False, epsilon=epsilon)
    gen = Generator(ROWS, COLS, resolution, size)
    vis = Plotter(data, cell_edge, ROWS, COLS)

    # Display maps until we get one we like
    while True:
        data.racetrack = gen.generate_map()
        data.get_start_line()
        data.get_finish_line()

        vis.visualize_racetrack()
        response = input("Is this map suitable for our test? (y/n): ")

        if response == 'y':
            break

    # Finish setting up RL
    env = Environment(data, gen, ROWS, COLS)
    mcc = Monte_Carlo_Control(data, ROWS, COLS)
    agent = Agent()

    print("Saving data")
    # Save arrays
    data.save_racetrack(directory=directory)
    data.save_Q_vals(directory=directory)
    data.save_C_vals(directory=directory)
    data.save_rewards(directory=directory)
    data.save_pi(directory=directory)

    # Train agent
    train_agent(mcc, env, agent, data, plot=True, image_name=image_name)

    # Sensitivity analysis settings for epsilon=0.01
    image_name = 'epsilon=0.01'
    epsilon = 0.01
    gif_name = 'epsilon=0.01'

    # Set up RL
    data2 = Data(ROWS, COLS, load_data=False, epsilon=epsilon)
    vis = Plotter(data2, cell_edge, ROWS, COLS)

    # Get old map
    data2.racetrack = data.racetrack
    data2.get_start_line()
    data2.get_finish_line()

    # Finish setting up RL
    env = Environment(data2, gen, ROWS, COLS)
    mcc = Monte_Carlo_Control(data2, ROWS, COLS)

    print("Saving data")
    # Save arrays
    data.save_racetrack(directory=directory)
    data.save_Q_vals(directory=directory)
    data.save_C_vals(directory=directory)
    data.save_rewards(directory=directory)
    data.save_pi(directory=directory)

    # Train agent
    train_agent(mcc, env, agent, data2, plot=True, image_name=image_name)

    get_gif(gif_name, data2, vis)

def run_premade_map():
    # Set up RL
    data = Data(ROWS, COLS, load_data=False, epsilon=0.25)
    gen = Generator(ROWS, COLS, resolution, size)
    vis = Plotter(data, cell_edge, ROWS, COLS)

    data.load_racetrack(directory='Sensitivity_Analysis')
    data.get_start_line()
    data.get_finish_line()
    vis.visualize_racetrack()

    # Finish setting up RL
    env = Environment(data, gen, ROWS, COLS)
    mcc = Monte_Carlo_Control(data, ROWS, COLS)
    agent = Agent()

    print("Saving data")
    # Save arrays
    data.save_racetrack(directory='Sensitivity_Analysis_2')
    data.save_Q_vals(directory='Sensitivity_Analysis_2')
    data.save_C_vals(directory='Sensitivity_Analysis_2')
    data.save_rewards(directory='Sensitivity_Analysis_2')
    data.save_pi(directory='Sensitivity_Analysis_2')

    # Train agent
    train_agent(mcc, env, agent, data, plot=True, image_name='sensitivity_epsilon=0.25')

    get_gif('epsilon=0.25', data, vis)

# Function which runs everything
def run_random_map(plot=False, directory='Random_Map_Data', image_name='random_reward_graph', epsilon=0.1):
    # Set up RL
    data = Data(ROWS, COLS, load_data=False, epsilon=epsilon)
    gen = Generator(ROWS, COLS, resolution, size)
    vis = Plotter(data, cell_edge, ROWS, COLS)

    # TODO:: Must be able to plot map without needing space bar
    # Display maps until we get one we like
    while True:
        data.racetrack = gen.generate_map()
        data.get_start_line()
        data.get_finish_line()

        vis.visualize_racetrack()
        response = input("Is this map suitable for our test? (y/n): ")

        if response == 'y':
            break

    # Finish setting up RL
    env = Environment(data, gen, ROWS, COLS)
    mcc = Monte_Carlo_Control(data, ROWS, COLS)
    agent = Agent()

    print("Saving data")
    # Save arrays
    data.save_racetrack(directory=directory)
    data.save_Q_vals(directory=directory)
    data.save_C_vals(directory=directory)
    data.save_rewards(directory=directory)
    data.save_pi(directory=directory)

    # Train agent
    train_agent(mcc, env, agent, data, plot=plot, image_name=image_name)

    if plot:
        get_gif(image_name, data, vis)


if __name__ == '__main__':
    # Options which can easily be changed are below:
    size = [2, 4]                           # size of the map, m. (Note it is expected that x > y where size = [x, y])
    resolution = 0.05                       # resolution of the map, cell/m [ie if resolution = 0.1, there will be 10 cells per meter]
    cell_edge = 6                           # Used for the plotter. If when plotting the track not all of it shows up, lower this.
    generate_plot = True                    # Whether to generate a new plot
    random_data_directory = 'Random_Map_Data'           # Code will save all of the data to this directory. Useful if
                                                        # You want to run multiple random racetracks and keep data
    image_name = 'random_reward_graph'      # Can be renamed to save multiple images. If not, old images are overwritten
    epsilon = 0.1                           # Probability that the behavioural policy will choose a random action
    # Calculate rows and columns for array representation of map
    COLS = int(size[0] / resolution)        # Represents x direction
    ROWS = int(size[1] / resolution)        # Represents y direction

    run_random_map(plot=generate_plot, directory=random_data_directory, image_name=image_name, epsilon=epsilon)
