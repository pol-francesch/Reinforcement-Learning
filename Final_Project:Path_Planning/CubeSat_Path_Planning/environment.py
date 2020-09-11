import numpy as np

def get_thrust_on_theta(angle):
    thrust = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

    return thrust[angle]

class Environment:
    # Helpful functions
    # Get new state after applying the action on the given state
    # Assume that the car keeps on moving with the current velocity, and then action is applied to change the velocity

    # Definition of state
    # state[0] = x
    # state[1] = y
    # state[2] = dx/dt
    # state[3] = dy/dt
    # state[4] = theta
    # state[5] = information on thrusters and reaction wheel.
    # state[5] = 0: Nothing
    # state[5] = 1: front thrusters on
    # state[5] = 2: back thrusters on
    # state[5] = 3 or 4: reaction wheel spin (right or left)

    # Definition of theta
    # theta = 0 -> 0   deg
    # theta = 1 -> pi/4  deg
    # theta = 2 -> pi/2  deg
    # theta = 3 -> 3pi/4 deg
    # theta = 4 -> pi deg
    # theta = 5 -> -3pi/4 deg
    # theta = 6 -> -pi/2 deg
    # theta = 7 -> -pi/4 deg
    def get_new_state(self, state, action):
        new_state = state.copy()

        # Update action states
        if action == 0:
            # Turn front thrusters on
            new_state[5] = 1
        elif action == 1:
            # Turn front thrusters off
            new_state[5] = 0
        elif action == 2:
            # Turn back thrusters on
            new_state[5] = 2
        elif action == 3:
            # Turn back thrusters off
            new_state[5] = 0
        elif action == 4:
            # Turn on reaction wheel (spin right)
            new_state[5] = 3
        elif action == 5:
            # Turn on reaction wheel (spin left)
            new_state[5] = 4
        elif action == 6:
            # Turn off reaction wheel
            new_state[5] = 0

        # Update velocity and angle
        if new_state[5] == 1:
            # Thruster nozzle is aligned with theta
            delta_v = get_thrust_on_theta(new_state[4])
            new_state[2] -= delta_v[0]
            new_state[3] -= delta_v[1]
        if new_state[5] == 2:
            # Thruster nozzle is aligned with -theta
            delta_v = get_thrust_on_theta(new_state[4])
            new_state[2] += delta_v[0]
            new_state[3] += delta_v[1]

        if new_state[5] == 3:
            if new_state[4] == 7:
                new_state[4] = 0
            else:
                new_state[4] += 1
        if new_state[5] == 4:
            if new_state[4] == 0:
                new_state[4] = 7
            else:
                new_state[4] -= 1

        # Update position using old velocity
        new_state[0] = state[0] - state[2]
        new_state[1] = state[1] + state[3]

        return new_state

    # Returns true if the CubeSat will cross into the finish area
    def is_at_finish_area(self, state, action):
        new_state = self.get_new_state(state, action)
        new_pos = new_state[0:2]

        for finish_pos in self.data.finish_line:
            if new_pos[0] == finish_pos[0] and new_pos[1] == finish_pos[1]:
                return True

        return False

    # Returns true if car goes out of track or if action is taken on state, false otherwise
    def is_out_of_track(self, state, action):
        new_state = self.get_new_state(state, action)
        old_cell, new_cell = state[0:2], new_state[0:2]

        if new_cell[0] < 0 or new_cell[0] >= self.ROWS or new_cell[1] < 0 or new_cell[1] >= self.COLS:
            return True
        else:
            return self.data.racetrack[tuple(new_cell)] == -1

    # Constructor
    # Initialize step count to be 0
    def __init__(self, data, gen, ROWS, COLS):
        self.data = data
        self.gen = gen
        self.step_count = 0
        self.ROWS = ROWS
        self.COLS = COLS

    # Member functions

    def reset(self):
        self.data.episode = dict({'S': [], 'A': [], 'probs': [], 'R': [None]})
        self.step_count = 0

    # Makes velocity of car 0
    # Return start state
    def start(self):
        # state[0] = x
        # state[1] = y
        # state[2] = dx/dt
        # state[3] = dy/dt
        # state[4] = theta
        # state[5] = information on thrusters and reaction wheel.
        # state[5] = 0: Nothing
        # state[5] = 1: front thrusters on
        # state[5] = 2: back thrusters on
        # state[5] = 3 or 4: reaction wheel spin (right or left)
        state = np.zeros(6, dtype='int')
        start_point = self.data.start_line
        state[0] = start_point[0]
        state[1] = start_point[1]

        return state

    # Returns reward and new state when action is taken on state
    # Check 2 cases maintaining the order:
    # 1. car finished by crossing the finish line
    # 2. car goes out of track
    # Ends episode by returning the reward as None and state as usual
    def step(self, state, action):
        self.data.episode['A'].append(action)
        reward = -1

        if self.is_at_finish_area(state, action):
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
