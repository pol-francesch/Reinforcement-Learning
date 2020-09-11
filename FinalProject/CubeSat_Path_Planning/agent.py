import numpy as np

def theta_to_angle(theta):
    # Definition of theta
    # theta = 0 -> 0   deg
    # theta = 1 -> pi/4  deg
    # theta = 2 -> pi/2  deg
    # theta = 3 -> 3pi/4 deg
    # theta = 4 -> pi deg
    # theta = 5 -> -3pi/4 deg
    # theta = 6 -> -pi/2 deg
    # theta = 7 -> -pi/4 deg
    group_theta = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]

    return group_theta[theta]

# This class generates an agent object which has actions it can perform
class Agent:
    def possible_actions(self, state):
        # 0. Turn on front thrusters
        # 1. Turn off front thrusters
        # 2. Turn on back thrusters
        # 3. Turn off back thrusters
        # 4. Turn on reaction wheel (spin right)
        # 5. Turn on reaction wheel (spin left)
        # 6. Turn off reaction wheel
        # 7. Do nothing
        alpha = [0, 1, 2, 3, 4, 5, 6, 7]

        # Splitting up state vector for simplicity
        velocity = state[2:4]
        theta = state[4]
        front_thruster = state[5] == 1
        back_thruster = state[5] == 2

        if state[5] == 3:
            reaction_wheel = 1
        elif state[5] == 4:
            reaction_wheel = -1
        else:
            reaction_wheel = 0

        # Check if the car is already going at top-speed. 4 grid/update seems reasonable
        beta = alpha.copy()
        max_speed = 4
        if abs(velocity[0]) >= max_speed or abs(velocity[1]) >= max_speed:
            # Do not allow CubeSat to take action which would speed it up.
            # If we take the CubeSat as a circle, we can divide it in two where the velocity vector splits one of these
            # half's in half. We do not want the Cubesat to turn on the thrusters on the other half, as that would speed
            # it up in this direction, hence breaching the speed limit.

            # Calculate angle of velocity
            theta_v = np.arctan2(velocity[1], velocity[0])
            angle = theta_to_angle(theta)
            if abs(theta_v - angle) <= np.pi/2:
                beta = np.delete(beta, np.where(beta == 2))

                # If the back thruster is on, such that we are speeding up and will now breach top speed,
                # We want to reduce the possible actions to turning off the back thruster. This way,
                # the algorithm cannot "game" the system. Same is true for the next block.
                if back_thruster == 1:
                    beta = np.delete(beta, np.where(beta == 1))
                    beta = np.delete(beta, np.where(beta == 6))
                    beta = np.delete(beta, np.where(beta == 7))
            else:
                beta = np.delete(beta, np.where(beta == 0))

                if front_thruster == 1:
                    beta = np.delete(beta, np.where(beta == 3))
                    beta = np.delete(beta, np.where(beta == 6))
                    beta = np.delete(beta, np.where(beta == 7))

        # The CubeSat can only be allowed to do one action at a time. So, if the reaction wheel is already on, then the
        # thrusters cannot be turned on, and so on.
        if front_thruster == 1 or back_thruster == 1 or abs(reaction_wheel) > 0:
            beta = np.delete(beta, np.where(beta == 0))
            beta = np.delete(beta, np.where(beta == 2))
            beta = np.delete(beta, np.where(beta == 4))
            beta = np.delete(beta, np.where(beta == 5))

        return beta

    # Constructor
    def __init__(self):
        pass

# End of Agent class definition
