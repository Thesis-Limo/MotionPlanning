import numpy as np
import matplotlib.pyplot as plt

def ackermann_trajectory_planning(x0, xf, v0, vf, m=100):
    """
    Point to Point Cubic Trajectory Planning for Ackermann Steering Vehicle

    Parameters
    ---
    x0  : Initial State [x, y, θ] (3x1)
    xf  : Final State [x, y, θ] (3x1)
    v0  : Initial Velocity
    vf  : Final Velocity
    m   : Discrete Time Steps (Optional)

    Returns
    ---
    x, xd, xdd : Position, Velocity, and Acceleration (3xm)
    """
    # Polynomial Parameters
    a0 = np.copy(x0)
    a1 = np.array([v0 * np.cos(x0[2]), v0 * np.sin(x0[2]), 0]) # Velocity converted to x, y, θ components
    a2 = 3 * (xf - x0) - 2 * a1 - np.array([vf * np.cos(xf[2]), vf * np.sin(xf[2]), 0])
    a3 = -2 * (xf - x0) + a1 + np.array([vf * np.cos(xf[2]), vf * np.sin(xf[2]), 0])

    timesteps = np.linspace(0, 1, num=m)

    x = np.zeros((3, m))
    xd = np.zeros((3, m))
    xdd = np.zeros((3, m))

    for i in range(len(timesteps)):
        t = timesteps[i]
        t_2 = t * t
        t_3 = t * t * t

        x[:, i] = a0 + a1 * t + a2 * t_2 + a3 * t_3
        xd[:, i] = a1 + 2 * a2 * t + 3 * a3 * t_2
        xdd[:, i] = 2 * a2 + 6 * a3 * t

    return x, xd, xdd

def plot_vehicle_trajectory(x, xd, xdd):
    """
    Function to plot vehicle trajectories

    Parameters
    ---
    x   : Vehicle Position [x, y, θ] (3xm)
    xd  : Vehicle Velocity (3xm)
    xdd : Vehicle Acceleration (3xm)

    Returns
    ---
    None
    """
    m = x.shape[1]
    timesteps = np.linspace(0, 1, num=m)

    fig, axis = plt.subplots(4, figsize=(10, 8))
    fig.suptitle("Vehicle Trajectories")

    # Velocity Plot
    axis[0].set_title("Velocity")
    axis[0].plot(timesteps, np.linalg.norm(xd[:2, :], axis=0))
    axis[0].set(xlabel="Time", ylabel="Velocity")

    axis[2].set_title("Acceleration")
    axis[2].plot(timesteps, np.linalg.norm(xdd[:2, :], axis=0))
    axis[2].set(xlabel="Time", ylabel="Acceleration")

    # Orientation Plot
    axis[3].set_title("Orientation")
    axis[3].plot(timesteps, x[2])
    axis[3].set(xlabel="Time", ylabel="Orientation (θ)")

    # steering angle plot    \theta(t) = \theta_0 + \frac{v_0 + a \cdot t \cdot \tan(\delta)}{L} \cdot t
    axis[1].set_title("Steering Angle")
    axis[1].plot(timesteps, np.arctan2(xd[1], xd[0]))
    axis[1].set(xlabel="Time", ylabel="Steering Angle")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    x0 = np.array([0, 0, np.pi/4])  # Initial position (x, y) and orientation (θ)
    xf = np.array([10, 10, np.pi/2])  # Final position (x, y) and orientation (θ)
    v0 = 0  # Initial velocity
    vf = 0  # Final velocity

    x, xd, xdd = ackermann_trajectory_planning(x0, xf, v0, vf)
    plot_vehicle_trajectory(x, xd, xdd)