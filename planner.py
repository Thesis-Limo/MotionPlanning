import math
import time

import matplotlib.pyplot as plt
import numpy as np

from Dubins.dubins_path_planner import plan_dubins_path
from Frenet.frenet_optimal_trajectory import (
    frenet_optimal_planning,
    generate_target_course,
)
from RRTStarDubins.rrt_star_dubins import RRTStarDubins
from utils.plot import plot_arrow

show_animation = True

ROBOT_RADIUS = 0.2  # robot radius [m]
SIM_LOOP = 500

obstacleList = [
    (x, y, ROBOT_RADIUS) for x, y in [(2, 5), (2.4, 5), (0, 7), (-3, 4), (-2.6, 4)]
]
goal = [3.0, 3.0, np.deg2rad(-45.0)]


def main_dubins():
    print("Dubins path planner sample start!!")

    start_x = 0.0  # [m]
    start_y = 0.0  # [m]
    start_yaw = np.deg2rad(90.0)  # [rad]

    end_x, end_y, end_yaw = goal

    curvature = 0.4

    start_time = time.time()

    path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature
    )

    end_time = time.time()

    runtime = end_time - start_time
    print("Runtime of the Dubins path planning: {:.4f} seconds".format(runtime))

    if show_animation:
        plt.plot(path_x, path_y, label="".join(mode))
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        # plot the obstacles
        for obstacle in obstacleList:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color="black")
            plt.gcf().gca().add_artist(circle)
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    return [[x, y] for x, y in zip(path_x, path_y)]


def main_rrt_star_dubins():
    print("Start rrt star with dubins planning")

    # Set Initial parameters
    start = [0.0, 0.0, np.deg2rad(0.0)]

    start_time = time.time()

    rrtstar_dubins = RRTStarDubins(
        start,
        goal,
        rand_area=[-2.0, 15.0],
        obstacle_list=obstacleList,
        max_iter=50,
        robot_radius=0.20,
    )
    path = rrtstar_dubins.planning(animation=show_animation)

    end_time = time.time()
    runtime = end_time - start_time
    print(
        "Runtime of the RRT* with Dubins path planning: {:.4f} seconds".format(runtime)
    )

    # Draw final path
    if show_animation:  # pragma: no cover
        rrtstar_dubins.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], "-r")
        plt.grid(True)
        plt.pause(0.001)

        plt.show()

    return path[::-1]


def main_frenet(path):
    print(__file__ + " start!!")

    wx, wy = zip(*path)
    # obstacle lists
    ob = np.array([obstacle[:2] for obstacle in obstacleList])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = 0.01  # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 5.0  # animation area length [m]

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob
        )

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]

        print(f"Time Step {i}")
        acceleration = path.s_dd[0]  # Longitudinal acceleration
        curvature = path.c[0]  # Curvature at each point
        wheelbase = 2.5  # Example wheelbase of the vehicle in meters
        steering_angle = math.atan2(
            wheelbase * curvature, 1.0
        )  # Steering angle in radians

        print(
            f"Acceleration = {acceleration:.2f} m/s^2, Steering Angle = {steering_angle:.2f} radians"
        )

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == "__main__":
    path = main_dubins()
    # main_rrt_star_dubins()
    main_frenet(path)
