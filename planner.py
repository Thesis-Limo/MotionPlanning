import math

import matplotlib.pyplot as plt
import numpy as np

from Dubins.dubins_path_planner import plan_dubins_path
from Frenet.frenet_optimal_trajectory import (
    frenet_optimal_planning,
    generate_target_course,
)

ROBOT_RADIUS = 0.2  # [m]
WHEELBASE = 0.2  # [m]
SIM_LOOP = 500


class Pose:

    def __init__(self, x: float, y: float, yaw: float):
        self.x = x
        self.y = y
        self.yaw = np.deg2rad(yaw)


class FrenetState:

    def __init__(self, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
        self.c_speed = c_speed
        self.c_accel = c_accel
        self.c_d = c_d
        self.c_d_d = c_d_d
        self.c_d_dd = c_d_dd
        self.s0 = s0


class MotionPlanner:

    def __init__(
        self,
        goal_pose: Pose,
        start_pose: Pose = Pose(0.0, 0.0, 0.0),
        obstacleList: list = [],
        initial_state: FrenetState = FrenetState(0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
    ):
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.obstacleList = obstacleList
        self.initial_state = initial_state

    def plan(self):
        path = self.get_dubins_path()
        self.calculate_frenet(path)

    def get_dubins_path(self, curvature: float = 1.0 / 0.4):

        step_size = 0.5
        start = self.start_pose
        goal = self.goal_pose

        path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(
            start.x, start.y, start.yaw, goal.x, goal.y, goal.yaw, curvature, step_size
        )

        return zip(path_x, path_y)

    def calculate_frenet(self, path, show_animation=True):
        wx, wy = zip(*path)
        tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

        area = 5.0  # animation area length [m]
        state = self.initial_state

        for i in range(SIM_LOOP):
            path = frenet_optimal_planning(
                csp,
                state.s0,
                state.c_speed,
                state.c_accel,
                state.c_d,
                state.c_d_d,
                state.c_d_dd,
                self.obstacleList,
            )

            state = FrenetState(
                c_speed=path.s_d[1],
                c_accel=path.s_dd[1],
                c_d=path.d[1],
                c_d_d=path.d_d[1],
                c_d_dd=path.d_dd[1],
                s0=path.s[1],
            )

            print(f"Time Step {i}")
            speed = path.s_d[1]
            curvature = path.c[1]
            steering_angle = math.atan2(WHEELBASE * curvature, 1.0)

            print(
                f"Speed = {speed:.2f} m/s^2, Steering Angle = {steering_angle:.2f} radians"
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
                for x, y in obstacleList:
                    circle = plt.Circle((x, y), 0.2, color="k", fill=False)
                    plt.gca().add_patch(circle)
                plt.plot(path.x[1:], path.y[1:], "-or")
                plt.plot(path.x[1], path.y[1], "vc")
                plt.xlim(path.x[1] - area, path.x[1] + area)
                plt.ylim(path.y[1] - area, path.y[1] + area)
                plt.title("v[m/s]:" + str(state.c_speed)[0:4])
                plt.grid(True)
                plt.pause(0.0001)

        print("Finish")
        if show_animation:  # pragma: no cover
            plt.grid(True)
            plt.pause(0.0001)
            plt.show()


def convert_lidar_data_to_2d_points(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    angle_min = angle_increment = 0.0
    ranges = []

    for line in lines:
        if "angle_min:" in line:
            angle_min = float(line.split(":")[1].strip())
        elif "angle_max:" in line:
            angle_max = float(line.split(":")[1].strip())
        elif "angle_increment:" in line:
            angle_increment = float(line.split(":")[1].strip())
        elif "ranges:" in line:
            ranges_str = line.split(":", 1)[1].strip().strip("[]")
            ranges = [
                float(x)
                for x in ranges_str.split(",")
                if x.strip().replace(".", "", 1).isdigit()
            ]

    points_2d = []
    current_angle = angle_min

    for range_value in ranges:
        if range_value > 0:
            x = range_value * math.cos(current_angle)
            y = range_value * math.sin(current_angle)
            points_2d.append((x, y))

        current_angle += angle_increment

    return points_2d


if __name__ == "__main__":

    file_path = "data/scan.txt"

    goal_pose = Pose(
        x=3.0,
        y=2.5,
        yaw=90,
    )

    obstacleList = np.array(convert_lidar_data_to_2d_points(file_path))

    planner = MotionPlanner(goal_pose, obstacleList=obstacleList)
    planner.plan()
