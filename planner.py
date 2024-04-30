import math
import time

import matplotlib.pyplot as plt
import numpy as np

import FrenetOptimalTrajectory.frenet_optimal_trajectory as frenet_optimal_trajectory
from Dubins.dubins_path_planner import plan_dubins_path

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
        motion_plan = self.calculate_frenet(path)
        self.plot(motion_plan)

    def get_dubins_path(self, curvature: float = 1.0 / 0.4):

        step_size = 1.0
        start = self.start_pose
        goal = self.goal_pose

        path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(
            start.x, start.y, start.yaw, goal.x, goal.y, goal.yaw, curvature, step_size
        )

        return zip(path_x, path_y)

    def calculate_frenet(self, path, show_animation=False):
        wx, wy = zip(*path)
        tx, ty, tyaw, tc, csp = frenet_optimal_trajectory.generate_target_course(
            list(np.array(wx)), list(np.array(wy))
        )

        state = self.initial_state

        motion_plan = []

        for i in range(SIM_LOOP):
            path = frenet_optimal_trajectory.frenet_optimal_planning(
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

            motion_plan.append(path)

            if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
                print("Goal")
                break

        return motion_plan

    def plot(self, motion_plan, area=5.0):
        for path in motion_plan:
            plt.cla()
            for x, y in self.obstacleList:
                circle = plt.Circle((x, y), 0.2, color="k", fill=False)
                plt.gca().add_patch(circle)
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[m/s]:" + str(path.s_d[1])[0:4])
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

    points_2d = [(2.0, 1.6)]
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
