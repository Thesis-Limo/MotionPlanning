# cython: language_level=3

from libc.math cimport cos, sin, atan2, pow, fabs, sqrt, pi
from libc.stdlib cimport malloc, free
from copy import deepcopy
import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt

from CubicSpline.cubic_spline_planner cimport CubicSpline2D
from QuinticPolynomialsPlanner.quintic_polynomials_planner cimport QuinticPolynomial

cdef double MAX_SPEED = 1.0
cdef double MAX_ACCEL = 1.0
cdef double MAX_CURVATURE = 2.5  # 1 / 0.4
cdef double MAX_ROAD_WIDTH = 7.0
cdef double D_ROAD_W = 1.0
cdef double DT = 0.2
cdef double MAX_T = 4.0
cdef double MIN_T = 2.0
cdef double D_T_S = 5.0 / 3.6
cdef int N_S_SAMPLE = 1
cdef double ROBOT_RADIUS = 0.2

cdef double K_J = 0.1
cdef double K_T = 0.1
cdef double K_D = 1.0
cdef double K_LAT = 1.0
cdef double K_LON = 1.0

cdef bint show_animation = True
cdef int SIM_LOOP = 500

cdef class QuarticPolynomial:

    def __init__(self, double xs, double vxs, double axs, double vxe, double axe, double time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        cdef double[:, :] A = np.array([[3 * time ** 2, 4 * time ** 3], [6 * time, 12 * time ** 2]], dtype=np.float64)
        cdef double[:] b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2], dtype=np.float64)
        cdef double[:] x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]

    cpdef double calc_point(self, double t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4

    cpdef double calc_first_derivative(self, double t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

    cpdef double calc_second_derivative(self, double t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

    cpdef double calc_third_derivative(self, double t):
        return 6 * self.a3 + 24 * self.a4 * t

cdef class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

# Functions to generate Frenet paths, calculate global paths, and perform collision checks

def calc_frenet_paths(double c_speed, double c_accel, double c_d, double c_d_d, double c_d_dd, double s0, double target_speed):
    cdef list frenet_paths = []
    cdef double di, Ti
    cdef FrenetPath fp
    cdef QuinticPolynomial lat_qp
    cdef QuarticPolynomial lon_qp
    cdef FrenetPath tfp

    # Generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):
        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(target_speed - D_T_S * N_S_SAMPLE, target_speed + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                tfp.cd = K_J * sum([pow(x, 2) for x in tfp.d_ddd]) + K_T * Ti + K_D * pow(tfp.d[-1], 2)
                tfp.cv = K_J * sum([pow(x, 2) for x in tfp.s_ddd]) + K_T * Ti + K_D * pow((target_speed - tfp.s_d[-1]), 2)
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

cdef list calc_global_paths(list fplist, CubicSpline2D csp):
    cdef FrenetPath fp
    cdef int i
    cdef double ix, iy, i_yaw, di, fx, fy
    for fp in fplist:
        # Calculate global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                continue
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * cos(i_yaw + pi / 2.0)
            fy = iy + di * sin(i_yaw + pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # Calculate yaw and ds
        for i in range(len(fp.x) - 1):
            fp.yaw.append(atan2(fp.y[i + 1] - fp.y[i], fp.x[i + 1] - fp.x[i]))
            fp.ds.append(sqrt(pow(fp.x[i + 1] - fp.x[i], 2) + pow(fp.y[i + 1] - fp.y[i], 2)))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # Calculate curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist

cdef bint check_collision(FrenetPath fp, cnp.ndarray[cnp.float64_t, ndim=2] ob):
    cdef int i
    cdef double ix, iy, di
    for i in range(len(ob)):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2) for ix, iy in zip(fp.x, fp.y)]
        if any(di <= ROBOT_RADIUS ** 2 for di in d):
            return False
    return True

cdef list check_paths(list fplist, cnp.ndarray[cnp.float64_t, ndim=2] ob):
    cdef list ok_ind = []
    cdef int i
    for i in range(len(fplist)):
        if any(v > MAX_SPEED for v in fplist[i].s_d):  # Max speed check
            continue
        elif any(fabs(a) > MAX_ACCEL for a in fplist[i].s_dd):  # Max accel check
            continue
        elif any(fabs(c) > MAX_CURVATURE for c in fplist[i].c):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue
        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]

cpdef tuple generate_target_course(list wx, list wy):
    csp = CubicSpline2D(np.array(wx, dtype=np.float64), np.array(wy, dtype=np.float64))
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

cpdef FrenetPath frenet_optimal_planning(CubicSpline2D csp, double s0, double c_speed, double c_accel, double c_d, double c_d_d, double c_d_dd, cnp.ndarray[cnp.float64_t, ndim=2] ob, double target_speed):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, target_speed)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path

def main():
    print(__file__ + " start!!")

    # Waypoints
    cdef list wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    cdef list wy = [0.0, -6.0, 5.0, 6.5, 0.0]

    # Obstacle lists
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ob = np.array([[20.0, 10.0],
                                                           [30.0, 6.0],
                                                           [30.0, 8.0],
                                                           [35.0, 8.0],
                                                           [50.0, 3.0]], dtype=np.float64)

    cdef double target_speed = 0.5

    # Generate target course
    cdef tuple target_course = generate_target_course(wx, wy)
    cdef list tx = target_course[0]
    cdef list ty = target_course[1]
    cdef list tyaw = target_course[2]
    cdef list tc = target_course[3]
    cdef CubicSpline2D csp = target_course[4]

    # Initial state
    cdef double c_speed = 0.01  # current speed [m/s]
    cdef double c_accel = 0.0  # current acceleration [m/ss]
    cdef double c_d = 0.0  # current lateral position [m]
    cdef double c_d_d = 0.0  # current lateral speed [m/s]
    cdef double c_d_dd = 0.0  # current lateral acceleration [m/s]
    cdef double s0 = 0.0  # current course position

    cdef double area = 5.0  # animation area length [m]
    cdef FrenetPath path

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob, target_speed)

        # Update state
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal reached")
            break

        if show_animation:
            plt.cla()
            plt.plot(tx, ty, label="Course")
            plt.plot(ob[:, 0], ob[:, 1], "xk", label="Obstacles")
            plt.plot(path.x[1:], path.y[1:], "-or", label="Trajectory")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title(f"Time Step: {i}")
            plt.pause(0.001)
            plt.legend()

    print("Finish")
    if show_animation:
        plt.show()

if __name__ == "__main__":
    main()
