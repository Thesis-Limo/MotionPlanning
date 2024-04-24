# cython: language_level=3

cdef double MIN_T = 5.0  # minimum time to the goal [s]
cdef double MAX_T = 100.0  # maximum time to the goal [s]
cdef bint show_animation = True  # boolean, use 'bint' for Cython boolean type

from libc.math cimport cos, sin, atan2, hypot, pow, fabs
import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt

cnp.import_array()

cdef class QuinticPolynomial:

    def __init__(self, double xs, double vxs, double axs, double xe, double vxe, double axe, double time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        cdef double[:, :] A = np.array([[time ** 3, time ** 4, time ** 5],
                                        [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                                        [6 * time, 12 * time ** 2, 20 * time ** 3]], dtype=np.float64)
        cdef double[:] b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                                     vxe - self.a1 - 2 * self.a2 * time,
                                     axe - 2 * self.a2], dtype=np.float64)
        cdef double[:] x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    cpdef double calc_point(self, double t):
        return (self.a0 + self.a1 * t + self.a2 * t ** 2 +
                self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5)

    cpdef double calc_first_derivative(self, double t):
        return (self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 
                4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4)

    cpdef double calc_second_derivative(self, double t):
        return (2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3)

    cpdef double calc_third_derivative(self, double t):
        return (6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2)

def quintic_polynomials_planner(double sx, double sy, double syaw, double sv, double sa,
                                double gx, double gy, double gyaw, double gv, double ga,
                                double max_accel, double max_jerk, double dt):
    cdef double vxs = sv * cos(syaw)
    cdef double vys = sv * sin(syaw)
    cdef double vxg = gv * cos(gyaw)
    cdef double vyg = gv * sin(gyaw)

    cdef double axs = sa * cos(syaw)
    cdef double ays = sa * sin(syaw)
    cdef double axg = ga * cos(gyaw)
    cdef double ayg = ga * sin(gyaw)

    cdef QuinticPolynomial xqp, yqp
    cdef double T, t, vx, vy, v, yaw, ax, ay, a, jx, jy, j
    cdef list time, rx, ry, ryaw, rv, ra, rj

    for T in np.arange(MIN_T, MAX_T, MIN_T):
        xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

        time = []
        rx = []
        ry = []
        ryaw = []
        rv = []
        ra = []
        rj = []

        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))
            
            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = hypot(vx, vy)
            yaw = atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)

            ax = xqp.calc_second_derivative(t)
            ay = yqp.calc_second_derivative(t)
            a = hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)

            jx = xqp.calc_third_derivative(t)
            jy = yqp.calc_third_derivative(t)
            j = hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)

        if max([fabs(i) for i in ra]) <= max_accel and max([fabs(i) for i in rj]) <= max_jerk:
            print("find path!!")
            break

    return time, rx, ry, ryaw, rv, ra, rj

def plot_arrow(double x, double y, double yaw, double length=1.0, double width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main():
    print("Quintic Polynomial Planner start!!")

    # Define start and goal states
    cdef double sx = 10.0  # start x position [m]
    cdef double sy = 10.0  # start y position [m]
    cdef double syaw = np.deg2rad(10.0)  # start yaw angle [rad]
    cdef double sv = 0  # start speed [m/s]
    cdef double sa = 0  # start acceleration [m/ss]
    cdef double gx = 100.0  # goal x position [m]
    cdef double gy = 10.0  # goal y position [m]
    cdef double gyaw = np.deg2rad(20.0)  # goal yaw angle [rad]
    cdef double gv = 0  # goal speed [m/s]
    cdef double ga = 0  # goal acceleration [m/ss]
    cdef double max_accel = 1.0  # maximum acceleration [m/ss]
    cdef double max_jerk = 10000.0  # maximum jerk [m/sss]
    cdef double dt = 0.1  # time tick [s]

    # Compute the quintic polynomial path
    cdef tuple result = quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)
    cdef list time = result[0]
    cdef list x = result[1]
    cdef list y = result[2]
    cdef list yaw = result[3]
    cdef list v = result[4]
    cdef list a = result[5]
    cdef list j = result[6]

    if show_animation:
        plt.figure()

        # Plot trajectory
        plt.plot(x, y, "-r", label="Path")
        plt.legend()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Quintic Polynomial Trajectory")
        plt.grid(True)
        plt.axis("equal")

        # Plot yaw
        plt.figure()
        plt.plot(time, [np.rad2deg(iyaw) for iyaw in yaw], "-r")
        plt.xlabel("Time [s]")
        plt.ylabel("Yaw [deg]")
        plt.title("Yaw over Time")
        plt.grid(True)

        # Plot speed
        plt.figure()
        plt.plot(time, v, "-r")
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [m/s]")
        plt.title("Speed over Time")
        plt.grid(True)

        # Plot acceleration
        plt.figure()
        plt.plot(time, a, "-r")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [m/ss]")
        plt.title("Acceleration over Time")
        plt.grid(True)

        # Plot jerk
        plt.figure()
        plt.plot(time, j, "-r")
        plt.xlabel("Time [s]")
        plt.ylabel("Jerk [m/sss]")
        plt.title("Jerk over Time")
        plt.grid(True)

        plt.show()

if __name__ == "__main__":
    main()
