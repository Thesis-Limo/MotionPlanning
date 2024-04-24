# cython: language_level=3

from libc.math cimport pow, atan2
import matplotlib.pyplot as plt
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

cdef class CubicSpline1D:

    def __init__(self, double[:] x, double[:] y):
        self.x = x
        self.y = y
        self.nx = x.shape[0]

        self.a = y.copy()
        self.b = np.zeros(self.nx - 1, dtype=np.float64)
        self.c = np.zeros(self.nx, dtype=np.float64)
        self.d = np.zeros(self.nx - 1, dtype=np.float64)

        self.__calc_coefficients()

    cdef void __calc_coefficients(self):
        cdef int i
        cdef double[:] h = np.diff(self.x)
        cdef double[:, :] A = np.zeros((self.nx, self.nx), dtype=np.float64)
        cdef double[:] B = np.zeros(self.nx, dtype=np.float64)
        
        A[0, 0] = 1.0
        A[self.nx - 1, self.nx - 1] = 1.0

        for i in range(1, self.nx - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]

        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]

        self.c = np.linalg.solve(A, B)

        for i in range(self.nx - 1):
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            self.b[i] = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (2.0 * self.c[i] + self.c[i + 1]) / 3.0

    cpdef double calc_position(self, double x):
        cdef int i
        cdef double dx, position

        if x < self.x[0] or x > self.x[-1]:
            return float('nan')

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3
        return position

    cpdef double calc_first_derivative(self, double x):
        cdef int i
        cdef double dx, dy

        if x < self.x[0] or x > self.x[-1]:
            return float('nan')

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2 * self.c[i] * dx + 3 * self.d[i] * dx**2
        return dy

    cpdef double calc_second_derivative(self, double x):
        cdef int i
        cdef double dx, ddy

        if x < self.x[0] or x > self.x[-1]:
            return float('nan')

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2 * self.c[i] + 6 * self.d[i] * dx
        return ddy

    cdef int __search_index(self, double x):
        cdef int idx = np.searchsorted(self.x, x) - 1
        if idx < 0:
            idx = 0
        elif idx >= self.nx - 1:
            idx = self.nx - 2
        return idx

cdef class CubicSpline2D:

    def __init__(self, double[:] x, double[:] y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    cdef double[:] __calc_s(self, double[:] x, double[:] y):
        cdef double[:] dx = np.diff(x)
        cdef double[:] dy = np.diff(y)
        cdef double[:] ds = np.hypot(dx, dy)
        cdef double[:] s = np.concatenate(([0.0], np.cumsum(ds)))
        return s

    cpdef tuple calc_position(self, double s):
        cdef double x = self.sx.calc_position(s)
        cdef double y = self.sy.calc_position(s)
        return (x, y)

    cpdef double calc_curvature(self, double s):
        cdef double dx = self.sx.calc_first_derivative(s)
        cdef double dy = self.sy.calc_first_derivative(s)
        cdef double ddx = self.sx.calc_second_derivative(s)
        cdef double ddy = self.sy.calc_second_derivative(s)
        if dx**2 + dy**2 == 0:
            return float('nan')
        return (ddy * dx - ddx * dy) / pow(dx**2 + dy**2, 1.5)

    cpdef double calc_yaw(self, double s):
        cdef double dx = self.sx.calc_first_derivative(s)
        cdef double dy = self.sy.calc_first_derivative(s)
        return atan2(dy, dx)

def main_1d():
    print("CubicSpline1D test")
    x = np.arange(5.0, dtype=np.float64)
    y = np.array([1.7, -6, 5, 6.5, 0.0], dtype=np.float64)
    sp = CubicSpline1D(x, y)
    xi = np.linspace(0.0, 5.0, 100)
    yi = [sp.calc_position(x) for x in xi]

    plt.figure()
    plt.plot(x, y, "xb", label="Data points")
    plt.plot(xi, yi, "r", label="Cubic spline interpolation")
    plt.grid(True)
    plt.legend()
    plt.show()

def main_2d():
    print("CubicSpline2D test")
    cdef double[:] x = np.array([-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0], dtype=np.float64)
    cdef double[:] y = np.array([0.7, -6, 5, 6.5, 0.0, 5.0, -2.0], dtype=np.float64)
    cdef double ds = 0.1  # [m] distance of each interpolated points

    cdef CubicSpline2D sp = CubicSpline2D(x, y)
    cdef double[:] s = np.arange(0, sp.s[-1], ds)
    cdef list rx = [], ry = [], ryaw = [], rk = []
    cdef double ix, iy, i_s

    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    plt.figure()
    # Convert memory views to NumPy arrays before plotting
    plt.plot(np.array(x), np.array(y), "xb", label="Data points")
    plt.plot(rx, ry, "-r", label="Cubic spline path")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()

