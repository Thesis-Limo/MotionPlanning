# cython: language_level=3

from libc.stdlib cimport free
import numpy as cnp

cdef class CubicSpline1D:
    cdef public int nx
    cdef public double[:] x, y, a, b, c, d
    cdef void __calc_coefficients(self)
    cpdef double calc_position(self, double x)
    cpdef double calc_first_derivative(self, double x)
    cpdef double calc_second_derivative(self, double x)
    cdef int __search_index(self, double x)

cdef class CubicSpline2D:
    cdef public CubicSpline1D sx, sy
    cdef public double[:] s
    cdef double[:] __calc_s(self, double[:] x, double[:] y)
    cpdef tuple calc_position(self, double s)
    cpdef double calc_curvature(self, double s)
    cpdef double calc_yaw(self, double s)
