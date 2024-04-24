# cython: language_level=3

from libc.stdlib cimport free
import numpy as cnp

cdef class QuarticPolynomial:
    cdef public double a0, a1, a2, a3, a4
    cpdef double calc_point(self, double t)
    cpdef double calc_first_derivative(self, double t)
    cpdef double calc_second_derivative(self, double t)
    cpdef double calc_third_derivative(self, double t)

cdef class FrenetPath:
    cdef public list t, d, d_d, d_dd, d_ddd, s, s_d, s_dd, s_ddd
    cdef public double cd, cv, cf
    cdef public list x, y, yaw, ds, c
