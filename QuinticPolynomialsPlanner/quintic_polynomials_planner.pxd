# cython: language_level=3

cdef class QuinticPolynomial:
    cdef double a0, a1, a2, a3, a4, a5
    cpdef double calc_point(self, double t)
    cpdef double calc_first_derivative(self, double t)
    cpdef double calc_second_derivative(self, double t)
    cpdef double calc_third_derivative(self, double t)