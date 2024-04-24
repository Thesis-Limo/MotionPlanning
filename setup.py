import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "cubic_spline_planner",
        ["CubicSpline/cubic_spline_planner.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "quintic_polynomials_planner",
        ["QuinticPolynomialsPlanner/quintic_polynomials_planner.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(name="Path Planning Library", ext_modules=cythonize(extensions))
