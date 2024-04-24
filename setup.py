import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "CubicSpline.cubic_spline_planner",
        sources=["CubicSpline/cubic_spline_planner.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "QuinticPolynomialsPlanner.quintic_polynomials_planner",
        sources=["QuinticPolynomialsPlanner/quintic_polynomials_planner.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "FrenetOptimalTrajectory.frenet_optimal_trajectory",
        sources=["FrenetOptimalTrajectory/frenet_optimal_trajectory.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(name="Path Planning Library", ext_modules=cythonize(extensions))
