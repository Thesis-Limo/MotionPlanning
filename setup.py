import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "cubic_spline_planner",
        ["CubicSpline/cubic_spline_planner.pyx"],
        include_dirs=[np.get_include()],
    )  # Ensure NumPy headers are included
]

setup(name="Cubic Spline Planner", ext_modules=cythonize(extensions))
