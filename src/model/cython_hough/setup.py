from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("_line_hough.pyx"),
    include_dirs=[numpy.get_include()]
)