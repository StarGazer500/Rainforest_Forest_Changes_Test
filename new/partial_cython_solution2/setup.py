# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="vdvi_calculator",
    ext_modules=cythonize("vdvi_calculator.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)



