# # setup.py
# from setuptools import setup
# from Cython.Build import cythonize
# import numpy as np

# setup(
#     name="vdvi_calculator",
#     ext_modules=cythonize("vdvi_calculator.pyx"),
#     include_dirs=[np.get_include()],
#     zip_safe=False,
# )



from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="cython_optimized_funcs",
    ext_modules=cythonize("cython_optimized_funcs.pyx"),
    include_dirs=[numpy.get_include()],
    # install_requires=["numpy", "rasterio"],
)