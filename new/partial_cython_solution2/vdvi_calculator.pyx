# vdvi_calculator.pyx
cimport cython
cimport numpy as np
import numpy as np

# Disable bounds checking and negative indexing for performance
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_vdvi_chunk(double[:, :] r, double[:, :] g, double[:, :] b):
    """
    Calculate VDVI (Visible Difference Vegetation Index) for RGB arrays.
    """
    cdef Py_ssize_t height = r.shape[0]
    cdef Py_ssize_t width = r.shape[1]
    cdef double[:, :] vdvi = np.zeros((height, width), dtype=np.float64)
    cdef double denominator, numerator
    cdef Py_ssize_t i, j
    cdef double eps = 1e-10  # Small epsilon to avoid division by zero

    for i in range(height):
        for j in range(width):
            denominator = 2 * g[i, j] + r[i, j] + b[i, j]
            if denominator == 0:
                denominator = eps
            numerator = 2 * g[i, j] - r[i, j] - b[i, j]
            vdvi[i, j] = numerator / denominator

    return np.asarray(vdvi)

@cython.boundscheck(False)
@cython.wraparound(False)
def count_positive_pixels(unsigned char[:, :] array):
    """
    Count the number of positive pixels (> 0) in the array.
    """
    cdef Py_ssize_t height = array.shape[0]
    cdef Py_ssize_t width = array.shape[1]
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t i, j

    for i in range(height):
        for j in range(width):
            if array[i, j] > 0:
                count += 1

    return count