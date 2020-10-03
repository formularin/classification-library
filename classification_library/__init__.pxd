#!python
#cython: language_level=3

cimport numpy as np


cdef class AudioClassifier:
    cdef list _classes, _cache_x, _cache_y

    cdef public int alpha

    cpdef void fit(np.ndarray X, np.ndarray[ndim=1] y) except *
    cpdef np.ndarray[ndim=1] predict(np.ndarray X)