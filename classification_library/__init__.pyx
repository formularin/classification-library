#!python
#cython: language_level=3
"""A classification library using a novel audio-inspired technique.
"""

import pickle

cimport numpy as np
import numpy as np


__version__ = "0.0.1"


cdef class AudioClassifier:
    """A very basic classifier model. Very beginner-friendly.

    Parameters
    ----------
    alpha: int
        A hyperparameter for the novel audio-inspired classification
        technique.
    """

    def __cinit__(self, int alpha):
        self.alpha = alpha
        self._classes = []
        self._cache_x = []
        self._cache_y = []

        np.random.seed(self.alpha)

    cpdef void fit(self, np.ndarray X, np.ndarray y) except *:
        """Fits the model to a training dataset.

        Parameters
        ----------
        X: np.ndarray
            An array containing X-values of samples.
        y: np.ndarray
            A 1-dimensional array of classes.
        """
        cdef object x_val
        cdef object y_val

        for x_val, y_val in zip(X, y):
            self._cache_x.append(x_val)
            self._cache_y.append(y_val)

    cpdef np.ndarray predict(self, np.ndarray X):
        """Makes predictions based off of the model's knowledge gained
        from training.

        Parameters
        ----------
        X: np.ndarray
            An array containing X-values of samples.

        Returns
        -------
        np.ndarray
            The classifier's predictions on the inputted X values.
        """
        cdef np.ndarray y = np.zeros(X.shape[0])
        cdef object x
        cdef int i, _cache_y_index

        i = 0
        for x_val in X:
            if x_val in self._cache_x:
                _cache_y_index = self._cache_x.index(x_val)
            else:
                _cache_y_index = np.random.randint(0, len(self._cache_y) - 1)
            y[i] = self._cache_y[_cache_y_index]
            i += 1

        return y