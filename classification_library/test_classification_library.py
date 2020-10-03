"""Unit tests for classification_library.
"""

import numpy as np
import pytest

from classification_library import AudioClassifier


def test_audio_classifier():
    """Tests the AudioClassifier class.
    """

    X_train = np.array([1, 2, 3, 4, 5, 6])
    y_train = np.array([1, 2, 3, 4, 5, 6])
    X_test = np.array([1, 2, 3, 4, 6, 7])
    y_test = np.array([1, 2, 3, 4, 6, 7])

    model = AudioClassifier(42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert (predictions[:-1] == y_test[:-1]).all()
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 2)))