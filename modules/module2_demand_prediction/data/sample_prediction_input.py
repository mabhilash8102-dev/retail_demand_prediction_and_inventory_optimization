"""Small retail-demand example independent of Member 1's data loader."""

import numpy as np
from numpy.typing import NDArray

FEATURE_ORDER = ("intercept", "day", "price", "promotion")


def sample_prediction_input() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the academic five-observation retail example as ``(A, b)``."""
    A = np.array(
        [
            [1.0, 1.0, 30.0, 0.0],
            [1.0, 2.0, 29.0, 0.0],
            [1.0, 3.0, 28.0, 1.0],
            [1.0, 4.0, 31.0, 1.0],
            [1.0, 5.0, 30.0, 0.0],
        ],
        dtype=np.float64,
    )
    b = np.array([[52.0], [55.0], [65.0], [68.0], [60.0]], dtype=np.float64)
    return A, b


__all__ = ["FEATURE_ORDER", "sample_prediction_input"]
