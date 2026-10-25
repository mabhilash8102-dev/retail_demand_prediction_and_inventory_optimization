"""Result models returned by the demand-prediction API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LeastSquaresResult:
    """Least-squares coefficients and normal-equation intermediates."""

    coefficients: FloatArray
    A_transpose: FloatArray
    AtA: FloatArray
    Atb: FloatArray
    rank: int


@dataclass(frozen=True)
class PredictionResult:
    """Predictions and residuals for a set of observations."""

    predicted_values: FloatArray
    residuals: FloatArray


@dataclass(frozen=True)
class PredictionMetrics:
    """Common prediction-error metrics."""

    mae: float
    mse: float
    rmse: float


__all__ = [
    "FloatArray",
    "LeastSquaresResult",
    "PredictionMetrics",
    "PredictionResult",
]
