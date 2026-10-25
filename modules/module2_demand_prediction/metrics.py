"""Prediction error metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .residuals import calculate_residuals
from .models import PredictionMetrics


def calculate_metrics(actual: ArrayLike, predicted: ArrayLike) -> PredictionMetrics:
    """Calculate mean absolute, mean squared, and root mean squared errors."""
    errors = calculate_residuals(actual, predicted)
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    return PredictionMetrics(mae=mae, mse=mse, rmse=float(np.sqrt(mse)))


__all__ = ["calculate_metrics"]
