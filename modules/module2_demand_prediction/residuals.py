"""Residual calculations and least-squares orthogonality checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .config import DEFAULT_TOLERANCE
from .least_squares import _as_finite_array
from .models import FloatArray


def _as_column(value: ArrayLike, name: str) -> FloatArray:
    array = _as_finite_array(value, name)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != 1:
        raise ValueError(f"{name} must be a vector or a column vector.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def calculate_residuals(actual: ArrayLike, predicted: ArrayLike) -> FloatArray:
    """Calculate the column residual vector ``actual - predicted``."""
    actual_values = _as_column(actual, "actual")
    predicted_values = _as_column(predicted, "predicted")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have the same number of values.")
    return actual_values - predicted_values


def check_residual_orthogonality(
    A: ArrayLike,
    residuals: ArrayLike,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Return whether ``A.T @ residuals`` is zero within absolute tolerance."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite, non-negative number.")
    matrix = _as_finite_array(A, "A")
    errors = _as_column(residuals, "residuals")
    if matrix.ndim != 2 or matrix.shape[0] != errors.shape[0]:
        raise ValueError("A rows must match the number of residuals.")
    return bool(np.all(np.abs(matrix.T @ errors) <= tolerance))


__all__ = ["calculate_residuals", "check_residual_orthogonality"]
