"""Historical and future demand prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .config import DEFAULT_TOLERANCE
from .least_squares import _as_finite_array
from .models import FloatArray


def predict_from_matrix(A: ArrayLike, coefficients: ArrayLike) -> FloatArray:
    """Return historical predictions ``A @ coefficients`` as a column."""
    matrix = _as_finite_array(A, "A")
    beta = _as_finite_array(coefficients, "coefficients")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("A must be a non-empty two-dimensional matrix.")
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
    if beta.ndim != 2 or beta.shape[1] != 1:
        raise ValueError("coefficients must be a vector or a column vector.")
    if matrix.shape[1] != beta.shape[0]:
        raise ValueError(
            f"A has {matrix.shape[1]} columns but coefficients has "
            f"{beta.shape[0]} values."
        )
    return matrix @ beta


def predict(
    coefficients: ArrayLike,
    features: Mapping[str, float],
    feature_order: Sequence[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Predict one future observation using an explicit coefficient order.

    ``feature_order`` lists one name per coefficient. The name ``intercept``
    is reserved and contributes 1.0; all other names must exist in
    ``features``. This matches Member 1's default ``[intercept, day, price,
    promotion]`` design-matrix column order when that same order is supplied.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite, non-negative number.")
    beta = _as_finite_array(coefficients, "coefficients")
    if beta.ndim == 2 and beta.shape[1] == 1:
        beta = beta[:, 0]
    if beta.ndim != 1 or beta.size == 0:
        raise ValueError("coefficients must be a non-empty vector or column vector.")
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping of feature names to values.")
    if feature_order is None:
        raise ValueError("feature_order is required; coefficient order is not inferred.")
    order = tuple(feature_order)
    if not order or len(order) != beta.size:
        raise ValueError(
            "feature_order must contain exactly one name per coefficient."
        )
    if any(not isinstance(name, str) or not name for name in order):
        raise ValueError("feature_order names must be non-empty strings.")
    if len(set(order)) != len(order):
        raise ValueError("feature_order cannot contain duplicate names.")

    row: list[float] = []
    for name in order:
        if name == "intercept":
            row.append(1.0)
        else:
            if name not in features:
                raise ValueError(f"Missing future feature '{name}'.")
            value = _as_finite_array(features[name], f"feature '{name}'")
            if value.ndim != 0:
                raise ValueError(f"Feature '{name}' must be a scalar.")
            row.append(float(value))
    return float(np.dot(np.asarray(row, dtype=np.float64), beta))


__all__ = ["predict", "predict_from_matrix"]
