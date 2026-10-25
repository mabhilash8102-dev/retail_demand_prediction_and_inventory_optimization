from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from .models import InventoryCalculationResult

NumericVector = Union[Sequence[float], np.ndarray]


def _coerce_vector(values: NumericVector, name: str) -> np.ndarray:
    if values is None:
        raise TypeError(f"{name} cannot be None.")

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric vector, not a string.")

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only numeric values.") from exc

    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative quantities.")

    return array.astype(float, copy=False)


def calculate_required_stock(predicted_demand: NumericVector, safety_stock: NumericVector) -> np.ndarray:
    """Calculate required stock vector as predicted demand plus safety stock."""
    predicted = _coerce_vector(predicted_demand, "predicted_demand")
    safety = _coerce_vector(safety_stock, "safety_stock")

    if predicted.shape != safety.shape:
        raise ValueError("predicted_demand and safety_stock must have matching dimensions.")

    with np.errstate(over="ignore"):
        required = predicted + safety
    if not np.all(np.isfinite(required)):
        raise ValueError("Required stock calculation produced non-finite values.")
    return required


def calculate_order_quantity(required_stock: NumericVector, current_stock: NumericVector) -> np.ndarray:
    """Calculate order quantity with element-wise max(0, required_stock - current_stock)."""
    required = _coerce_vector(required_stock, "required_stock")
    current = _coerce_vector(current_stock, "current_stock")

    if required.shape != current.shape:
        raise ValueError("required_stock and current_stock must have matching dimensions.")

    deficit = required - current
    return np.maximum(deficit, 0.0)


def optimize_inventory(
    predicted_demand: NumericVector,
    current_stock: NumericVector,
    safety_stock: NumericVector,
) -> InventoryCalculationResult:
    """Compute the complete inventory optimization result."""
    predicted = _coerce_vector(predicted_demand, "predicted_demand")
    current = _coerce_vector(current_stock, "current_stock")
    safety = _coerce_vector(safety_stock, "safety_stock")

    if predicted.size == 0 or current.size == 0 or safety.size == 0:
        raise ValueError("predicted_demand, current_stock, and safety_stock cannot be empty.")

    if not (predicted.shape == current.shape == safety.shape):
        raise ValueError(
            "predicted_demand, current_stock, and safety_stock must have matching dimensions."
        )

    required_stock = calculate_required_stock(predicted, safety)
    order_quantity = calculate_order_quantity(required_stock, current)

    return InventoryCalculationResult(
        predicted_demand=predicted.copy(),
        current_stock=current.copy(),
        safety_stock=safety.copy(),
        required_stock=required_stock.copy(),
        order_quantity=order_quantity.copy(),
    )


__all__ = [
    "calculate_required_stock",
    "calculate_order_quantity",
    "optimize_inventory",
]
