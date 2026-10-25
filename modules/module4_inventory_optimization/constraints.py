from __future__ import annotations

import numpy as np
from typing import Optional, Union

from .config import NUMERICAL_TOLERANCE
from .models import ConstraintValidationResult

NumericInput = Union[float, list[float], tuple[float, ...], np.ndarray]


def _coerce_numeric_vector(values: NumericInput, name: str) -> np.ndarray:
    if values is None:
        raise TypeError(f"{name} cannot be None.")

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be numeric, not a string.")

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
        raise ValueError(f"{name} cannot contain negative values.")

    return array.astype(float, copy=False)


def validate_capacity(
    required_stock: NumericInput,
    maximum_capacity: Optional[NumericInput] = None,
) -> ConstraintValidationResult:
    """Validate whether required stock exceeds the configured maximum storage capacity."""
    required = _coerce_numeric_vector(required_stock, "required_stock")

    if maximum_capacity is None:
        return ConstraintValidationResult(
            exceeds_capacity=False,
            maximum_capacity=None,
            required_stock=required.copy(),
            valid=True,
            message="No maximum capacity constraint supplied; capacity validation is not enforced.",
        )

    capacity = _coerce_numeric_vector(maximum_capacity, "maximum_capacity")

    if capacity.size == 1 and required.size > 1:
        capacity = np.full(required.shape, float(capacity[0]), dtype=float)
    elif required.shape != capacity.shape:
        raise ValueError("required_stock and maximum_capacity must have matching dimensions.")

    if np.any(capacity < 0):
        raise ValueError("maximum_capacity cannot contain negative values.")

    exceeds = required > capacity + NUMERICAL_TOLERANCE
    valid = not bool(np.any(exceeds))

    if required.size == 1:
        exceeds_value = bool(exceeds[0])
    else:
        exceeds_value = exceeds.tolist()

    return ConstraintValidationResult(
        exceeds_capacity=exceeds_value,
        maximum_capacity=capacity.copy() if capacity.size > 1 else float(capacity[0]),
        required_stock=required.copy(),
        valid=valid,
        message=(
            "Capacity constraint satisfied."
            if valid
            else "Required stock exceeds the configured capacity. Adjust order quantities or storage limits."
        ),
    )


def apply_minimum_order_quantity(
    order_quantity: NumericInput,
    minimum_order_quantity: Optional[NumericInput],
    enabled: bool = False,
) -> np.ndarray:
    """Apply minimum order quantity only when explicitly enabled.

    This function does not change zero-order quantities. It also keeps an adjusted order subject to
    separate storage-capacity validation.
    """
    order = _coerce_numeric_vector(order_quantity, "order_quantity")

    if minimum_order_quantity is None:
        return order.copy()

    minimum = _coerce_numeric_vector(minimum_order_quantity, "minimum_order_quantity")

    if minimum.size == 1 and order.size > 1:
        minimum = np.full(order.shape, float(minimum[0]), dtype=float)
    elif order.shape != minimum.shape:
        raise ValueError("order_quantity and minimum_order_quantity must have matching dimensions.")

    if not enabled:
        return order.copy()

    adjusted = order.copy()
    mask = (order > 0) & (minimum > 0)
    adjusted[mask] = np.maximum(order[mask], minimum[mask])
    return adjusted


__all__ = [
    "validate_capacity",
    "apply_minimum_order_quantity",
]
