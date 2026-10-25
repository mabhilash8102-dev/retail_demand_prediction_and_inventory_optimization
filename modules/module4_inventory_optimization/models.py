from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np


@dataclass
class InventoryCalculationResult:
    predicted_demand: Union[np.ndarray, list, tuple]
    current_stock: Union[np.ndarray, list, tuple]
    safety_stock: Union[np.ndarray, list, tuple]
    required_stock: Union[np.ndarray, list, tuple]
    order_quantity: Union[np.ndarray, list, tuple]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_demand": np.asarray(self.predicted_demand, dtype=float).tolist(),
            "current_stock": np.asarray(self.current_stock, dtype=float).tolist(),
            "safety_stock": np.asarray(self.safety_stock, dtype=float).tolist(),
            "required_stock": np.asarray(self.required_stock, dtype=float).tolist(),
            "order_quantity": np.asarray(self.order_quantity, dtype=float).tolist(),
        }


@dataclass
class ProductRecommendation:
    product_name: str
    predicted_demand: float
    current_stock: float
    safety_stock: float
    required_stock: float
    recommended_order_quantity: float
    stock_status: str
    overall_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_name": self.product_name,
            "predicted_demand": float(self.predicted_demand),
            "current_stock": float(self.current_stock),
            "safety_stock": float(self.safety_stock),
            "required_stock": float(self.required_stock),
            "recommended_order_quantity": float(self.recommended_order_quantity),
            "stock_status": self.stock_status,
            "overall_message": self.overall_message,
        }


@dataclass
class ConstraintValidationResult:
    exceeds_capacity: Union[bool, list[bool], np.ndarray]
    maximum_capacity: Optional[Union[float, list[float], np.ndarray]]
    required_stock: Optional[Union[float, list[float], np.ndarray]]
    valid: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        maximum_capacity = self.maximum_capacity
        required_stock = self.required_stock
        if isinstance(maximum_capacity, np.ndarray):
            maximum_capacity = maximum_capacity.tolist()
        if isinstance(required_stock, np.ndarray):
            required_stock = required_stock.tolist()
        if isinstance(self.exceeds_capacity, np.ndarray):
            exceeds_capacity = self.exceeds_capacity.tolist()
        else:
            exceeds_capacity = self.exceeds_capacity
        return {
            "exceeds_capacity": exceeds_capacity,
            "maximum_capacity": maximum_capacity,
            "required_stock": required_stock,
            "valid": self.valid,
            "message": self.message,
        }


__all__ = [
    "InventoryCalculationResult",
    "ProductRecommendation",
    "ConstraintValidationResult",
]
