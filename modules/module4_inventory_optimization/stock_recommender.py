from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import NUMERICAL_TOLERANCE
from .inventory_optimizer import NumericVector, optimize_inventory
from .models import ProductRecommendation


def _status_for(current_stock: float, required_stock: float) -> str:
    if current_stock < required_stock - NUMERICAL_TOLERANCE:
        return "UNDERSTOCKED"
    if abs(current_stock - required_stock) <= NUMERICAL_TOLERANCE:
        return "ADEQUATE"
    return "OVERSTOCKED"


def generate_recommendations(
    products: Sequence[str],
    predicted_demand: NumericVector,
    current_stock: NumericVector,
    safety_stock: NumericVector,
) -> list[ProductRecommendation]:
    """Generate product-level stock recommendations for each product entry."""
    if isinstance(products, (str, bytes)):
        raise TypeError("products must be a sequence of product names, not a string.")

    product_names = list(products)
    input_sizes = (
        np.asarray(predicted_demand).size,
        np.asarray(current_stock).size,
        np.asarray(safety_stock).size,
    )
    if any(size != len(product_names) for size in input_sizes):
        raise ValueError("The product names and inventory vectors must all contain the same number of entries.")

    calculation = optimize_inventory(predicted_demand, current_stock, safety_stock)
    predicted = calculation.predicted_demand
    current = calculation.current_stock
    safety = calculation.safety_stock
    required = calculation.required_stock
    orders = calculation.order_quantity

    recommendations = []
    for idx, product_name in enumerate(product_names):
        demand = float(predicted[idx])
        stock = float(current[idx])
        safety_stock = float(safety[idx])
        required_stock = float(required[idx])
        order_quantity = float(orders[idx])
        status = _status_for(stock, required_stock)

        if status == "UNDERSTOCKED":
            message = (
                f"{product_name} is UNDERSTOCKED. Current stock {stock} is below the required stock "
                f"{required_stock}. Recommended order quantity is {order_quantity}."
            )
        elif status == "ADEQUATE":
            message = (
                f"{product_name} is ADEQUATE. Current stock matches the required stock {required_stock}, "
                f"so no replenishment is required. Order quantity remains {order_quantity}."
            )
        else:
            message = (
                f"{product_name} is OVERSTOCKED. Current stock {stock} exceeds the required stock "
                f"{required_stock}. No replenishment is needed."
            )

        recommendations.append(
            ProductRecommendation(
                product_name=str(product_name),
                predicted_demand=demand,
                current_stock=stock,
                safety_stock=safety_stock,
                required_stock=required_stock,
                recommended_order_quantity=order_quantity,
                stock_status=status,
                overall_message=message,
            )
        )

    return recommendations


__all__ = ["generate_recommendations"]
