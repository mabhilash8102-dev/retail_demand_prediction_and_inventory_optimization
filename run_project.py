"""End-to-end integration for the four linear-algebra modules."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
MODULES_ROOT = ROOT / "modules"
IMPORT_ROOT = MODULES_ROOT if MODULES_ROOT.is_dir() else ROOT
for import_root in (ROOT, IMPORT_ROOT):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from module1_data_matrix.module1_data_matrix.data_loader import load_sales_data
from module1_data_matrix.module1_data_matrix.config import DAY_MAP
from module1_data_matrix.module1_data_matrix.data_validator import validate_records
from module1_data_matrix.module1_data_matrix.matrix_builder import construct_matrices
from module2_demand_prediction import calculate_metrics, calculate_residuals, fit_least_squares, predict
from module3_pattern_analysis import (
    analyze_eigenvalues,
    calculate_covariance,
    detect_patterns,
    prepare_sales_matrix,
)
from module4_inventory_optimization import generate_recommendations, optimize_inventory


MODULE1_SALES_CSV = next(
    path
    for path in (
        ROOT / "data" / "sample_sales.csv",
        ROOT / "module1_data_matrix" / "module1_data_matrix" / "data" / "sample_sales.csv",
        MODULES_ROOT / "module1_data_matrix" / "module1_data_matrix" / "data" / "sample_sales.csv",
    )
    if path.exists()
)
MODULE4_INVENTORY_CSV = next(
    path
    for path in (
        ROOT / "data" / "inventory.csv",
        ROOT / "module4_inventory_optimization" / "data" / "sample_inventory.csv",
        MODULES_ROOT / "module4_inventory_optimization" / "data" / "sample_inventory.csv",
    )
    if path.exists()
)


def _load_inventory_table(path: Path) -> tuple[list[str], list[float], list[float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return _inventory_vectors(rows)


def _inventory_vectors(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[float], list[float]]:
    if not rows:
        raise ValueError("Inventory data contains no products.")
    products = []
    current = []
    safety = []
    for row in rows:
        products.append(row["product"])
        current.append(float(row["current_stock"]))
        safety.append(float(row["safety_stock"]))
    return products, current, safety


def _product_sales_by_day(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], list[str], list[int]]:
    products = list(dict.fromkeys(record["product"] for record in records))
    days = sorted({DAY_MAP[record["day"].strip().lower()] for record in records})
    sales_by_day = {day: {product: 0.0 for product in products} for day in days}
    for record in records:
        day = DAY_MAP[record["day"].strip().lower()]
        sales_by_day[day][record["product"]] += float(record["sales"])
    return (
        {product: [sales_by_day[day][product] for day in days] for product in products},
        products,
        days,
    )


def _forecast_for_products(
    records: list[dict[str, Any]],
    products: list[str],
    coefficients: np.ndarray,
) -> tuple[list[dict[str, float]], list[float]]:
    records_by_product: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_product.setdefault(record["product"].casefold(), []).append(record)

    scenarios: list[dict[str, float]] = []
    forecasts: list[float] = []
    forecast_basis: list[str] = []
    feature_order = ["intercept", "day", "price", "promotion"]
    most_recent_sale = records[-1]
    for product in products:
        history = records_by_product.get(product.casefold())
        last_sale = history[-1] if history else most_recent_sale
        next_day = DAY_MAP[last_sale["day"].strip().lower()] % 7 + 1
        scenario = {
            "day": float(next_day),
            "price": float(last_sale["price"]),
            "promotion": float(last_sale["promotion"]),
        }
        scenarios.append(scenario)
        forecasts.append(predict(coefficients, scenario, feature_order=feature_order))
        forecast_basis.append(
            "Product history" if history else "Latest sales row (product absent from sales input)"
        )
    return scenarios, forecasts, forecast_basis


def run_workflow(
    sales_records: list[dict[str, Any]] | None = None,
    inventory_records: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    """Execute the workflow using supplied validated sales rows or the sample dataset."""
    # Module 1: build the historical design matrix and demand vector.
    records = (
        validate_records(sales_records)
        if sales_records is not None
        else load_sales_data(MODULE1_SALES_CSV)
    )
    A_list, b_list = construct_matrices(records)
    A = np.asarray(A_list, dtype=float)
    b = np.asarray(b_list, dtype=float)
    fit = fit_least_squares(A, b)
    historical_predictions = A @ fit.coefficients
    residuals = calculate_residuals(b, historical_predictions)
    metrics = calculate_metrics(b, historical_predictions)

    if inventory_records is None:
        product_names, current_stock, safety_stock = _load_inventory_table(MODULE4_INVENTORY_CSV)
    else:
        product_names, current_stock, safety_stock = _inventory_vectors(inventory_records)

    # Module 2: forecast each inventory product from its most recent observed sale.
    future_scenarios, forecast, forecast_basis = _forecast_for_products(
        records, product_names, fit.coefficients
    )

    # Module 3: aggregate selected sales by weekday and product for pattern analysis.
    sales_data, pattern_product_names, pattern_days = _product_sales_by_day(records)
    if len(next(iter(sales_data.values()))) < 2:
        raise ValueError("Pattern analysis requires sales data from at least two weekdays.")
    prepared = prepare_sales_matrix(sales_data, pattern_product_names)
    covariance = calculate_covariance(prepared.centered_matrix)
    eigen = analyze_eigenvalues(covariance.covariance_matrix)
    pattern = detect_patterns(eigen, pattern_product_names)

    # Module 4: use the per-product Module 2 forecasts with the active inventory inputs.
    inventory_result = optimize_inventory(
        predicted_demand=forecast,
        current_stock=current_stock,
        safety_stock=safety_stock,
    )
    recommendations = generate_recommendations(
        products=product_names,
        predicted_demand=inventory_result.predicted_demand,
        current_stock=inventory_result.current_stock,
        safety_stock=inventory_result.safety_stock,
    )

    report = {
        "records": records,
        "A": A,
        "b": b,
        "fit": fit,
        "historical_predictions": historical_predictions,
        "residuals": residuals,
        "metrics": metrics,
        "forecast": forecast,
        "future_scenarios": future_scenarios,
        "forecast_basis": forecast_basis,
        "pattern_sales_data": sales_data,
        "pattern_product_names": pattern_product_names,
        "pattern_days": pattern_days,
        "prepared": prepared,
        "covariance": covariance,
        "eigen": eigen,
        "pattern": pattern,
        "product_names": product_names,
        "current_stock": current_stock,
        "safety_stock": safety_stock,
        "inventory_result": inventory_result,
        "recommendations": recommendations,
    }
    return report


def main() -> None:
    report = run_workflow()
    records = report["records"]
    A = report["A"]
    b = report["b"]
    fit = report["fit"]
    residuals = report["residuals"]
    metrics = report["metrics"]
    forecast = report["forecast"]
    eigen = report["eigen"]
    pattern = report["pattern"]
    recommendations = report["recommendations"]

    print("=" * 70)
    print("DEMAND PREDICTION AND INVENTORY OPTIMIZATION REPORT")
    print("=" * 70)
    print(f"Historical records loaded: {len(records)}")
    print(f"Design matrix A shape: {A.shape if hasattr(A, 'shape') else (len(A), len(A[0]))}")
    print(f"Target vector b shape: {b.shape if hasattr(b, 'shape') else (len(b), len(b[0]))}")
    print("Least-squares coefficients:")
    print(f"  {[round(float(value)) for value in fit.coefficients.flatten()]}")
    print(f"Matrix rank: {fit.rank}")
    print(
        f"MAE={metrics.mae:.0f}, MSE={metrics.mse:.0f}, "
        f"RMSE={metrics.rmse:.0f}"
    )
    residual_check = (A.T @ residuals).flatten()
    print(f"Residual check (A^T e): {[round(float(value)) for value in residual_check]}")

    print("\nFuture demand forecasts:")
    for index, value in enumerate(forecast, start=1):
        print(f"  Scenario {index}: {value:.0f} units")

    print("\nSales pattern summary:")
    print(f"  Total variance: {eigen.total_variance:.0f}")
    print(f"  Dominant eigenvalue: {eigen.eigenvalues[0]:.0f}")
    print(
        "  Explained variance shares: "
        f"{[round(float(value)) for value in eigen.explained_variance_percentages]}%"
    )
    print(
        "  Dominant pattern contributions: "
        f"{ {product: round(float(value)) for product, value in pattern.product_contributions.items()} }"
    )

    print("\nInventory recommendations:")
    for recommendation in recommendations:
        print(f"  - {recommendation.product_name}: {recommendation.stock_status} | "
              f"required={recommendation.required_stock:.0f}, order={recommendation.recommended_order_quantity:.0f}, "
              f"predicted_demand={recommendation.predicted_demand:.0f}, "
              f"current_stock={recommendation.current_stock:.0f}, "
              f"safety_stock={recommendation.safety_stock:.0f}")

    print("\nProject flow: Module 1 -> Module 2 -> Module 3 -> Module 4")


if __name__ == "__main__":
    main()
