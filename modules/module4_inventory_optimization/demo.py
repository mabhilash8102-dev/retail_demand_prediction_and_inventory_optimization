from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from module4_inventory_optimization import generate_recommendations
else:
    from . import generate_recommendations


def load_inventory_rows():
    csv_path = Path(__file__).resolve().parent / "data" / "sample_inventory.csv"
    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    return rows


def main():
    rows = load_inventory_rows()
    products = [row["product"] for row in rows]
    predicted_demand = [float(row["predicted_demand"]) for row in rows]
    current_stock = [float(row["current_stock"]) for row in rows]
    safety_stock = [float(row["safety_stock"]) for row in rows]

    recommendations = generate_recommendations(products, predicted_demand, current_stock, safety_stock)
    total_order = sum(rec.recommended_order_quantity for rec in recommendations)

    print("========================================")
    print("INVENTORY OPTIMIZATION DEMO")
    print("========================================")
    print()
    print(f"{'Product':<8} {'Demand':>8} {'Current':>8} {'Safety':>8} {'Required':>10} {'Order':>8} {'Status':>12}")
    for rec in recommendations:
        print(
            f"{rec.product_name:<8} "
            f"{rec.predicted_demand:>8.0f} "
            f"{rec.current_stock:>8.0f} "
            f"{rec.safety_stock:>8.0f} "
            f"{rec.required_stock:>10.0f} "
            f"{rec.recommended_order_quantity:>8.0f} "
            f"{rec.stock_status:>12}"
        )
    print()
    print(f"Total recommended order: {total_order:.0f} units")


if __name__ == "__main__":
    main()
