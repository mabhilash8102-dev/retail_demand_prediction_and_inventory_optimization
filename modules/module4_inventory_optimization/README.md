# Module 4: Inventory Optimization and Stock Recommendation

This project is Member 4 of the college Linear Algebra mini-project titled "Demand Prediction and Inventory Optimization for Value Retailers Using Linear Algebra".

## Objective

Member 4 receives predicted demand from Member 2 and converts it into a practical inventory recommendation. It uses NumPy vectors to compute the required stock and recommended order quantities without implementing any demand-forecasting model.

## Mathematical model

For each product:

- predicted demand vector: d
- current inventory vector: i
- safety stock vector: s
- required stock vector: r = d + s
- recommended order vector: o = max(0, r - i)

The max operation is applied element-wise, so all products are processed simultaneously.

## Folder structure

```text
module4_inventory_optimization/
├── inventory_optimizer.py
├── stock_recommender.py
├── constraints.py
├── models.py
├── config.py
├── __init__.py
├── data/
│   └── sample_inventory.csv
├── tests/
│   ├── test_inventory_optimizer.py
│   ├── test_stock_recommender.py
│   └── test_constraints.py
├── demo.py
├── requirements.txt
├── README.md
```

## Input data

The included sample CSV contains:

```csv
product,predicted_demand,current_stock,safety_stock
Milk,75,30,10
Bread,60,50,5
Rice,45,60,5
Oil,35,20,5
```

## Public API

```python
from module4_inventory_optimization import (
    calculate_required_stock,
    calculate_order_quantity,
    optimize_inventory,
    generate_recommendations,
    validate_capacity,
    apply_minimum_order_quantity,
)
```

## Example

```python
from module4_inventory_optimization import optimize_inventory

result = optimize_inventory(
    predicted_demand=[75, 60, 45, 35],
    current_stock=[30, 50, 60, 20],
    safety_stock=[10, 5, 5, 5],
)

print(result.required_stock)
print(result.order_quantity)
```

## Constraint handling

Optional constraints are disabled by default.

- `validate_capacity(required_stock, maximum_capacity)` checks required stock per product against a scalar capacity (applied to each product) or a vector of product-specific capacities.
- `apply_minimum_order_quantity(order_quantity, minimum_order_quantity, enabled=False)` raises the order only when explicitly enabled.
- Adjusted orders should still be validated against storage constraints separately.

## Installation and execution

Use Python 3.10 or newer. Run these commands from the workspace parent directory (the directory containing `module4_inventory_optimization/`):

```bash
python -m pip install -r module4_inventory_optimization/requirements.txt
python module4_inventory_optimization/demo.py
python -m pytest module4_inventory_optimization/tests -q
```

## Notes and limitations

- Member 4 receives demand forecasts as input; it does not predict demand itself.
- Capacity is checked per product; this module does not model shared warehouse volume, product dimensions, or cross-product capacity allocation.
- An enabled minimum-order adjustment can increase an order, so validate adjusted required inventory against capacity separately.
- Inputs must be one-dimensional, finite, non-negative numeric vectors with matching lengths.
- The project is intentionally modular and standalone, so it can be combined with the other project members without dependency on their source files.
