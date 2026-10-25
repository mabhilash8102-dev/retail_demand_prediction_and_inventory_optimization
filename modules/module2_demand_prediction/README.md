# Member 2: Least-Squares Demand Prediction

This package solves the least-squares demand model from a design matrix `A`
and observed demand vector `b`. It is independently testable and does not load
or clean CSV data; Member 1 remains responsible for constructing `A` and `b`.

## Mathematical model

The observed system is generally overdetermined:

```text
A X ≈ b
```

`A` contains one observation per row and one model feature per column. `X`
contains one coefficient per column, including an intercept coefficient when
the first column of `A` is the intercept. The predicted historical demand is
`b_hat = A X`.

Least squares minimizes the sum of squared residuals. The implementation
explicitly calculates `Aᵀ`, `AᵀA`, and `Aᵀb`, then solves:

```text
(AᵀA)X = Aᵀb
```

When `AᵀA` is invertible, this is mathematically equivalent to
`X = (AᵀA)^(-1)Aᵀb`. In code, solving the linear system is preferable to
forming an inverse because it avoids unnecessary work and is generally more
numerically reliable. The design matrix is checked for deficient or
near-deficient rank; such inputs raise `SingularMatrixError` rather than
silently switching to a pseudoinverse.

The demo varies a few prices from the illustrative retail table in the
specification. In that exact table, price is `30 - 2 * promotion`, making
those two design columns linearly dependent and the coefficient vector
non-unique; the adjusted example demonstrates a full-rank fit instead.

The fitted vector `b_hat` is the orthogonal projection of `b` onto the column
space of `A`, so `b_hat ∈ Col(A)`. The residual vector is `e = b - b_hat`.
At a least-squares solution it is orthogonal to each column of `A`, which the
demo verifies numerically with `Aᵀe ≈ 0`.

The package also reports mean absolute error (MAE), mean squared error (MSE),
and root mean squared error (RMSE) between actual and predicted values.

## Package structure

```text
module2_demand_prediction/
├── least_squares.py
├── predictor.py
├── residuals.py
├── metrics.py
├── models.py
├── config.py
├── __init__.py
├── data/
│   └── sample_prediction_input.py
├── tests/
│   ├── test_least_squares.py
│   ├── test_predictor.py
│   ├── test_residuals.py
│   └── test_metrics.py
├── demo.py
├── requirements.txt
└── README.md
```

## Public API

```python
from module2_demand_prediction import (
    calculate_metrics,
    calculate_residuals,
    fit_least_squares,
    predict,
    predict_from_matrix,
)

result = fit_least_squares(A, b)
historical = predict_from_matrix(A, result.coefficients)
residuals = calculate_residuals(b, historical)
metrics = calculate_metrics(b, historical)
```

`fit_least_squares` accepts nested Python lists or NumPy arrays. It returns a
`LeastSquaresResult` with column-vector `coefficients`, `A_transpose`, `AtA`,
`Atb`, and the design-matrix rank. `b` may be a flat vector or a column vector.

Future prediction requires an explicit `feature_order`, with exactly one
entry for each coefficient. The reserved name `"intercept"` contributes 1;
every other name is looked up in the supplied feature mapping. For example,
if the columns are `[intercept, day, price, promotion]`:

```python
future_demand = predict(
    result.coefficients,
    {"day": 11, "price": 30, "promotion": 1},
    feature_order=["intercept", "day", "price", "promotion"],
)
```

This explicit order must match the columns used when building `A`; it is not
inferred from the mapping.

## Standalone use

Install dependencies and run the tests/demo from the project root (the
directory containing both member packages):

```bash
python -m pip install -r module2_demand_prediction/requirements.txt
python -m pytest module2_demand_prediction/tests -q
python -m module2_demand_prediction.demo
```

The tests include the exact standalone system
`A = [[1,1], [1,2], [1,3], [1,4]]`, `b = [[3], [5], [7], [9]]`, whose
coefficients are `[1, 2]`; neither tests nor demo depend on Member 1.

## Integration boundary with Member 1

Member 1's current `build_matrix` returns a `MatrixData` object with `.matrix`
and `.target_vector` nested lists. Pass those directly to Member 2:

```python
from module1_data_matrix import build_matrix
from module2_demand_prediction import fit_least_squares, predict

matrix_data = build_matrix(records, features=["day", "price", "promotion"])
result = fit_least_squares(matrix_data.matrix, matrix_data.target_vector)
future_demand = predict(
    result.coefficients,
    {"day": 11, "price": 30, "promotion": 1},
    feature_order=matrix_data.features,
)
```

Member 2 consumes Member 1's prepared matrix/vector and does not read the CSV.
This package supplies a stable consumer API; it does not claim that the entire
multi-member project has been integrated.

## Scope

Member 2 owns least-squares fitting, historical and future demand prediction,
residuals, error metrics, and the projection/orthogonality demonstration. CSV
loading and cleaning, matrix construction and RREF/rank belong to Member 1;
eigen-analysis belongs to Member 3; inventory and ordering recommendations
belong to Member 4 and are deliberately not implemented here.
