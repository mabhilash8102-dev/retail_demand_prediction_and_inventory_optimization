# Member 1: Historical Sales Data Processing and Matrix Construction

This project prepares the historical sales matrix for the larger retail-demand model. It validates CSV data, builds the design matrix for linear regression, and performs essential matrix operations such as transpose, multiplication, rank, and RREF.

## 1. Required data format

The expected CSV columns are:

- day
- product
- price
- promotion
- sales

The project uses the sample file at `data/sample_sales.csv`, which contains 10 valid records. The sample dataset is also mirrored in the root-level `sales_data.csv` file for compatibility with the original workspace.

## 2. Data validation requirements

The loader rejects:

- missing files
- empty CSV files
- missing columns
- malformed rows
- blank values
- NaN or infinity
- price <= 0
- sales < 0
- promotion values other than 0 or 1

Invalid rows are not silently skipped.

## 3. Matrix construction

The default feature list is:

```python
["day", "price", "promotion"]
```

The feature matrix includes an intercept column first, so each row is structured as:

```python
[1, day, price, promotion]
```

with the target vector `b` containing the observed sales values.

For 10 observations, the resulting matrix shapes are:

- `A`: `(10, 4)`
- `b`: `(10, 1)`

The `day` value is mapped to a numeric weekday index, where Monday = 1 through Sunday = 7.

## 4. Linear algebra operations

The project includes manual implementations of:

- transpose
- matrix multiplication
- rank
- reduced row echelon form (RREF)

The RREF routine uses Gaussian elimination with pivot selection, row swapping, pivot normalization, and elimination. It does not call `numpy.linalg.matrix_rank` or a built-in RREF implementation.

## 5. Project structure

```text
module1_data_matrix/
├── data_loader.py
├── data_validator.py
├── matrix_builder.py
├── matrix_operations.py
├── models.py
├── config.py
├── __init__.py
├── data/
│   └── sample_sales.csv
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_validator.py
│   ├── test_matrix_builder.py
│   └── test_matrix_operations.py
├── demo.py
├── main.py
├── requirements.txt
├── sales_data.csv
└── README.md
```

## 6. Public API

```python
from module1_data_matrix import (
    load_sales_data,
    validate_row,
    validate_records,
    build_matrix,
    construct_matrices,
    transpose,
    matrix_multiply,
    rref,
    rank,
)
```

## 7. Running the demo

From the parent directory containing `module1_data_matrix/`:

```bash
python -m pytest module1_data_matrix/tests -q
python module1_data_matrix/demo.py
```

The demo loads the sample CSV, builds the matrix, checks the final shapes, and prints representative matrix operations.

## 8. Notes

This module only covers Member 1 responsibilities: data cleaning, matrix preparation, and linear algebra utilities. Demand prediction and inventory optimization belong to other members.
