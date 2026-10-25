"""Validate, extract, and mean-center named sales columns."""
import math
from .config import DEFAULT_TOLERANCE
from .models import PreparedSales

def prepare_sales_matrix(data, columns, tolerance=DEFAULT_TOLERANCE):
    """Prepare a mapping from names to value lists; rejects missing/non-numeric cells."""
    if isinstance(columns, str) or not hasattr(columns, "__iter__"):
        raise TypeError("Selected columns must be provided as a sequence of names")
    columns = list(columns)
    if not columns:
        raise ValueError("Select at least one product column")
    if len(set(columns)) != len(columns):
        raise ValueError("Selected columns must be unique")
    if not hasattr(data, "keys"):
        raise TypeError("Pass a mapping from column names to lists of values")
    missing = [name for name in columns if name not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    try:
        lengths = {len(data[name]) for name in columns}
    except TypeError as exc:
        raise ValueError("Selected sales columns must contain sequence-like values") from exc
    if len(lengths) != 1:
        raise ValueError("Selected columns must have equal lengths")
    n = lengths.pop()
    if n < 2:
        raise ValueError("At least two observations are required")
    try:
        matrix = [[float(data[name][i]) for name in columns] for i in range(n)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("Selected sales values must be numeric") from exc
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError("Sales data cannot contain missing or infinite values")
    p = len(columns)
    means = [sum(row[j] for row in matrix) / n for j in range(p)]
    centered = [[row[j] - means[j] for j in range(p)] for row in matrix]
    if any(abs(sum(row[j] for row in centered) / n) > tolerance for j in range(p)):
        raise ArithmeticError("Centering check failed")
    return PreparedSales(matrix, means, centered, list(columns))
