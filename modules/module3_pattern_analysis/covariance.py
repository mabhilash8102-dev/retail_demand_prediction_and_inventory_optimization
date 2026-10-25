"""Transparent sample covariance calculation."""
import math

from .config import DEFAULT_TOLERANCE
from .models import CovarianceResult


def calculate_covariance(centered_matrix):
    """Calculate C = centered.T @ centered / (n - 1) using Python lists."""
    if centered_matrix is None or not hasattr(centered_matrix, "__iter__"):
        raise ValueError("Expected a rectangular 2-D matrix with at least two rows and one column")
    try:
        x = [[float(value) for value in row] for row in centered_matrix]
    except TypeError as exc:
        raise ValueError("Expected a rectangular 2-D matrix with numeric values") from exc
    if not x or len(x) < 2 or not x[0] or any(len(row) != len(x[0]) for row in x):
        raise ValueError("Expected a rectangular 2-D matrix with at least two rows and one column")
    if any(not math.isfinite(value) for row in x for value in row):
        raise ValueError("Covariance matrix cannot contain NaN or infinite values")
    p = len(x[0]); n = len(x)
    c = [[sum(x[k][i] * x[k][j] for k in range(n)) / (n - 1) for j in range(p)] for i in range(p)]
    if any(not math.isfinite(value) for row in c for value in row):
        raise ValueError("Covariance calculation produced non-finite values")
    return CovarianceResult(c)


def is_symmetric(matrix, tolerance=DEFAULT_TOLERANCE):
    """Return whether a square matrix is symmetric within tolerance."""
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return False
    if any(not math.isfinite(value) for row in matrix for value in row):
        return False
    return all(abs(matrix[i][j] - matrix[j][i]) <= tolerance for i in range(len(matrix)) for j in range(len(matrix)))
