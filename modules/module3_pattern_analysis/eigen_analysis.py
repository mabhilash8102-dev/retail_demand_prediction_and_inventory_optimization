"""Jacobi eigenanalysis for symmetric matrices, implemented in pure Python."""
import math

from .config import DEFAULT_TOLERANCE
from .models import EigenAnalysisResult
from .covariance import is_symmetric


def _jacobi(matrix, tolerance, max_iterations=10000):
    """Return eigenvalues and normalized eigenvectors (as rows) for symmetric matrix."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_iterations):
        p, q, largest = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    p, q, largest = i, j, abs(a[i][j])
        if largest <= tolerance:
            break
        tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
        t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + (1.0 + tau * tau) ** 0.5)
        c = 1.0 / (1.0 + t * t) ** 0.5
        s = t * c
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k != p and k != q:
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    else:
        raise ArithmeticError("Jacobi eigensolver did not converge")
    return [a[i][i] for i in range(n)], [[v[row][col] for row in range(n)] for col in range(n)]


def _sanitize_eigenvalues(values, tolerance):
    """Reject materially invalid negative eigenvalues while accepting tiny numerical noise."""
    cleaned = []
    ghost_tolerance = max(1e-12, tolerance * 1000.0)
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Eigenvalues must be finite numbers")
        if value < 0 and abs(value) <= ghost_tolerance:
            cleaned.append(0.0)
        elif value < 0:
            raise ValueError(
                "Covariance matrix has a materially invalid negative eigenvalue; "
                "this indicates a non-positive-semidefinite matrix or unstable input"
            )
        else:
            cleaned.append(float(value))
    return cleaned


def analyze_eigenvalues(covariance_matrix, tolerance=DEFAULT_TOLERANCE):
    """Calculate descending eigenvalues and corresponding eigenvectors."""
    if covariance_matrix is None or not hasattr(covariance_matrix, "__iter__"):
        raise ValueError("Covariance matrix must be a non-empty square matrix")
    try:
        c = [[float(value) for value in row] for row in covariance_matrix]
    except TypeError as exc:
        raise ValueError("Covariance matrix must contain numeric values") from exc
    if not c:
        raise ValueError("Covariance matrix must be a non-empty square matrix")
    if any(len(row) != len(c) for row in c):
        raise ValueError("Covariance matrix must be square")
    if not is_symmetric(c, tolerance):
        raise ValueError("Covariance matrix must be square and symmetric")
    if any(not math.isfinite(value) for row in c for value in row):
        raise ValueError("Covariance matrix contains non-finite values")
    values, vectors = _jacobi(c, tolerance)
    values = _sanitize_eigenvalues(values, tolerance)
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    values = [values[i] for i in order]
    vectors = [vectors[i] for i in order]
    total = sum(values)
    if total < 0 and abs(total) <= max(1e-12, tolerance * 1000.0):
        total = 0.0
    elif total < 0:
        raise ValueError("Total variance must be non-negative")
    if total <= tolerance:
        explained = [0.0] * len(values)
    else:
        explained = [value / total for value in values]
    cumulative, running = [], 0.0
    for share in explained:
        running += share
        cumulative.append(running)
    percentages = [share * 100.0 for share in explained]
    cumulative_percentages = [value * 100.0 for value in cumulative]
    return EigenAnalysisResult(
        values,
        vectors,
        explained,
        cumulative,
        total_variance=float(total),
        explained_variance_percentages=percentages,
        cumulative_explained_variance_percentages=cumulative_percentages,
    )


def verify_eigenpairs(matrix, analysis, tolerance=1e-8):
    """Check C v = lambda v for every returned eigenpair."""
    if not matrix or not analysis.eigenvalues:
        return False
    for value, vector in zip(analysis.eigenvalues, analysis.eigenvectors):
        product = [sum(matrix[i][j] * vector[j] for j in range(len(vector))) for i in range(len(vector))]
        if any(abs(product[i] - value * vector[i]) > tolerance for i in range(len(vector))):
            return False
    return True
