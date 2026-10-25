"""Translate the leading eigenvector into product contributions."""
from .models import PatternResult


def detect_patterns(analysis, product_names):
    """Return signed loadings for the dominant sales-variation direction."""
    if not analysis.eigenvalues or not analysis.eigenvectors:
        raise ValueError("Eigen-analysis must include at least one eigenpair")
    if len(product_names) != len(analysis.eigenvalues):
        raise ValueError("Product-name count must match covariance dimensions")
    vector = list(analysis.eigenvectors[0])
    if vector and vector[max(range(len(vector)), key=lambda i: abs(vector[i]))] < 0:
        vector = [-value for value in vector]
    return PatternResult(float(analysis.eigenvalues[0]), vector, dict(zip(product_names, vector)))
