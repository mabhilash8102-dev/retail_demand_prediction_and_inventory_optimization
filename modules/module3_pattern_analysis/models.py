"""Data containers returned by analysis functions."""
from dataclasses import dataclass, field

@dataclass
class PreparedSales:
    original_matrix: list[list[float]]
    means: list[float]
    centered_matrix: list[list[float]]
    column_names: list[str]

@dataclass
class CovarianceResult:
    covariance_matrix: list[list[float]]

@dataclass
class EigenAnalysisResult:
    eigenvalues: list[float]
    eigenvectors: list[list[float]]  # each inner list is an eigenvector
    explained_variance: list[float]
    cumulative_explained_variance: list[float]
    total_variance: float = 0.0
    explained_variance_percentages: list[float] = field(default_factory=list)
    cumulative_explained_variance_percentages: list[float] = field(default_factory=list)

@dataclass
class PatternResult:
    dominant_eigenvalue: float
    dominant_eigenvector: list[float]
    product_contributions: dict[str, float]
