"""Least-squares demand prediction for Member 2."""

from .least_squares import SingularMatrixError, fit_least_squares
from .metrics import calculate_metrics
from .models import LeastSquaresResult, PredictionMetrics, PredictionResult
from .predictor import predict, predict_from_matrix
from .residuals import calculate_residuals, check_residual_orthogonality

__all__ = [
    "LeastSquaresResult",
    "PredictionMetrics",
    "PredictionResult",
    "SingularMatrixError",
    "calculate_metrics",
    "calculate_residuals",
    "check_residual_orthogonality",
    "fit_least_squares",
    "predict",
    "predict_from_matrix",
]
