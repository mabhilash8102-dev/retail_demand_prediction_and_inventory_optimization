"""Member 3: standalone covariance and sales-pattern analysis."""
from .data_processor import prepare_sales_matrix
from .covariance import calculate_covariance, is_symmetric
from .eigen_analysis import analyze_eigenvalues, verify_eigenpairs
from .integration import fit_demand_model, predict_historical_demand, predict_future_demand
from .pattern_detector import detect_patterns

__all__ = [
    "prepare_sales_matrix",
    "calculate_covariance",
    "is_symmetric",
    "analyze_eigenvalues",
    "verify_eigenpairs",
    "detect_patterns",
    "fit_demand_model",
    "predict_historical_demand",
    "predict_future_demand",
]

