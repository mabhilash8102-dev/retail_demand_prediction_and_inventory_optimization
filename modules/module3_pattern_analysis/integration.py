"""Small compatibility bridge to the standalone Module 2 demand-prediction API."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from module2_demand_prediction import fit_least_squares as _fit_least_squares
from module2_demand_prediction import predict as _predict
from module2_demand_prediction import predict_from_matrix as _predict_from_matrix


def fit_demand_model(A, b, tolerance=1e-10):
    """Call Module 2's least-squares solver with the same arguments."""
    return _fit_least_squares(A, b, tolerance=tolerance)


def predict_historical_demand(A, coefficients):
    """Return historical demand predictions using Module 2's matrix predictor."""
    return _predict_from_matrix(A, coefficients)


def predict_future_demand(coefficients, features, feature_order):
    """Predict one future demand point using Module 2's explicit feature-order API."""
    if feature_order is None:
        raise ValueError("feature_order is required by the Module 2 predictor")
    return _predict(coefficients, features, feature_order=tuple(feature_order))


__all__ = [
    "fit_demand_model",
    "predict_future_demand",
    "predict_historical_demand",
]
