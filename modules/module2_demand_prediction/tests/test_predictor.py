import numpy as np
import pytest

from module2_demand_prediction import fit_least_squares, predict, predict_from_matrix


def test_historical_prediction_for_exact_relationship():
    A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
    result = fit_least_squares(A, [[3], [5], [7], [9]])

    predicted = predict_from_matrix(A, result.coefficients)

    np.testing.assert_allclose(predicted, [[3], [5], [7], [9]], atol=1e-10)


def test_future_prediction_uses_explicit_feature_order():
    result = predict(
        [1, 2],
        {"day": 5},
        feature_order=["intercept", "day"],
    )
    assert result == pytest.approx(11)


def test_future_prediction_supports_member_one_feature_order():
    result = predict(
        [10, 2, -0.5, 3],
        {"day": 11, "price": 30, "promotion": 1},
        feature_order=["intercept", "day", "price", "promotion"],
    )
    assert result == pytest.approx(20.0)


def test_future_prediction_rejects_ambiguous_or_missing_feature_order():
    with pytest.raises(ValueError, match="feature_order"):
        predict([1, 2], {"day": 5})
    with pytest.raises(ValueError, match="Missing future feature 'day'"):
        predict([1, 2], {}, ["intercept", "day"])
    with pytest.raises(ValueError, match="duplicate"):
        predict([1, 2], {"day": 5}, ["day", "day"])


def test_historical_prediction_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="columns"):
        predict_from_matrix([[1, 2]], [1])
