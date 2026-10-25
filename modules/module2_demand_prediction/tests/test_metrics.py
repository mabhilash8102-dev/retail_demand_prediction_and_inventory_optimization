import pytest

from module2_demand_prediction import calculate_metrics


def test_metrics_match_hand_calculation():
    metrics = calculate_metrics([1, 2, 3], [2, 2, 5])

    assert metrics.mae == pytest.approx(1.0)
    assert metrics.mse == pytest.approx(5 / 3)
    assert metrics.rmse == pytest.approx((5 / 3) ** 0.5)


def test_exact_fit_has_zero_metrics():
    metrics = calculate_metrics([[3], [5]], [[3], [5]])

    assert metrics.mae == pytest.approx(0)
    assert metrics.mse == pytest.approx(0)
    assert metrics.rmse == pytest.approx(0)
