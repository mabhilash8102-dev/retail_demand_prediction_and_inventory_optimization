import numpy as np
import pytest

from module4_inventory_optimization import (
    calculate_order_quantity,
    calculate_required_stock,
    optimize_inventory,
)


def test_calculate_required_stock_valid():
    predicted = np.array([75, 60])
    safety = np.array([10, 5])
    required = calculate_required_stock(predicted, safety)
    np.testing.assert_allclose(required, np.array([85.0, 65.0]))


def test_calculate_required_stock_rejects_mismatch():
    with pytest.raises(ValueError, match="matching dimensions"):
        calculate_required_stock([75, 60], [10])


def test_calculate_required_stock_rejects_negative_values():
    with pytest.raises(ValueError, match="negative"):
        calculate_required_stock([75, -1], [10, 5])


@pytest.mark.parametrize("invalid", [[float("nan")], [float("inf")], ["not-a-number"]])
def test_calculate_required_stock_rejects_invalid_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        calculate_required_stock(invalid, [0])


def test_calculate_required_stock_supports_zero_safety_stock():
    np.testing.assert_allclose(calculate_required_stock([1, 2], [0, 0]), [1, 2])


def test_calculate_required_stock_rejects_non_vector_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_required_stock([[1, 2]], [[0, 0]])


def test_calculate_required_stock_rejects_overflow():
    with pytest.raises(ValueError, match="non-finite"):
        calculate_required_stock([np.finfo(float).max], [np.finfo(float).max])


def test_calculate_order_quantity_valid():
    required = np.array([85.0, 65.0, 50.0])
    current = np.array([30.0, 50.0, 80.0])
    order = calculate_order_quantity(required, current)
    np.testing.assert_allclose(order, np.array([55.0, 15.0, 0.0]))


@pytest.mark.parametrize("invalid", [[-1], [float("nan")], [float("inf")], ["invalid"]])
def test_calculate_order_quantity_rejects_invalid_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        calculate_order_quantity(invalid, [0])


def test_calculate_order_quantity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="matching dimensions"):
        calculate_order_quantity([10, 20], [1])


def test_calculate_required_stock_rejects_empty_vector():
    with pytest.raises(ValueError, match="empty"):
        calculate_required_stock([], [])


def test_optimize_inventory_supports_multiple_input_types():
    result = optimize_inventory([75, 60], (30, 50), np.array([10, 5]))
    assert result.predicted_demand.tolist() == [75.0, 60.0]
    assert result.current_stock.tolist() == [30.0, 50.0]
    assert result.required_stock.tolist() == [85.0, 65.0]
    assert result.order_quantity.tolist() == [55.0, 15.0]


def test_optimize_inventory_rejects_empty_vectors():
    with pytest.raises(ValueError, match="empty"):
        optimize_inventory([], [], [])


def test_optimize_inventory_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="matching dimensions"):
        optimize_inventory([1, 2], [1], [0, 0])


def test_inventory_result_serializes_to_json():
    import json

    result = optimize_inventory([75], [30], [10])
    serialized = json.dumps(result.to_dict())
    assert '"required_stock": [85.0]' in serialized
