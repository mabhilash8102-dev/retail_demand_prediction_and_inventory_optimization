import json

import numpy as np
import pytest

from module4_inventory_optimization import apply_minimum_order_quantity, validate_capacity


def test_validate_capacity_with_limit_and_no_excess():
    result = validate_capacity([85, 65], maximum_capacity=100)
    assert result.valid is True
    assert result.exceeds_capacity == [False, False]


def test_validate_capacity_detects_excess():
    result = validate_capacity([85, 65], maximum_capacity=[80, 60])
    assert result.valid is False
    assert result.exceeds_capacity == [True, True]


def test_validate_capacity_rejects_negative_value():
    with pytest.raises(ValueError, match="negative"):
        validate_capacity([85], maximum_capacity=-1)


@pytest.mark.parametrize("capacity", [float("nan"), float("inf"), "invalid"])
def test_validate_capacity_rejects_non_finite_or_non_numeric_values(capacity):
    with pytest.raises((TypeError, ValueError)):
        validate_capacity([85], maximum_capacity=capacity)


def test_validate_capacity_reports_violation_per_product():
    result = validate_capacity([85, 65], maximum_capacity=[90, 60])
    assert result.valid is False
    assert result.exceeds_capacity == [False, True]
    assert "exceeds" in result.message


def test_validate_capacity_uses_numerical_tolerance():
    from module4_inventory_optimization.config import NUMERICAL_TOLERANCE

    result = validate_capacity([10 + NUMERICAL_TOLERANCE / 2], [10])
    assert result.valid is True


def test_validate_capacity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="matching dimensions"):
        validate_capacity([10, 20], [10, 20, 30])


def test_validate_capacity_when_constraint_is_omitted():
    result = validate_capacity([10, 20])
    assert result.valid is True
    assert result.exceeds_capacity is False


def test_constraint_result_serializes_to_json():
    result = validate_capacity([85, 65], [90, 60])
    serialized = json.dumps(result.to_dict())
    assert '"exceeds_capacity": [false, true]' in serialized


def test_apply_minimum_order_quantity_when_enabled():
    order = np.array([5.0, 12.0, 0.0])
    adjusted = apply_minimum_order_quantity(order, 10, enabled=True)
    np.testing.assert_allclose(adjusted, np.array([10.0, 12.0, 0.0]))


def test_apply_minimum_order_quantity_keeps_zero_order():
    order = np.array([0.0, 5.0])
    adjusted = apply_minimum_order_quantity(order, 10, enabled=True)
    np.testing.assert_allclose(adjusted, np.array([0.0, 10.0]))


def test_apply_minimum_order_quantity_disabled_leaves_order():
    order = np.array([5.0, 12.0])
    adjusted = apply_minimum_order_quantity(order, 10, enabled=False)
    np.testing.assert_allclose(adjusted, order)


def test_apply_minimum_order_quantity_accepts_zero_minimum():
    np.testing.assert_allclose(
        apply_minimum_order_quantity([0, 5], 0, enabled=True), [0, 5]
    )


@pytest.mark.parametrize("minimum", [-1, float("nan"), float("inf"), "invalid"])
def test_apply_minimum_order_quantity_rejects_invalid_minimum(minimum):
    with pytest.raises((TypeError, ValueError)):
        apply_minimum_order_quantity([5], minimum, enabled=False)


def test_apply_minimum_order_quantity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="matching dimensions"):
        apply_minimum_order_quantity([5, 10], [1, 2, 3], enabled=True)


def test_apply_minimum_order_quantity_rejects_invalid_order_quantity():
    with pytest.raises(ValueError, match="negative"):
        apply_minimum_order_quantity([-1], 10, enabled=True)
