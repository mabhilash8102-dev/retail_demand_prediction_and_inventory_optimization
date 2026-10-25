import pytest

from module4_inventory_optimization import generate_recommendations


def test_generate_recommendations_for_multiple_products():
    products = ["Milk", "Bread", "Rice", "Beans"]
    predicted = [75, 60, 45, 50]
    current = [30, 50, 60, 50]
    safety = [10, 5, 5, 0]

    recommendations = generate_recommendations(products, predicted, current, safety)
    assert len(recommendations) == 4
    assert recommendations[0].stock_status == "UNDERSTOCKED"
    assert recommendations[0].recommended_order_quantity == 55.0
    assert recommendations[1].stock_status == "UNDERSTOCKED"
    assert recommendations[1].recommended_order_quantity == 15.0
    assert recommendations[2].stock_status == "OVERSTOCKED"
    assert recommendations[2].recommended_order_quantity == 0.0
    assert recommendations[3].stock_status == "ADEQUATE"
    assert recommendations[3].recommended_order_quantity == 0.0


def test_generate_recommendations_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number of entries"):
        generate_recommendations(["Milk", "Bread"], [75], [30, 50], [10, 5])


def test_generate_recommendations_maps_products_to_values():
    recommendations = generate_recommendations(
        ["Milk", "Bread"], [75, 60], [30, 50], [10, 5]
    )
    assert recommendations[0].product_name == "Milk"
    assert recommendations[0].predicted_demand == 75
    assert recommendations[0].required_stock == 85
    assert recommendations[1].product_name == "Bread"
    assert recommendations[1].predicted_demand == 60
    assert recommendations[1].required_stock == 65


@pytest.mark.parametrize(
    ("demand", "current", "safety"),
    [
        ([float("nan")], [0], [0]),
        ([float("inf")], [0], [0]),
        ([0], [-1], [0]),
        ([0], [0], [-1]),
        (["invalid"], [0], [0]),
    ],
)
def test_generate_recommendations_rejects_invalid_inventory_values(demand, current, safety):
    with pytest.raises((TypeError, ValueError)):
        generate_recommendations(["Item"], demand, current, safety)


def test_generate_recommendations_uses_tolerance_for_adequate_status():
    from module4_inventory_optimization.config import NUMERICAL_TOLERANCE

    recommendations = generate_recommendations(
        ["Item"], [10], [10 + NUMERICAL_TOLERANCE / 2], [0]
    )
    assert recommendations[0].stock_status == "ADEQUATE"
    assert recommendations[0].recommended_order_quantity == 0


def test_generate_recommendations_rejects_product_string():
    with pytest.raises(TypeError, match="sequence of product names"):
        generate_recommendations("Milk", [10], [5], [0])
