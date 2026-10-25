from module1_data_matrix.matrix_builder import build_matrix


def test_default_matrix_shape_and_order():
    records = [
        {"day": "Monday", "product": "Rice", "price": 55, "promotion": 0, "sales": 120},
        {"day": "Tuesday", "product": "Rice", "price": 53, "promotion": 1, "sales": 145},
    ]
    matrix_data = build_matrix(records)
    assert matrix_data.features == ["intercept", "day", "price", "promotion"]
    assert len(matrix_data.matrix) == 2
    assert len(matrix_data.matrix[0]) == 4
    assert matrix_data.matrix[0] == [1.0, 1.0, 55.0, 0.0]
    assert matrix_data.target_vector == [[120.0], [145.0]]


def test_custom_features_are_supported():
    records = [{"day": "Monday", "product": "Rice", "price": 55, "promotion": 0, "sales": 120}]
    matrix_data = build_matrix(records, features=["price", "promotion"])
    assert matrix_data.features == ["intercept", "price", "promotion"]
    assert matrix_data.matrix == [[1.0, 55.0, 0.0]]
