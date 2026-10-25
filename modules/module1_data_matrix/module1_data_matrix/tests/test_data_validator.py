import pytest

from module1_data_matrix.data_validator import validate_records, validate_row


def test_validate_row_accepts_valid_data():
    row = {"day": "Monday", "product": "Rice", "price": "55", "promotion": "0", "sales": "120"}
    validated = validate_row(row)
    assert validated == {"day": "Monday", "product": "Rice", "price": 55.0, "promotion": 0, "sales": 120.0}


def test_validate_row_rejects_invalid_promotion():
    with pytest.raises(ValueError, match="Promotion"):
        validate_row({"day": "Monday", "product": "Rice", "price": 55, "promotion": 2, "sales": 120})


def test_validate_row_rejects_nan_and_infinity():
    with pytest.raises(ValueError, match="NaN|infinity"):
        validate_row({"day": "Monday", "product": "Rice", "price": "nan", "promotion": 0, "sales": 120})


def test_validate_records_rejects_empty_list():
    with pytest.raises(ValueError, match="no sales records"):
        validate_records([])
