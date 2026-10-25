import tempfile
from pathlib import Path

import pytest

from module1_data_matrix.data_loader import load_sales_data


def create_csv(content):
    temp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", encoding="utf-8", newline="", delete=False)
    temp.write(content)
    temp.close()
    path = Path(temp.name)
    return path


def test_valid_csv_10_rows():
    csv_text = (
        "day,product,price,promotion,sales\n"
        "Monday,Rice,55,0,120\n"
        "Tuesday,Rice,53,1,145\n"
        "Wednesday,Rice,55,0,110\n"
        "Thursday,Oil,140,1,90\n"
        "Friday,Oil,145,0,75\n"
        "Saturday,Soap,35,1,160\n"
        "Sunday,Soap,35,0,130\n"
        "Monday,Oil,138,1,95\n"
        "Tuesday,Soap,42,0,170\n"
        "Wednesday,Milk,48,1,180\n"
    )
    path = create_csv(csv_text)
    try:
        records = load_sales_data(path)
        assert len(records) == 10
        assert records[0]["day"] == "Monday"
        assert records[0]["price"] == 55.0
        assert records[0]["sales"] == 120.0
    finally:
        path.unlink(missing_ok=True)


def test_missing_file():
    path = Path(tempfile.gettempdir()) / "missing_sales_test_987654.csv"
    with pytest.raises(FileNotFoundError):
        load_sales_data(path)


def test_empty_file():
    path = create_csv("")
    try:
        with pytest.raises(ValueError):
            load_sales_data(path)
    finally:
        path.unlink(missing_ok=True)


def test_missing_column():
    path = create_csv("day,product,price,promotion\nMonday,Rice,55,0\n")
    try:
        with pytest.raises(ValueError):
            load_sales_data(path)
    finally:
        path.unlink(missing_ok=True)


def test_invalid_numeric_values_are_rejected():
    path = create_csv("day,product,price,promotion,sales\nMonday,Rice,abc,0,120\n")
    try:
        with pytest.raises(ValueError):
            load_sales_data(path)
    finally:
        path.unlink(missing_ok=True)


def test_invalid_promotion_is_rejected():
    path = create_csv("day,product,price,promotion,sales\nMonday,Rice,55,2,120\n")
    try:
        with pytest.raises(ValueError):
            load_sales_data(path)
    finally:
        path.unlink(missing_ok=True)


def test_invalid_values_like_nan_or_infinity_are_rejected():
    path = create_csv("day,product,price,promotion,sales\nMonday,Rice,nan,0,120\n")
    try:
        with pytest.raises(ValueError):
            load_sales_data(path)
    finally:
        path.unlink(missing_ok=True)
