from __future__ import annotations

from .config import DAY_MAP, DEFAULT_FEATURES, TARGET_COLUMN
from .models import MatrixData


def _normalize_day_value(day: str | int) -> float:
    if isinstance(day, (int, float)):
        numeric_day = int(day)
        if 1 <= numeric_day <= 7:
            return float(numeric_day)
        raise ValueError("Day values must be in the range 1 to 7.")

    text = str(day).strip().lower()
    if text not in DAY_MAP:
        raise ValueError(f"Invalid day value '{day}'.")
    return float(DAY_MAP[text])


def build_matrix(records, features=None, target_column: str = TARGET_COLUMN) -> MatrixData:
    if records is None or not records:
        raise ValueError("Records cannot be empty.")

    selected_features = list(DEFAULT_FEATURES if features is None else features)
    selected_features = [str(feature).strip().lower() for feature in selected_features]
    if not selected_features:
        raise ValueError("At least one feature must be supplied.")

    matrix: list[list[float]] = []
    target_vector: list[list[float]] = []
    for record in records:
        row: list[float] = [1.0]
        for feature in selected_features:
            if feature == "day":
                row.append(_normalize_day_value(record["day"]))
            elif feature == "price":
                row.append(float(record["price"]))
            elif feature == "promotion":
                row.append(float(record["promotion"]))
            elif feature == "intercept":
                continue
            elif feature in record:
                row.append(float(record[feature]))
            else:
                raise ValueError(f"Feature '{feature}' is not available in the dataset.")
        row = [float(value) for value in row]
        matrix.append(row)
        if target_column not in record:
            raise ValueError(f"Target column '{target_column}' is missing.")
        target_vector.append([float(record[target_column])])

    return MatrixData(
        matrix=matrix,
        target_vector=target_vector,
        features=["intercept", *selected_features],
        target_column=target_column,
    )


def construct_matrices(records, features=None, target_column: str = TARGET_COLUMN):
    matrix_data = build_matrix(records, features=features, target_column=target_column)
    return matrix_data.matrix, matrix_data.target_vector


__all__ = ["build_matrix", "construct_matrices"]
