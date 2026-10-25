from __future__ import annotations

import math
from typing import Any

from .config import DAY_MAP, REQUIRED_COLUMNS


def validate_row(row: dict[str, Any], line_number: int | None = None) -> dict[str, Any]:
    if row is None or not isinstance(row, dict):
        raise ValueError("Each record must be a dictionary-like row.")

    normalized = {
        str(key).strip().lower(): value for key, value in row.items() if key is not None
    }
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in normalized or normalized[column] is None or str(normalized[column]).strip() == ""
    ]
    if missing:
        raise ValueError(f"Missing required field(s): {sorted(missing)}")

    day = str(normalized["day"]).strip()
    product = str(normalized["product"]).strip()
    if not day or not product:
        raise ValueError("Day and product cannot be blank.")

    day_key = day.lower()
    if day_key not in DAY_MAP:
        raise ValueError(f"Invalid day value '{day}'. Expected a valid weekday name.")

    try:
        price = float(normalized["price"])
        sales = float(normalized["sales"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Price and sales must be numeric values.") from exc

    if not math.isfinite(price) or not math.isfinite(sales):
        raise ValueError("Price and sales cannot be NaN or infinity.")

    if price <= 0:
        raise ValueError("Price must be greater than zero.")
    if sales < 0:
        raise ValueError("Sales cannot be negative.")

    try:
        promotion = float(normalized["promotion"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Promotion must be numeric.") from exc

    if not math.isfinite(promotion):
        raise ValueError("Promotion cannot be NaN or infinity.")
    if promotion not in (0.0, 1.0):
        raise ValueError("Promotion must be either 0 or 1.")

    return {
        "day": day,
        "product": product,
        "price": price,
        "promotion": int(promotion),
        "sales": sales,
    }


def validate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if records is None:
        raise ValueError("Records cannot be None.")
    if not records:
        raise ValueError("CSV contains no sales records.")

    validated: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        validated.append(validate_row(row, index))
    return validated


__all__ = ["validate_row", "validate_records"]
