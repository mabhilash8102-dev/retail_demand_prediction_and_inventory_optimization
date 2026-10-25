
from __future__ import annotations

from pathlib import Path

from .config import REQUIRED_COLUMNS
from .data_validator import validate_records, validate_row


def load_sales_data(file_path) -> list[dict[str, float | int | str]]:
    """Load and validate a sales CSV file without silently discarding invalid rows."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Sales data file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {path}")

    import csv

    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)

        if not reader.fieldnames:
            raise ValueError(f"No CSV headers found in: {path}")

        normalized_fieldnames = [str(name).strip().lower() for name in reader.fieldnames]
        reader.fieldnames = normalized_fieldnames

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized_fieldnames]
        if missing_columns:
            raise ValueError(f"Missing columns: {sorted(missing_columns)}. Found: {normalized_fieldnames}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            if row is None:
                continue
            if all((value is None or str(value).strip() == "") for value in row.values()):
                raise ValueError(f"Blank row encountered on line {line_number}.")
            rows.append(validate_row(row, line_number))

    return validate_records(rows)


__all__ = ["load_sales_data"]
