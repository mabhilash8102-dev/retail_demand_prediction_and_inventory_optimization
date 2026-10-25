REQUIRED_COLUMNS = ["day", "product", "price", "promotion", "sales"]
DEFAULT_FEATURES = ["day", "price", "promotion"]
TARGET_COLUMN = "sales"
TOLERANCE = 1e-10

DAY_MAP = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}

__all__ = [
    "REQUIRED_COLUMNS",
    "DEFAULT_FEATURES",
    "TARGET_COLUMN",
    "TOLERANCE",
    "DAY_MAP",
]
