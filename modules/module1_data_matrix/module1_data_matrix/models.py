from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatrixData:
    matrix: list[list[float]]
    target_vector: list[list[float]]
    features: list[str]
    target_column: str
    observation_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.observation_count = len(self.matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "target_column": self.target_column,
            "matrix": self.matrix,
            "target_vector": self.target_vector,
            "observation_count": self.observation_count,
        }


__all__ = ["MatrixData"]
