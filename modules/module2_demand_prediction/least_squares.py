"""Least-squares fitting through the normal equations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .config import DEFAULT_TOLERANCE
from .models import FloatArray, LeastSquaresResult


class SingularMatrixError(ValueError):
    """Raised when the normal-equation system is singular or nearly singular."""


def _as_finite_array(value: ArrayLike, name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a rectangular array of real numeric values."
        ) from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_inputs(
    A: ArrayLike, b: ArrayLike, tolerance: float
) -> tuple[FloatArray, FloatArray]:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite, non-negative number.")

    matrix = _as_finite_array(A, "A")
    target = _as_finite_array(b, "b")

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("A must be a non-empty two-dimensional matrix.")
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    elif target.ndim != 2 or target.shape[1] != 1:
        raise ValueError("b must be a vector or a two-dimensional column vector.")
    if target.shape[0] == 0:
        raise ValueError("b cannot be empty.")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError(
            f"A has {matrix.shape[0]} rows but b has {target.shape[0]} values."
        )
    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError("A must have at least as many observations as features.")
    return matrix, target


def fit_least_squares(
    A: ArrayLike, b: ArrayLike, tolerance: float = DEFAULT_TOLERANCE
) -> LeastSquaresResult:
    """Solve ``(A.T @ A) X = A.T @ b`` and return all key intermediates.

    The normal-equation system is solved directly; an inverse is never
    explicitly formed. Rank and singular values are checked first so
    singular or nearly singular inputs produce a meaningful exception.
    """
    matrix, target = _validate_inputs(A, b, tolerance)
    transpose = matrix.T.copy()
    ata = transpose @ matrix
    atb = transpose @ target

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0])
    rank_threshold = tolerance * largest
    matrix_rank = int(np.count_nonzero(singular_values > rank_threshold))
    if matrix_rank < matrix.shape[1]:
        raise SingularMatrixError(
            "A (and therefore A.T @ A) is singular or nearly singular "
            f"at tolerance {tolerance:g}; rank is {matrix_rank} "
            f"but {matrix.shape[1]} columns are required."
        )

    try:
        coefficients = np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError as error:
        raise SingularMatrixError(
            "The normal-equation matrix A.T @ A could not be solved."
        ) from error

    return LeastSquaresResult(
        coefficients=coefficients,
        A_transpose=transpose,
        AtA=ata,
        Atb=atb,
        rank=matrix_rank,
    )


__all__ = ["SingularMatrixError", "fit_least_squares"]
