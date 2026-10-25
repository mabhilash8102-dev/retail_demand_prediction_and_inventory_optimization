from __future__ import annotations

from .config import TOLERANCE


def transpose(matrix):
    """Return the transpose of a matrix."""
    if matrix is None:
        raise ValueError("Matrix cannot be None.")
    if not matrix:
        return []
    column_count = len(matrix[0])
    if any(len(row) != column_count for row in matrix):
        raise ValueError("Matrix rows must have equal lengths.")
    return [[matrix[row_index][column_index] for row_index in range(len(matrix))] for column_index in range(column_count)]


def matrix_multiply(matrix_a, matrix_b):
    """Multiply two matrices using standard row-by-column multiplication."""
    if matrix_a is None or matrix_b is None:
        raise ValueError("Matrices cannot be None.")
    if not matrix_a or not matrix_b:
        raise ValueError("Matrices cannot be empty.")

    if any(len(row) != len(matrix_a[0]) for row in matrix_a):
        raise ValueError("matrix_a must be rectangular.")
    if any(len(row) != len(matrix_b[0]) for row in matrix_b):
        raise ValueError("matrix_b must be rectangular.")
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Incompatible matrix dimensions for multiplication.")

    result = []
    for row in matrix_a:
        new_row = []
        for col_index in range(len(matrix_b[0])):
            total = 0.0
            for inner_index in range(len(row)):
                total += row[inner_index] * matrix_b[inner_index][col_index]
            new_row.append(total)
        result.append(new_row)
    return result


def rref(matrix, tolerance: float = TOLERANCE):
    """Compute the reduced row echelon form via manual Gaussian elimination."""
    if matrix is None:
        raise ValueError("Matrix cannot be None.")
    if not matrix:
        return []

    matrix_copy = [[float(value) for value in row] for row in matrix]
    if any(len(row) != len(matrix_copy[0]) for row in matrix_copy):
        raise ValueError("Matrix rows must have equal lengths.")

    row = 0
    rows = len(matrix_copy)
    cols = len(matrix_copy[0])

    for col in range(cols):
        if row >= rows:
            break

        pivot_row = max(range(row, rows), key=lambda r: abs(matrix_copy[r][col]), default=row)
        pivot_value = matrix_copy[pivot_row][col]
        if abs(pivot_value) <= tolerance:
            continue

        matrix_copy[row], matrix_copy[pivot_row] = matrix_copy[pivot_row], matrix_copy[row]
        pivot_value = matrix_copy[row][col]

        for c in range(col, cols):
            matrix_copy[row][c] /= pivot_value

        for r in range(rows):
            if r == row:
                continue
            factor = matrix_copy[r][col]
            if abs(factor) <= tolerance:
                continue
            for c in range(col, cols):
                matrix_copy[r][c] -= factor * matrix_copy[row][c]

        row += 1

    return [[0.0 if abs(value) <= tolerance else value for value in row] for row in matrix_copy]


def rank(matrix, tolerance: float = TOLERANCE):
    """Compute matrix rank manually without NumPy's matrix_rank."""
    if matrix is None:
        raise ValueError("Matrix cannot be None.")
    if not matrix:
        return 0

    matrix_copy = [[float(value) for value in row] for row in matrix]
    if any(len(row) != len(matrix_copy[0]) for row in matrix_copy):
        raise ValueError("Matrix rows must have equal lengths.")

    rows = len(matrix_copy)
    cols = len(matrix_copy[0])
    pivot_row = 0
    rank_value = 0

    for col in range(cols):
        if pivot_row >= rows:
            break
        pivot_index = max(range(pivot_row, rows), key=lambda r: abs(matrix_copy[r][col]), default=pivot_row)
        if abs(matrix_copy[pivot_index][col]) <= tolerance:
            continue
        matrix_copy[pivot_row], matrix_copy[pivot_index] = matrix_copy[pivot_index], matrix_copy[pivot_row]
        pivot = matrix_copy[pivot_row][col]
        for c in range(col, cols):
            matrix_copy[pivot_row][c] /= pivot
        for r in range(rows):
            if r == pivot_row:
                continue
            factor = matrix_copy[r][col]
            if abs(factor) <= tolerance:
                continue
            for c in range(col, cols):
                matrix_copy[r][c] -= factor * matrix_copy[pivot_row][c]
        rank_value += 1
        pivot_row += 1

    return rank_value


def transpose_matrix(matrix):
    return transpose(matrix)


def matrix_rank(matrix, tolerance: float = TOLERANCE):
    return rank(matrix, tolerance=tolerance)


def reduced_row_echelon_form(matrix, tolerance: float = TOLERANCE):
    return rref(matrix, tolerance=tolerance)


def construct_matrices(records, features=None, target_column: str = "sales"):
    from .matrix_builder import construct_matrices as builder_construct_matrices
    return builder_construct_matrices(records, features=features, target_column=target_column)


__all__ = [
    "transpose",
    "transpose_matrix",
    "matrix_multiply",
    "rref",
    "reduced_row_echelon_form",
    "rank",
    "matrix_rank",
    "construct_matrices",
]
