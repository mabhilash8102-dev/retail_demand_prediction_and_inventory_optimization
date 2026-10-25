from pathlib import Path

import pytest

from module1_data_matrix.data_loader import load_sales_data
from module1_data_matrix.matrix_builder import construct_matrices
from module1_data_matrix.matrix_operations import (
    matrix_multiply,
    matrix_rank,
    rank,
    reduced_row_echelon_form,
    rref,
    transpose,
    transpose_matrix,
)


def test_construct_matrices_includes_day_feature_and_intercept():
    records = [
        {"day": "Monday", "product": "Rice", "price": 55, "promotion": 0, "sales": 120},
        {"day": "Tuesday", "product": "Rice", "price": 53, "promotion": 1, "sales": 145},
    ]
    A, b = construct_matrices(records)
    assert A == [[1.0, 1.0, 55.0, 0.0], [1.0, 2.0, 53.0, 1.0]]
    assert b == [[120.0], [145.0]]


def test_empty_records_rejected():
    with pytest.raises(ValueError):
        construct_matrices([])


def test_transpose_matrix():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose_matrix([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


def test_matrix_multiplication():
    assert matrix_multiply([[1, 2]], [[3], [4]]) == [[11.0]]


def test_demo_matrix_transpose_and_ata_results():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_sales.csv"
    records = load_sales_data(data_path)
    matrix, _ = construct_matrices(records)

    transposed_matrix = transpose(matrix)
    ata = matrix_multiply(transposed_matrix, matrix)

    assert (len(matrix), len(matrix[0])) == (10, 4)
    assert (len(transposed_matrix), len(transposed_matrix[0])) == (4, 10)
    assert (len(ata), len(ata[0])) == (4, 4)
    assert ata == [
        [10.0, 34.0, 746.0, 5.0],
        [34.0, 154.0, 2432.0, 16.0],
        [746.0, 2432.0, 75046.0, 414.0],
        [5.0, 16.0, 414.0, 5.0],
    ]


def test_matrix_multiplication_rejects_incompatible_dimensions():
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2]], [[1, 2]])


def test_rank_for_dependent_and_independent_matrices():
    assert rank([[1, 2], [2, 4]]) == 1
    assert rank([[1, 0], [0, 1]]) == 2
    assert matrix_rank([[1, 2], [2, 4]]) == 1


def test_rref_for_rank_deficient_and_identity_cases():
    assert rref([[1, 2], [2, 4]]) == [[1.0, 2.0], [0.0, 0.0]]
    assert reduced_row_echelon_form([[1, 0], [0, 1]]) == [[1.0, 0.0], [0.0, 1.0]]


def test_rref_handles_rectangular_matrix():
    result = reduced_row_echelon_form([[1, 2, 3], [0, 0, 0]])
    assert result[0] == [1.0, 2.0, 3.0]
    assert result[1] == [0.0, 0.0, 0.0]


def test_rref_handles_zero_rows():
    assert reduced_row_echelon_form([[0, 0], [0, 0]]) == [[0.0, 0.0], [0.0, 0.0]]
