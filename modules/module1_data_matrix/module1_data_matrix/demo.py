from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1_data_matrix.data_loader import load_sales_data
from module1_data_matrix.matrix_builder import construct_matrices
from module1_data_matrix.matrix_operations import matrix_multiply, matrix_rank, reduced_row_echelon_form, transpose


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "data" / "sample_sales.csv"
    records = load_sales_data(csv_path)
    A, b = construct_matrices(records)

    print("MEMBER 1: HISTORICAL SALES DATA AND MATRIX PROCESSING")
    print("=" * 60)
    print(f"Rows loaded: {len(records)}")
    print(f"A shape: {len(A)} x {len(A[0])}")
    print(f"b shape: {len(b)} x {len(b[0])}")
    print("First row of A:", A[0])
    print("First row of b:", b[0])
    print("Rank(A):", matrix_rank(A))
    print("RREF example:")
    for row in reduced_row_echelon_form([[1, 2], [2, 4]]):
        print(row)

    transposed_A = transpose(A)
    print("Transpose of A:")
    for row in transposed_A:
        print(row)
    ata = matrix_multiply(transposed_A, A)
    print("A^T * A:")
    for row in ata:
        print(row)


if __name__ == "__main__":
    main()
