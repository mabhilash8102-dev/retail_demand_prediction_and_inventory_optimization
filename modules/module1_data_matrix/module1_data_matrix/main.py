
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1_data_matrix.data_loader import load_sales_data
from module1_data_matrix.matrix_builder import construct_matrices
from module1_data_matrix.matrix_operations import matrix_multiply, matrix_rank, reduced_row_echelon_form, transpose


def main():
    data_file = Path(__file__).resolve().parent / "data" / "sample_sales.csv"
    records = load_sales_data(data_file)
    A, b = construct_matrices(records)

    print("MEMBER 1: HISTORICAL SALES DATA AND MATRIX PROCESSING")
    print("=" * 60)
    print(f"Records loaded: {len(records)}")
    print(f"A shape: ({len(A)}, {len(A[0])})")
    print(f"b shape: ({len(b)}, {len(b[0])})")
    print("Feature order: intercept, day, price, promotion")
    print("Sample A rows:")
    for row in A[:3]:
        print(row)
    print("Sample b rows:")
    for row in b[:3]:
        print(row)
    print("Transpose of A:")
    for row in transpose(A)[:3]:
        print(row)
    print("A^T A:")
    for row in matrix_multiply(transpose(A), A)[:3]:
        print(row)
    print("Rank(A):", matrix_rank(A))
    print("RREF example:")
    for row in reduced_row_echelon_form([[1, 2], [2, 4]]):
        print(row)


if __name__ == "__main__":
    main()