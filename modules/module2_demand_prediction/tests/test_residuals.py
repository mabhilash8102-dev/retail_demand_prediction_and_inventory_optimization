import numpy as np
import pytest

from module2_demand_prediction import (
    calculate_residuals,
    check_residual_orthogonality,
    fit_least_squares,
    predict_from_matrix,
)


def test_residuals_and_perfect_fit():
    A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
    actual = np.array([[3], [5], [7], [9]], dtype=float)
    result = fit_least_squares(A, actual)
    residuals = calculate_residuals(actual, predict_from_matrix(A, result.coefficients))

    np.testing.assert_allclose(residuals, np.zeros((4, 1)), atol=1e-10)
    assert check_residual_orthogonality(A, residuals)


def test_residual_orthogonality_on_noisy_overdetermined_fit():
    A = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
    actual = np.array([[1], [3], [4], [6]], dtype=float)
    result = fit_least_squares(A, actual)
    residuals = calculate_residuals(
        actual, predict_from_matrix(A, result.coefficients)
    )

    assert check_residual_orthogonality(A, residuals, tolerance=1e-10)
    np.testing.assert_allclose(A.T @ residuals, [[0], [0]], atol=1e-10)


def test_residuals_validate_shape_and_tolerance():
    with pytest.raises(ValueError, match="same number"):
        calculate_residuals([1, 2], [1])
    with pytest.raises(ValueError, match="non-negative"):
        check_residual_orthogonality([[1], [2]], [0, 0], tolerance=-1)
