import numpy as np
import pytest

from module2_demand_prediction import SingularMatrixError, fit_least_squares


def test_exact_linear_relationship_and_intermediates():
    A = [[1, 1], [1, 2], [1, 3], [1, 4]]
    b = [[3], [5], [7], [9]]

    result = fit_least_squares(A, b)

    np.testing.assert_allclose(result.coefficients, [[1], [2]], atol=1e-10)
    np.testing.assert_allclose(result.A_transpose, [[1, 1, 1, 1], [1, 2, 3, 4]])
    np.testing.assert_allclose(result.AtA, [[4, 10], [10, 30]])
    np.testing.assert_allclose(result.Atb, [[24], [70]])
    assert result.rank == 2


def test_general_overdetermined_solution():
    A = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 3, 4, 6], dtype=float)

    result = fit_least_squares(A, b)

    np.testing.assert_allclose(result.coefficients[:, 0], [1.1, 1.6], atol=1e-10)
    assert result.coefficients.shape == (2, 1)


@pytest.mark.parametrize(
    ("A", "b", "message"),
    [
        ([], [], "non-empty"),
        ([[1, 2], [3]], [1, 2], "rectangular"),
        ([[1], [2]], [[1, 2], [3, 4]], "column vector"),
        ([[1], [2]], [1], "rows"),
        ([[1, 2]], [1], "at least as many"),
        ([[1], [2]], [1, float("nan")], "finite"),
    ],
)
def test_rejects_invalid_input(A, b, message):
    with pytest.raises(ValueError, match=message):
        fit_least_squares(A, b)


@pytest.mark.parametrize(
    "A",
    [
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1 + 1e-12], [1, 1 + 2e-12]],
    ],
)
def test_singular_or_nearly_singular_design_raises_meaningful_error(A):
    with pytest.raises(SingularMatrixError, match="singular or nearly singular"):
        fit_least_squares(A, [1, 2, 3])


def test_configurable_tolerance_can_accept_well_conditioned_small_scale_data():
    A = [[1, 1], [1, 2], [1, 3]]
    result = fit_least_squares(A, [3, 5, 7], tolerance=1e-12)
    np.testing.assert_allclose(result.coefficients, [[1], [2]], atol=1e-10)
