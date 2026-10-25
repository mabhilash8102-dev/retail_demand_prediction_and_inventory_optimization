import unittest

from module3_pattern_analysis import calculate_covariance, is_symmetric


class CovarianceTests(unittest.TestCase):
    def test_sample_covariance_and_symmetry(self):
        result = calculate_covariance([[-1, -2], [1, 2]]).covariance_matrix
        self.assertEqual(result, [[2, 4], [4, 8]])
        self.assertTrue(is_symmetric(result))

    def test_reject_bad_dimensions(self):
        for matrix in ([], [[1]], [[1, 2], [3]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                calculate_covariance(matrix)

    def test_reject_non_finite_values(self):
        with self.assertRaises(ValueError):
            calculate_covariance([[1.0, float("nan")], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            calculate_covariance([[1.0, float("inf")], [2.0, 3.0]])

    def test_symmetry_rejects_nonsquare_or_nonsymmetric(self):
        self.assertFalse(is_symmetric([[1, 2]]))
        self.assertFalse(is_symmetric([[1, 2], [3, 4]]))
