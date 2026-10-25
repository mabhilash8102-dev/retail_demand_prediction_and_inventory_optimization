import unittest

from module3_pattern_analysis import analyze_eigenvalues, verify_eigenpairs


class EigenAnalysisTests(unittest.TestCase):
    def test_known_matrix_order_equation_normalization_orthogonality(self):
        matrix = [[2, 1], [1, 2]]
        result = analyze_eigenvalues(matrix)
        self.assertAlmostEqual(result.eigenvalues[0], 3)
        self.assertAlmostEqual(result.eigenvalues[1], 1)
        self.assertTrue(verify_eigenpairs(matrix, result))
        for vector in result.eigenvectors:
            self.assertAlmostEqual(sum(x * x for x in vector), 1)
        dot = sum(a * b for a, b in zip(*result.eigenvectors))
        self.assertAlmostEqual(dot, 0)
        self.assertAlmostEqual(sum(result.explained_variance), 1)
        self.assertAlmostEqual(result.cumulative_explained_variance[-1], 1)
        self.assertAlmostEqual(result.total_variance, 4)
        self.assertAlmostEqual(result.explained_variance_percentages[0], 75.0)
        self.assertAlmostEqual(result.cumulative_explained_variance_percentages[-1], 100.0)

    def test_zero_variance_and_tiny_negative_noise_are_handled(self):
        result = analyze_eigenvalues([[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(result.eigenvalues, [0.0, 0.0])
        self.assertEqual(result.explained_variance, [0.0, 0.0])
        self.assertEqual(result.cumulative_explained_variance, [0.0, 0.0])
        self.assertEqual(result.explained_variance_percentages, [0.0, 0.0])

    def test_reject_materially_invalid_negative_eigenvalue(self):
        with self.assertRaises(ValueError):
            analyze_eigenvalues([[1.0, 0.0], [0.0, -0.01]])

    def test_reject_nonsymmetric_matrix(self):
        with self.assertRaises(ValueError): analyze_eigenvalues([[1, 2], [0, 1]])
