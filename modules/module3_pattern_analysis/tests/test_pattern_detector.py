import unittest

from module3_pattern_analysis import analyze_eigenvalues, detect_patterns


class PatternDetectorTests(unittest.TestCase):
    def test_dominant_contributions(self):
        result = analyze_eigenvalues([[2, 0], [0, 3]])
        pattern = detect_patterns(result, ["Milk", "Bread"])
        self.assertEqual(pattern.dominant_eigenvalue, 3)
        self.assertEqual(set(pattern.product_contributions), {"Milk", "Bread"})
        self.assertAlmostEqual(sum(x * x for x in pattern.dominant_eigenvector), 1)

    def test_reject_wrong_product_name_count(self):
        result = analyze_eigenvalues([[2, 0], [0, 3]])
        with self.assertRaises(ValueError): detect_patterns(result, ["Milk"])
