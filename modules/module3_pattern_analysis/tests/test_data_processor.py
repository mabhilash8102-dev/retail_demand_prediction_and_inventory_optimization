import unittest

from module3_pattern_analysis import prepare_sales_matrix


class DataProcessorTests(unittest.TestCase):
    def test_centering_means_and_names(self):
        result = prepare_sales_matrix({"a": [1, 3], "b": [4, 8]}, ["a", "b"])
        self.assertEqual(result.means, [2, 6])
        self.assertEqual(result.column_names, ["a", "b"])
        for column in range(2):
            self.assertAlmostEqual(sum(row[column] for row in result.centered_matrix), 0)

    def test_reject_missing_and_nonnumeric_values(self):
        for values in ([1, None], [1, "bad"], [1, float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                prepare_sales_matrix({"a": values}, ["a"])

    def test_reject_missing_column_empty_selection_and_too_few_rows(self):
        with self.assertRaises(ValueError): prepare_sales_matrix({"a": [1, 2]}, ["b"])
        with self.assertRaises(ValueError): prepare_sales_matrix({"a": [1, 2]}, [])
        with self.assertRaises(ValueError): prepare_sales_matrix({"a": [1]}, ["a"])

    def test_reject_unequal_column_lengths(self):
        with self.assertRaises(ValueError): prepare_sales_matrix({"a": [1, 2], "b": [3]}, ["a", "b"])
