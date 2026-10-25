import unittest

import numpy as np

from module2_demand_prediction import fit_least_squares, predict
from module2_demand_prediction.data.sample_prediction_input import FEATURE_ORDER, sample_prediction_input
from module3_pattern_analysis import fit_demand_model, predict_future_demand, predict_historical_demand


class Module2IntegrationTests(unittest.TestCase):
    def test_bridge_matches_module2_api(self):
        A, b = sample_prediction_input()

        direct = fit_least_squares(A, b)
        bridge = fit_demand_model(A, b)

        self.assertTrue(np.allclose(direct.coefficients, bridge.coefficients))
        self.assertEqual(direct.rank, bridge.rank)

        historical = predict_historical_demand(A, bridge.coefficients)
        self.assertEqual(historical.shape, b.shape)
        self.assertTrue(np.allclose(historical, A @ bridge.coefficients))

        features = {"day": 6.0, "price": 30.0, "promotion": 0.0}
        bridge_prediction = predict_future_demand(bridge.coefficients, features, FEATURE_ORDER)
        direct_prediction = predict(bridge.coefficients, features, FEATURE_ORDER)
        self.assertAlmostEqual(bridge_prediction, direct_prediction)

    def test_missing_feature_order_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_future_demand(np.array([1.0, 2.0]), {"day": 3.0}, None)
