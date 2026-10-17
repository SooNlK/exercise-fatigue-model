import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from src.experiments.pmdata.recalculate import held_out_predictions, prediction_metrics


class PMDataEvaluationTests(unittest.TestCase):
    def test_whole_person_is_held_out_and_every_row_has_a_prediction(self):
        # Row-wise CV would leave the person's other row in training and change these means.
        X = pd.DataFrame({'x': range(6)})
        groups = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
        y = pd.Series([1., 1., 3., 3., 9., 9.])
        pred, choices = held_out_predictions(DummyRegressor(), X, y, groups)
        np.testing.assert_allclose(pred, [6., 6., 5., 5., 2., 2.])
        self.assertEqual(choices, [])

    def test_delta_reconstruction_preserves_errors_but_changes_r2_denominator(self):
        today = np.array([10., -5., 20., 0.])
        delta_true = np.array([1., -1., 2., -2.])
        delta_pred = np.zeros(4)
        delta = prediction_metrics(delta_true, delta_pred, today, True)
        level = prediction_metrics(delta_true + today, delta_pred + today, today, False)
        self.assertAlmostEqual(delta['mae'], level['mae'])
        self.assertAlmostEqual(delta['rmse'], level['rmse'])
        self.assertAlmostEqual(delta['next_day_r2'], level['next_day_r2'])
        self.assertNotAlmostEqual(delta['target_r2'], delta['next_day_r2'])
        self.assertAlmostEqual(delta['mae'], mean_absolute_error(delta_true + today, today))


if __name__ == '__main__':
    unittest.main()
