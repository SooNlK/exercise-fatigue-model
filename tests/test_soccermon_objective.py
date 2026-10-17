import unittest

import numpy as np
import pandas as pd

from src.data.soccermon.objective import GPS_FEATURES, summarize
from src.experiments.soccermon.objective_pilot import common_cohort, pilot_splits


class ObjectivePilotTests(unittest.TestCase):
    def test_invalid_signals_do_not_become_zero_load(self):
        empty = summarize([np.nan, -1], [np.inf, -1], 2)
        self.assertTrue(np.isnan(empty["gps_speed_mean"]))
        self.assertTrue(np.isnan(empty["gps_impulse_mean"]))
        valid = summarize([0, 2, np.nan, -1], [0, 4], 4)
        self.assertEqual(valid["gps_speed_mean"], 1)
        self.assertEqual(valid["gps_impulse_mean"], 2)
        self.assertEqual(valid["valid_speed_samples"], 2)

    def test_cohort_excludes_absent_or_invalid_gps_without_imputing_it(self):
        dates = pd.date_range("2020-06-01", periods=3)
        subjective = pd.DataFrame({"participant": "a", "date": dates, "target": [3, 2, 4]})
        gps = pd.DataFrame({"participant": "a", "date": dates[:2]})
        for col in GPS_FEATURES:
            gps[col] = 1.
        gps.loc[1, "gps_speed_mean"] = np.nan
        cohort = common_cohort(subjective, gps)
        self.assertEqual(len(cohort), 1)
        self.assertEqual(cohort.date.iloc[0], dates[0])
        self.assertEqual(cohort.target.iloc[0], 3)

    def test_pilot_splits_separate_people_or_future_targets(self):
        dates = pd.date_range("2020-06-01", "2020-12-31")
        data = pd.concat([pd.DataFrame({"participant": str(i), "date": dates}) for i in range(3)], ignore_index=True)
        data["target_date"] = data.date + pd.Timedelta(days=1)
        chronological_tests = []
        for name, _, train, test in pilot_splits(data):
            tr, te = data.iloc[train], data.iloc[test]
            if name == "leave_one_player_out":
                self.assertFalse(set(tr.participant) & set(te.participant))
            else:
                self.assertLess(tr.target_date.max(), te.date.min())
                chronological_tests.append(set(test))
        self.assertEqual(len(chronological_tests), 2)
        self.assertFalse(chronological_tests[0] & chronological_tests[1])


if __name__ == "__main__":
    unittest.main()
