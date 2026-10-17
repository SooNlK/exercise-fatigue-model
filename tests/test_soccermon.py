import unittest

import numpy as np
import pandas as pd

from src.data.soccermon.preparation import WELLNESS, make_features
from src.experiments.soccermon.compare import make_splits
from src.experiments.soccermon.stability import event_pairs, stability_splits


class SoccerMonTests(unittest.TestCase):
    def daily(self):
        dates = pd.date_range("2021-01-01", periods=45)
        data = pd.DataFrame({"participant": "a", "date": dates, "daily_srpe": np.arange(45) * 10.0})
        for col in WELLNESS:
            data[col] = 1 + np.arange(45) % 5
        return data

    def test_future_changes_do_not_change_features(self):
        data = self.daily()
        first, sets = make_features(data)
        changed = data.copy()
        changed.loc[changed.date > "2021-01-25", WELLNESS + ["daily_srpe"]] = 100
        second, _ = make_features(changed)
        columns = sets["wellness_load"] + ["intense_threshold", "intense_day"]
        pd.testing.assert_frame_equal(first.loc[:24, columns], second.loc[:24, columns])

    def test_missing_target_stays_missing(self):
        data = self.daily()
        data.loc[10, "fatigue"] = np.nan
        result, _ = make_features(data)
        self.assertTrue(pd.isna(result.loc[9, "target"]))
        self.assertTrue(pd.isna(result.loc[11, "fatigue_lag1"]))

    def test_missing_calendar_day_and_duplicate_are_rejected(self):
        data = self.daily()
        with self.assertRaises(ValueError):
            make_features(data.drop(index=10))
        with self.assertRaises(ValueError):
            make_features(pd.concat([data, data.iloc[[10]]]))

    def test_validation_boundaries(self):
        data = pd.concat([self.daily().assign(participant=str(i)) for i in range(10)], ignore_index=True)
        data["target_date"] = data.date + pd.Timedelta(days=1)
        for name, _, train, test in make_splits(data, "2021-02-01"):
            tr, te = data.iloc[train], data.iloc[test]
            if name == "group5":
                self.assertFalse(set(tr.participant) & set(te.participant))
            else:
                self.assertLess(tr.target_date.max(), te.date.min())

    def test_quarterly_tests_do_not_overlap_or_train_on_future(self):
        dates = pd.date_range("2020-01-01", "2021-12-31")
        data = pd.concat([pd.DataFrame({"participant": str(i), "date": dates}) for i in range(5)], ignore_index=True)
        data["target_date"] = data.date + pd.Timedelta(days=1)
        test_sets = []
        for period, _, train, test in stability_splits(data):
            if period == "group5":
                continue
            self.assertLess(data.iloc[train].target_date.max(), data.iloc[test].date.min())
            test_sets.append(set(test))
        self.assertEqual(len(test_sets), 5)
        for i, first in enumerate(test_sets):
            for second in test_sets[i + 1:]:
                self.assertFalse(first & second)

    def test_event_pairs_keep_missing_reports_and_subsequent_training(self):
        dates = pd.date_range("2021-01-01", periods=100)
        data = pd.DataFrame({"participant": "a", "date": dates, "fatigue": 3.,
                             "daily_srpe": 100., "intense_threshold": 250., "intense_day": False})
        data.loc[[20, 21, 50], "intense_day"] = True
        data.loc[[20, 21, 50], "daily_srpe"] = 400.
        data.loc[22, "fatigue"] = np.nan
        observations, manifest = event_pairs(data)
        first = manifest.loc[manifest.event_date == dates[20]].iloc[0]
        self.assertEqual(first.event_intense_next3, 1)
        self.assertFalse(first.isolated_next3)
        self.assertFalse(((observations.pair_id == first.pair_id) & (observations.offset == 2)).any())
        self.assertTrue(manifest.loc[manifest.event_date == dates[50], "isolated_next3"].iloc[0])
        self.assertTrue(observations.contrast.eq(0).all())


if __name__ == "__main__":
    unittest.main()
