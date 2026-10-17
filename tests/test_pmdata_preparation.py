import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.pmdata.features import add_ewma_load_features, add_rolling_load_features
from src.data.pmdata.preparation import (
    add_target_column, build_full_calendar, compute_gap_lengths,
    load_participant_data, mark_long_gaps, prepare_dataset,
    resolve_target_mode,
)


class PMDataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def participant(self, base=None):
        root = (base or self.root) / "p01"
        (root / "pmsys").mkdir(parents=True)
        (root / "fitbit").mkdir()
        return root

    def wellness(self, root, rows):
        records = []
        for timestamp, readiness in rows:
            records.append({"effective_time_frame": timestamp, "readiness": readiness,
                            "fatigue": 2., "soreness": 3., "sleep_duration_h": 7., "sleep_quality": 3.})
        pd.DataFrame(records).to_csv(root / "pmsys/wellness.csv", index=False)

    def sessions(self, root, rows):
        pd.DataFrame(rows, columns=["end_date_time", "perceived_exertion", "duration_min"]).to_csv(root / "pmsys/srpe.csv", index=False)

    def test_independent_calendar_keeps_sessions_without_wellness(self):
        root = self.participant()
        self.wellness(root, [("2020-01-01T08:00:00Z", 5), ("2020-01-03T08:00:00Z", 7)])
        self.sessions(root, [("2020-01-02T12:00:00Z", 5, 20), ("2020-01-02T18:00:00Z", 4, 50), ("2020-01-04T12:00:00Z", 6, 30)])
        df = load_participant_data(str(root)).set_index("date")
        self.assertEqual(len(df), 4)
        self.assertEqual(df.loc["2020-01-02", "daily_srpe"], 300)
        self.assertEqual(df.loc["2020-01-02", "n_sessions"], 2)
        self.assertEqual(df.loc["2020-01-04", "daily_srpe"], 180)
        self.assertTrue(pd.isna(df.loc["2020-01-02", "readiness"]))

    def test_latest_report_uses_timestamp_not_file_order(self):
        root = self.participant()
        self.wellness(root, [("2020-01-01T20:00:00Z", 8), ("2020-01-01T08:00:00Z", 4)])
        pd.DataFrame([
            {"timestamp": "2020-01-01T09:00:00Z", "overall_score": 90},
            {"timestamp": "2020-01-01T06:00:00Z", "overall_score": 60},
        ]).to_csv(root / "fitbit/sleep_score.csv", index=False)
        df = load_participant_data(str(root))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.readiness.iloc[0], 8)
        self.assertEqual(df.sleep_overall_score.iloc[0], 90)
        self.assertEqual(df.attrs["source_audit"]["extra_same_day_wellness_reports"], 1)

    def test_no_recorded_session_is_zero_recorded_load_without_interpolation(self):
        root = self.participant()
        self.wellness(root, [("2020-01-01T08:00:00Z", 5), ("2020-01-03T08:00:00Z", 8)])
        self.sessions(root, [("2020-01-01T12:00:00Z", 5, 20), ("2020-01-03T12:00:00Z", 5, 60)])
        middle = load_participant_data(str(root)).iloc[1]
        self.assertEqual(middle.daily_srpe, 0)
        self.assertFalse(middle.srpe_reported)
        self.assertTrue(pd.isna(middle.readiness))
        self.assertTrue(pd.isna(middle.resting_heart_rate))

    def test_absent_or_incomplete_training_data_is_not_zero(self):
        root = self.participant()
        self.wellness(root, [("2020-01-01T08:00:00Z", 5)])
        self.assertTrue(load_participant_data(str(root)).daily_srpe.isna().all())
        self.sessions(root, [("2020-01-01T12:00:00Z", 5, 20), ("2020-01-01T18:00:00Z", None, 30)])
        df = load_participant_data(str(root))
        self.assertTrue(pd.isna(df.daily_srpe.iloc[0]))
        self.assertEqual(df.n_sessions.iloc[0], 2)

    def test_unknown_load_invalidates_windows_and_restarts_ewma(self):
        df = pd.DataFrame({"participant": "a", "date": pd.date_range("2020-01-01", periods=4), "daily_srpe": [100., np.nan, 300., 400.]})
        rolling = add_rolling_load_features(df, acute_window=2, chronic_window=3, missing_policy="preserve")
        self.assertTrue(rolling.acute_load.iloc[1:3].isna().all())
        self.assertEqual(rolling.acute_load.iloc[3], 700)
        self.assertTrue(pd.isna(rolling.chronic_load.iloc[3]))
        ewma = add_ewma_load_features(df, acute_span=3, missing_policy="preserve")
        self.assertTrue(pd.isna(ewma.ewma_acute_load.iloc[1]))
        self.assertEqual(ewma.ewma_acute_load.iloc[2], 300)
        self.assertEqual(ewma.ewma_acute_load.iloc[3], 350)
        # Existing API artifact retains its earlier explicit zero-fill default.
        legacy = add_ewma_load_features(df, acute_span=3)
        self.assertEqual(legacy.ewma_acute_load.iloc[1], 50)

    def test_targets_follow_calendar_and_keep_missing_observations(self):
        df = pd.DataFrame({"participant": "a", "date": pd.date_range("2020-01-01", periods=4), "readiness": [5., np.nan, 8., 9.]})
        one = add_target_column(df, "readiness")
        self.assertTrue(pd.isna(one.target.iloc[0]))
        self.assertEqual(one.target.iloc[2], 9)
        two = add_target_column(df, "readiness", horizon=2, predict_delta=True)
        self.assertEqual(two.target.iloc[0], 3)
        self.assertEqual(two.target_date.iloc[0], pd.Timestamp("2020-01-03"))
        with self.assertRaises(ValueError):
            add_target_column(df.iloc[[0, 2]], "readiness")
        with self.assertRaises(ValueError):
            build_full_calendar(pd.concat([df, df.iloc[[0]]]))
        for invalid in [0, -1, True]:
            with self.assertRaises(ValueError):
                add_target_column(df, "readiness", horizon=invalid)

    def test_no_gap_input_is_supported(self):
        df = pd.DataFrame({"participant": "a", "date": pd.date_range("2020-01-01", periods=3), "readiness": [5., 6., 7.]})
        gaps = compute_gap_lengths(df)
        self.assertTrue(gaps.empty)
        self.assertFalse(mark_long_gaps(df, gaps).in_long_gap.any())

    def test_target_mode_metadata_prevents_wrong_baseline(self):
        self.assertTrue(resolve_target_mode({"predict_delta": True}))
        self.assertFalse(resolve_target_mode({"predict_delta": False}))
        with self.assertRaises(ValueError):
            resolve_target_mode({"predict_delta": True}, requested=False)

    def test_fitbit_only_dates_and_missing_zone_measurements_survive(self):
        root = self.participant()
        self.wellness(root, [("2020-01-02T08:00:00Z", 5)])
        (root / "fitbit/resting_heart_rate.json").write_text(json.dumps([
            {"dateTime": "2020-01-01T00:00:00Z", "value": {"value": 60}},
            {"dateTime": "2020-01-03T00:00:00Z", "value": {"value": 62}},
        ]))
        (root / "fitbit/time_in_heart_rate_zones.json").write_text(json.dumps([
            {"dateTime": "2020-01-01T00:00:00Z", "value": {"valuesInZones": {"IN_DEFAULT_ZONE_2": 10}}},
        ]))
        df = load_participant_data(str(root))
        self.assertEqual(len(df), 3)
        self.assertEqual(df.resting_heart_rate.iloc[0], 60)
        self.assertEqual(df.resting_heart_rate.iloc[2], 62)
        self.assertTrue(df.hr_zone_active_minutes.isna().all())

    def write_history(self, base, days):
        root = self.participant(base)
        dates = pd.date_range("2020-01-01", periods=days)
        # Tail holes in the prefix acquire a future endpoint only in full data.
        wellness = pd.DataFrame({"effective_time_frame": dates.strftime("%Y-%m-%dT08:00:00Z"),
            "readiness": 5 + np.arange(days) % 4, "fatigue": 1 + np.arange(days) % 3,
            "soreness": 1 + np.arange(days) % 4, "sleep_duration_h": 7., "sleep_quality": 3.})
        wellness.loc[30:, ["readiness", "fatigue", "soreness"]] = [10, 5, 5]
        wellness = wellness.drop(index=[10, 11, 12, 13, 28, 29])
        wellness.to_csv(root / "pmsys/wellness.csv", index=False)
        self.sessions(root, [(d.strftime("%Y-%m-%dT12:00:00Z"), 4 + i % 3, 30) for i, d in enumerate(dates) if i % 3 == 0])
        (root / "fitbit/resting_heart_rate.json").write_text(json.dumps([
            {"dateTime": d.strftime("%Y-%m-%dT00:00:00Z"), "value": {"value": 55 + i % 4}}
            for i, d in enumerate(dates)]))
        return root

    def test_future_extension_cannot_rewrite_past_features_or_fill_long_gaps(self):
        self.write_history(self.root / "prefix", 30)
        self.write_history(self.root / "full", 45)
        with contextlib.redirect_stdout(io.StringIO()):
            short, short_daily, _ = prepare_dataset(self.root / "prefix", self.root / "short.csv", predict_delta=True)
            full, full_daily, metadata = prepare_dataset(self.root / "full", self.root / "full.csv", predict_delta=True)
        columns = ["date", "participant", "readiness", "fatigue", "soreness", "daily_srpe", "resting_heart_rate", "composite_wellness", "rhr_relative"]
        pd.testing.assert_frame_equal(short_daily[columns], full_daily.iloc[:30][columns].reset_index(drop=True))
        for column in ["readiness", "fatigue", "soreness", "composite_wellness"]:
            self.assertTrue(full_daily.loc[full_daily.date.between("2020-01-11", "2020-01-14"), column].isna().all())
        shared = full.loc[full.date.isin(short.date)].reset_index(drop=True)
        pd.testing.assert_frame_equal(short, shared)
        missing_target_day = pd.Timestamp("2020-01-10")
        self.assertFalse(full.date.eq(missing_target_day).any())
        after_gap = full.loc[full.date.eq(pd.Timestamp("2020-01-15"))].iloc[0]
        self.assertTrue(pd.isna(after_gap.readiness_lag1))
        self.assertTrue(pd.isna(after_gap.readiness_lag2))
        self.assertTrue(pd.isna(after_gap.readiness_lag3))
        self.assertFalse(short.empty)
        self.assertEqual(metadata["pipeline_version"], "pmdata-observed-calendar-v2")


if __name__ == "__main__":
    unittest.main()
