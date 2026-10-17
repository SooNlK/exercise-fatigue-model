"""Ocena dodatkowej wartości danych GPS w prognozie zmęczenia (SoccerMon)
========================================================================

1. Łączy dataset.csv z podsumowaniami gps_daily.csv po osobie i dacie.
2. Zachowuje wspólną próbę dni z rzeczywistym pomiarem celu i danymi GPS,
   aby porównywane modele korzystały z tych samych obserwacji.
3. Porównuje prognozę „jutro jak dziś”, Ridge i boosting bez GPS oraz
   z dodatkowymi statystykami sygnałów i liczbą próbek.
4. Testuje na niewidzianych osobach i w dwóch późniejszych okresach 2020 r.,
   osobno raportując wszystkie dni i dni po intensywnym wysiłku.
5. Zapisuje wspólną próbę, predykcje, metryki, podziały i audyt do
   artifacts/soccermon/objective/.

To pilotaż bez strojenia parametrów, na części sezonu jednego zespołu.
Nie odczytuje ponownie surowych plików Parquet i nie wdraża modelu do API.
Użycie: python -m src.experiments.soccermon.objective_pilot
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from src.data.common import SOCCERMON_DIR, SOCCERMON_OBJECTIVE_DIR, PolishArgumentParser
from src.experiments.common import print_outputs, SOCCERMON_ARTIFACTS, print_table, VALUES
from src.data.soccermon.objective import GPS_FEATURES
from src.experiments.soccermon.compare import metrics, models


def pilot_splits(data):
    """Wyznacza testy na niewidzianych osobach i w późniejszych okresach."""
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(data, groups=data.participant), 1):
        yield "leave_one_player_out", fold, train, test
    boundaries = ["2020-09-01", "2020-11-01", "2021-01-01"]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        train = np.flatnonzero(data.target_date.lt(pd.Timestamp(start)))
        test = np.flatnonzero(data.date.ge(pd.Timestamp(start)) & data.target_date.lt(pd.Timestamp(end)))
        if len(train) and len(test):
            yield f"from_{start}", 1, train, test


def common_cohort(subjective, gps):
    """Wybiera wspólne dni z wellness oraz poprawnymi podsumowaniami GPS."""
    data = subjective.merge(gps[["participant", "date"] + GPS_FEATURES], on=["participant", "date"], how="inner", validate="one_to_one")
    return data.loc[data.gps_speed_mean.notna() & data.gps_impulse_mean.notna() & data.gps_sample_count.gt(0)].reset_index(drop=True)


def run(subjective_dir, gps_dir, output):
    """Porównuje modele na tej samej próbie i zapisuje ocenę korzyści z GPS."""
    data = pd.read_csv(subjective_dir / "dataset.csv", parse_dates=["date", "target_date"])
    gps = pd.read_csv(gps_dir / "gps_daily.csv", parse_dates=["date"])
    audit = json.loads((subjective_dir / "audit.json").read_text())
    data = common_cohort(data, gps)
    if data.participant.nunique() < 3:
        raise ValueError("Wymagane co najmniej trzy osoby z dopasowanymi danymi")
    baseline_cols = audit["feature_sets"]["wellness_load"]
    factories = {name: factory for name, _, factory in models()}
    configurations = [("persistence", [], None),
                      ("ridge_base", baseline_cols, factories["ridge_wellness_load"]),
                      ("ridge_base_gps", baseline_cols + GPS_FEATURES, factories["ridge_wellness_load"]),
                      ("boosting_base", baseline_cols, factories["boosting_wellness_load"]),
                      ("boosting_base_samples", baseline_cols + ["gps_sample_count"], factories["boosting_wellness_load"]),
                      ("boosting_base_signals", baseline_cols + [c for c in GPS_FEATURES if c != "gps_sample_count"], factories["boosting_wellness_load"]),
                      ("boosting_base_gps", baseline_cols + GPS_FEATURES, factories["boosting_wellness_load"])]
    parts, split_log = [], []
    for validation, fold, train, test in pilot_splits(data):
        tr, te = data.iloc[train], data.iloc[test]
        split_log.append({"validation": validation, "fold": fold, "train_rows": len(tr), "test_rows": len(te),
                          "train_max_target_date": str(tr.target_date.max().date()), "test_min_date": str(te.date.min().date())})
        for name, columns, factory in configurations:
            if factory is None:
                pred = te.fatigue.to_numpy()
            else:
                estimator = factory()
                estimator.fit(tr[columns], tr.target)
                pred = estimator.predict(te[columns])
            part = te[["participant", "date", "target_date", "target", "intense_day"]].copy()
            part["prediction"] = np.clip(pred, 1, 5)
            part["validation"], part["fold"], part["model"] = validation, fold, name
            parts.append(part)
        print(f"Zakończono pilotaż GPS: podział {fold}, walidacja {VALUES.get(validation, validation).replace("from_", "od ")}", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    summaries, gains = [], []
    rng = np.random.default_rng(42)
    for validation, d in predictions.groupby("validation"):
        for subset in ["all", "intense"]:
            selected = d if subset == "all" else d.loc[d.intense_day]
            if selected.empty:
                continue
            for name, group in selected.groupby("model"):
                row = {"validation": validation, "subset": subset, "model": name, **metrics(group)}
                row["MAE_player_mean"] = float(group.assign(error=(group.target-group.prediction).abs()).groupby("participant").error.mean().mean())
                summaries.append(row)
            errors = selected.assign(error=(selected.target-selected.prediction).abs()).groupby(["participant", "model"]).error.mean().unstack()
            for family in ["ridge", "boosting"]:
                delta = (errors[f"{family}_base"] - errors[f"{family}_base_gps"]).dropna().to_numpy()
                boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
                gains.append({"validation": validation, "subset": subset, "family": family, "participants": len(delta),
                              "player_mean_MAE_gain_from_gps": float(delta.mean()),
                              "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))})
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "common_cohort.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(gains).to_csv(output / "gps_gain.csv", index=False)
    pd.DataFrame(split_log).to_csv(output / "splits.csv", index=False)
    (output / "audit.json").write_text(json.dumps({"rows": len(data), "participants": int(data.participant.nunique()),
        "intense_rows": int(data.intense_day.sum()), "date_start": str(data.date.min().date()), "date_end": str(data.date.max().date()),
        "base_feature_count": len(baseline_cols), "gps_features": GPS_FEATURES,
        "fair_comparison": "Te same zaobserwowane daty i osoby we wszystkich wariantach; każdy model uczony ponownie na tej wspólnej próbie",
        "bootstrap": "2000 losowań osób przy stałych wytrenowanych modelach; przedziały punktowe bez uwzględnienia wyboru modelu",
        "limitations": "Jeden zespół i część sezonu 2020; statystyki w jednostkach źródła; dobór dni według dostępności GPS; bez pełnego modelowania sygnałów i strojenia parametrów"}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(summary)
    print_outputs("SoccerMon — pilotaż GPS", output,
                  ["common_cohort.csv", "predictions.csv", "metrics.csv", "gps_gain.csv", "splits.csv", "audit.json"])


if __name__ == "__main__":
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument("--subjective-dir", type=Path, default=SOCCERMON_DIR, help='Folder przygotowanych dziennych danych SoccerMon.')
    parser.add_argument("--gps-dir", type=Path, default=SOCCERMON_OBJECTIVE_DIR, help='Folder dziennych podsumowań GPS (gps_daily.csv).')
    parser.add_argument("--output-dir", type=Path, default=SOCCERMON_ARTIFACTS / "objective", help='Folder plików wynikowych; domyślnie podfolder danego eksperymentu w artifacts/.')
    args = parser.parse_args()
    run(args.subjective_dir, args.gps_dir, args.output_dir)
