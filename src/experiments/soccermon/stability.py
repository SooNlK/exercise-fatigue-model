"""Stabilność prognoz i analiza zmęczenia po intensywnym wysiłku (SoccerMon)
============================================================================

1. Wczytuje dzienne raporty i sesje z oryginalnego zbioru SoccerMon.
2. Porównuje obciążenie opublikowane z sumą sesji, zachowując tę samą próbę
   i definicję intensywnego dnia dla obu wariantów.
3. Sprawdza modele w podziałach według osób i pięciu kolejnych kwartałach.
4. Dopasowuje intensywne epizody do dni kontrolnych tej samej osoby i opisuje
   zmiany fatigue w oknie od trzech dni przed do siedmiu dni po wysiłku.
5. Zapisuje predykcje, metryki, pary epizodów, audyt i wykres do
   artifacts/soccermon/stability/.

Nie stroi parametrów. Analiza epizodów jest obserwacyjna i retrospektywna;
nie wyznacza przyczynowego wpływu treningu ani czasu pełnej regeneracji.
Użycie: python -m src.experiments.soccermon.stability --data-dir /sciezka/do/soccermon
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.common import PolishArgumentParser
from src.experiments.common import print_outputs, SOCCERMON_ARTIFACTS, print_table, VALUES
from src.data.soccermon.preparation import load_daily, make_features
from src.experiments.soccermon.compare import metrics, models


PERIODS = ["2020-10-01", "2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01"]
OFFSETS = list(range(-3, 8))


def source_variants(root: Path):
    """Buduje dwa warianty obciążenia i audyt rozbieżności źródeł."""
    daily, audit = load_daily(root)
    with (root / "training-load" / "session.json").open() as stream:
        raw = json.load(stream)
    records = pd.DataFrame([{"participant": p, **r} for p, rows in raw.items() for r in rows])
    records["date"] = pd.to_datetime(records.date, format="%d.%m.%Y")
    sums = records.groupby(["participant", "date"]).srpe.sum().rename("session_srpe").reset_index()
    joined = daily.merge(sums, on=["participant", "date"], how="left", validate="one_to_one")
    # Zachowujemy tę samą interpretację braku zapisanego obciążenia co w daily_load.csv.
    joined["session_srpe"] = joined.session_srpe.fillna(0)
    differences = joined.loc[~np.isclose(joined.daily_srpe, joined.session_srpe),
                             ["participant", "team", "date", "daily_srpe", "session_srpe"]].copy()
    differences["published_minus_sessions"] = differences.daily_srpe - differences.session_srpe
    published, sets = make_features(daily)
    alternative = daily.copy()
    alternative["daily_srpe"] = joined.session_srpe.to_numpy()
    sessions, _ = make_features(alternative)
    changed_events = int((published.intense_day != sessions.intense_day).sum())
    keep = published.fatigue.notna() & published.target.notna() & published.fatigue_history_count.ge(7)
    # Wspólna próba i definicja epizodów zapewniają porównanie źródeł parami.
    sessions["intense_day"] = published.intense_day
    datasets = {"published": published.loc[keep].reset_index(drop=True),
                "sessions": sessions.loc[keep].reset_index(drop=True)}
    audit.update({"event_labels_changed_with_session_load_full_calendar": changed_events,
                  "difference_total": float(differences.published_minus_sessions.sum()),
                  "difference_min": float(differences.published_minus_sessions.min()),
                  "difference_max": float(differences.published_minus_sessions.max()),
                  "difference_by_team": differences.groupby("team").size().to_dict(),
                  "difference_by_year": {str(k): int(v) for k, v in differences.groupby(differences.date.dt.year).size().items()},
                  "comparison_policy": "Ta sama próba zaobserwowanych celów i intensywne dni według opublikowanego obciążenia; zachowano wszystkie sesje, również nierozróżnialne powtórzenia."})
    return published, datasets, sets, differences, audit


def stability_splits(data):
    """Wyznacza podziały według osób i testy w kolejnych kwartałach."""
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(data, groups=data.participant), 1):
        yield "group5", fold, train, test
    for start, end in zip(PERIODS[:-1], PERIODS[1:]):
        train = np.flatnonzero(data.target_date.lt(pd.Timestamp(start)))
        test = np.flatnonzero(data.date.ge(pd.Timestamp(start)) & data.target_date.lt(pd.Timestamp(end)))
        if len(train) and len(test):
            yield start[:7], 1, train, test


def compare(datasets, sets, output):
    """Porównuje źródła obciążenia i zapisuje predykcje oraz metryki."""
    base = datasets["published"]
    results, predictions, logs = [], [], []
    for period, fold, train, test in stability_splits(base):
        logs.append({"period": period, "fold": fold, "train_rows": len(train), "test_rows": len(test),
                     "train_max_target_date": str(base.iloc[train].target_date.max().date()),
                     "test_min_date": str(base.iloc[test].date.min().date()),
                     "test_max_target_date": str(base.iloc[test].target_date.max().date())})
        for source, data in datasets.items():
            tr, te = data.iloc[train], data.iloc[test]
            for name, feature_set, factory in models():
                if name == "train_mean":
                    continue
                if name == "persistence":
                    pred = te.fatigue.to_numpy()
                else:
                    estimator = factory()
                    columns = sets[feature_set]
                    estimator.fit(tr[columns], tr.target)
                    pred = estimator.predict(te[columns])
                part = te[["participant", "date", "target_date", "target", "intense_day"]].copy()
                part["prediction"] = np.clip(pred, 1, 5)
                part["source"], part["period"], part["fold"], part["model"] = source, period, fold, name
                predictions.append(part)
        print(f"Zakończono ocenę stabilności: okres {VALUES.get(period, period)}, podział {fold}", flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    for (source, period, name), d in pred.groupby(["source", "period", "model"]):
        for subset in ["all", "intense"]:
            selected = d if subset == "all" else d.loc[d.intense_day]
            if selected.empty:
                continue
            row = {"source": source, "period": period, "model": name, "subset": subset, **metrics(selected)}
            row["MAE_player_mean"] = float(selected.assign(error=(selected.target - selected.prediction).abs()).groupby("participant").error.mean().mean())
            results.append(row)
    table = pd.DataFrame(results)
    pred.to_csv(output / "stability_predictions.csv", index=False)
    table.to_csv(output / "stability_metrics.csv", index=False)
    pd.DataFrame(logs).to_csv(output / "stability_splits.csv", index=False)
    # Przedziały różnic per osoba są warunkowe względem stałych modeli.
    rng = np.random.default_rng(42)
    gains = []
    for (source, period), d in pred.groupby(["source", "period"]):
        for subset in ["all", "intense"]:
            selected = d if subset == "all" else d.loc[d.intense_day]
            wide = selected.assign(error=(selected.target - selected.prediction).abs()).groupby(["participant", "model"]).error.mean().unstack()
            for reference in ["persistence", "boosting_wellness"]:
                delta = (wide[reference] - wide.boosting_wellness_load).dropna().to_numpy()
                if not len(delta):
                    continue
                boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
                gains.append({"source": source, "period": period, "subset": subset, "reference": reference,
                              "player_mean_MAE_gain": float(delta.mean()), "ci_low": float(np.quantile(boot, .025)),
                              "ci_high": float(np.quantile(boot, .975)), "players": len(delta)})
    pd.DataFrame(gains).to_csv(output / "stability_gains.csv", index=False)
    return table


def event_pairs(calendar):
    """Dobiera rozłączne okna epizodów i dni kontrolne tej samej osoby.

    Dobór ma charakter opisowy i może korzystać z późniejszych obserwacji
    zakończonych epizodów; informacje te nie trafiają do cech predykcyjnych.
    """
    rows, manifest = [], []
    for participant, group in calendar.groupby("participant"):
        g = group.sort_values("date").reset_index(drop=True)
        fatigue = g.fatigue.to_numpy(dtype=float)
        load = g.daily_srpe.to_numpy(dtype=float)
        intense = g.intense_day.to_numpy(dtype=bool)
        known = g.intense_threshold.notna().to_numpy()
        baseline = pd.Series(fatigue).shift(1).rolling(3, min_periods=2).mean().to_numpy()
        # Kontrole: dodatnie, nieintensywne obciążenie, brak intensywnego dnia
        # w sąsiedztwie ±1 dnia i zbliżone zmęczenie przed epizodem.
        neighbors = pd.Series(intense.astype(int)).rolling(3, center=True, min_periods=3).sum().to_numpy()
        candidates = np.flatnonzero((load > 0) & known & ~intense & (neighbors == 0) & np.isfinite(baseline) & np.isfinite(fatigue))
        used_controls = set()
        last_event = -100
        for index in np.flatnonzero(intense & np.isfinite(baseline) & np.isfinite(fatigue)):
            if index < 3 or index + 7 >= len(g) or index - last_event < 11:
                continue
            options = [c for c in candidates if 3 <= c < len(g) - 7 and c not in used_controls
                       and 14 <= abs(c - index) <= 42
                       and g.date.iloc[c].dayofweek == g.date.iloc[index].dayofweek
                       and abs(baseline[c] - baseline[index]) <= .5]
            if not options:
                continue
            control = min(options, key=lambda c: (abs(baseline[c] - baseline[index]), abs(c - index), c))
            last_event = index
            used_controls.add(control)
            pair_id = f"{participant}:{g.date.iloc[index].date()}"
            next3_event_intense = int(intense[index + 1:index + 4].sum())
            next3_control_intense = int(intense[control + 1:control + 4].sum())
            isolated = next3_event_intense == 0 and next3_control_intense == 0 and known[index + 1:index + 4].all() and known[control + 1:control + 4].all()
            manifest.append({"participant": participant, "pair_id": pair_id,
                             "event_date": g.date.iloc[index], "control_date": g.date.iloc[control],
                             "event_load": load[index], "control_load": load[control],
                             "event_baseline": baseline[index], "control_baseline": baseline[control],
                             "event_intense_next3": next3_event_intense, "control_intense_next3": next3_control_intense,
                             "isolated_next3": isolated})
            for offset in OFFSETS:
                event_value, control_value = fatigue[index + offset], fatigue[control + offset]
                # Porównujemy zmiany wyłącznie tam, gdzie istnieją oba raporty pary.
                if not (np.isfinite(event_value) and np.isfinite(control_value)):
                    continue
                rows.append({"participant": participant, "pair_id": pair_id, "offset": offset,
                             "event_change": event_value - baseline[index], "control_change": control_value - baseline[control],
                             "contrast": (event_value - baseline[index]) - (control_value - baseline[control]),
                             "event_load": load[index + offset], "control_load": load[control + offset],
                             "isolated_next3": isolated})
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def event_summary(rows):
    """Podsumowuje zmiany wokół wysiłku i punktowe przedziały z losowania osób."""
    rng = np.random.default_rng(42)
    summaries = []
    for subset in ["matched", "isolated_next3"]:
        selected = rows if subset == "matched" else rows.loc[rows.isolated_next3]
        for offset, group in selected.groupby("offset"):
            per_person = group.groupby("participant")[["event_change", "control_change", "contrast", "event_load", "control_load"]].mean()
            contrast = per_person.contrast.to_numpy()
            boot = rng.choice(contrast, size=(2000, len(contrast)), replace=True).mean(axis=1)
            summaries.append({"subset": subset, "offset": int(offset), "pairs": len(group), "participants": len(per_person),
                              **{c: float(per_person[c].mean()) for c in per_person},
                              "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))})
    return pd.DataFrame(summaries)


def plot_results(metrics_table, trajectory, output):
    """Zapisuje wykres stabilności prognoz oraz opisowych zmian po wysiłku."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    d = metrics_table.loc[(metrics_table.source == "published") & (metrics_table.subset == "all") & (metrics_table.period != "group5")]
    for name, label in [("persistence", "Jutro jak dziś"), ("boosting_wellness", "Wellness"), ("boosting_wellness_load", "Wellness + obciążenie")]:
        x = d.loc[d.model == name].sort_values("period")
        axes[0].plot(x.period, x.MAE, marker="o", label=label)
    axes[0].set(xlabel="Początek kwartału testowego", ylabel="MAE (punkty skali fatigue)", title="Walidacja w kolejnych kwartałach")
    axes[0].legend(fontsize=8)
    for subset, label in [("matched", "Dopasowane epizody"), ("isolated_next3", "Bez kolejnego intensywnego dnia przez 3 dni")]:
        x = trajectory.loc[trajectory.subset == subset].sort_values("offset")
        axes[1].plot(x.offset, x.contrast, marker="o", label=label)
        axes[1].fill_between(x.offset, x.ci_low, x.ci_high, alpha=.15)
    axes[1].axhline(0, color="gray", linewidth=.8)
    axes[1].axvline(0, color="gray", linewidth=.8, linestyle="--")
    axes[1].set(xlabel="Dzień względem intensywnego wysiłku", ylabel="Zmiana fatigue względem kontroli\nWartość ujemna = większe zmęczenie",
                title="Porównanie opisowe; punktowe przedziały 95%")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "stability_and_events.png", dpi=180)
    plt.close(fig)


def run(root, output):
    """Uruchamia ocenę stabilności, analizę epizodów i zapis raportów."""
    output.mkdir(parents=True, exist_ok=True)
    calendar, datasets, sets, differences, audit = source_variants(root)
    differences.to_csv(output / "load_discrepancies.csv", index=False)
    table = compare(datasets, sets, output)
    rows, manifest = event_pairs(calendar)
    if rows.empty:
        raise ValueError("Brak dopasowanych epizodów; sprawdź kryteria doboru")
    summary = event_summary(rows)
    rows.to_csv(output / "event_observations.csv", index=False)
    manifest.to_csv(output / "event_pairs.csv", index=False)
    summary.to_csv(output / "event_summary.csv", index=False)
    audit.update({"matched_episode_pairs": len(manifest), "matched_participants": int(manifest.participant.nunique()),
                  "isolated_next3_pairs": int(manifest.isolated_next3.sum()), "period_boundaries": PERIODS,
                  "event_matching": "Ta sama osoba i dzień tygodnia; kontrola 14–42 dni od epizodu; różnica średniego fatigue z poprzednich 3 dni <=0,5 (minimum 2 raporty); epizody oddalone o >=11 dni; kontrola użyta najwyżej raz, bez intensywnego dnia w sąsiedztwie ±1 dnia; obserwacje parami dla dni −3…+7",
                  "uncertainty": "2000 losowań osób; równe wagi osób; przedziały punktowe bez korekty wielokrotnych porównań; dobór obserwacyjny nie ustala przyczynowości",
                  "subsequent_training": "Zachowano liczbę intensywnych dni przez kolejne 3 dni oraz dzienne obciążenie; wydzielony podzbiór wyklucza kolejne intensywne dni, ale nie wszystkie treningi"})
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_results(table, summary, output)
    print_table(table.loc[(table.source == "published") & (table.subset == "all") & table.model.isin(["persistence", "boosting_wellness", "boosting_wellness_load"])])
    print_table(summary)
    print_outputs("SoccerMon — stabilność i epizody wysiłku", output,
                  ["load_discrepancies.csv", "stability_predictions.csv", "stability_metrics.csv",
                   "stability_splits.csv", "stability_gains.csv", "event_observations.csv",
                   "event_pairs.csv", "event_summary.csv", "audit.json", "stability_and_events.png"])


if __name__ == "__main__":
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help='Folder oryginalnych danych SoccerMon (wellness i training-load).')
    parser.add_argument("--output-dir", type=Path, default=SOCCERMON_ARTIFACTS / "stability", help='Folder plików wynikowych; domyślnie podfolder danego eksperymentu w artifacts/.')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
