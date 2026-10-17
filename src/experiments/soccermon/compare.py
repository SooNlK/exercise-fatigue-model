"""Porównanie modeli prognozujących zmęczenie następnego dnia (SoccerMon)
========================================================================

1. Wczytuje przygotowany dataset.csv i zestawy cech z audit.json.
2. Porównuje prognozę „jutro jak dziś”, średnią treningową, Ridge i boosting,
   korzystające z bieżącego zmęczenia, wellness oraz obciążenia treningowego.
3. Ocenia modele w pięciu podziałach rozłącznych według osób oraz w podziale
   chronologicznym. Test czasowy może zawierać osoby znane z treningu.
4. Zapisuje predykcje, metryki ogólne i per osoba, ocenę korzyści z obciążenia
   oraz opis podziałów do artifacts/soccermon/comparison/.

Parametry są stałe; porównanie jest eksploracyjne i nie wdraża modeli do API.
Użycie: python -m src.experiments.soccermon.compare
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.common import SOCCERMON_DIR, PolishArgumentParser
from src.experiments.common import print_outputs, SOCCERMON_ARTIFACTS, print_table, VALUES


def make_splits(data: pd.DataFrame, cutoff: str):
    """Wyznacza rozłączne podziały według osób oraz jeden test chronologiczny."""
    for i, (train, test) in enumerate(GroupKFold(n_splits=5).split(data, groups=data.participant), 1):
        yield "group5", i, train, test
    boundary = pd.Timestamp(cutoff)
    train = np.flatnonzero(data.target_date.lt(boundary))
    test = np.flatnonzero(data.date.ge(boundary))
    if not len(train) or not len(test):
        raise ValueError("Podział chronologiczny ma pustą część treningową lub testową")
    yield "chronological", 1, train, test


def models():
    """Definiuje stałe konfiguracje modeli i odpowiadające im zestawy cech."""
    # Ta sama konfiguracja modelu pozwala ocenić korzyść z dodania obciążenia.
    def ridge():
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=10))

    def boosting():
        return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=8, min_samples_leaf=30, l2_regularization=10, early_stopping=False, random_state=42)

    return [
        ("persistence", "today_only", None),
        ("train_mean", "today_only", None),
        ("ridge_today", "today_only", ridge),
        ("ridge_wellness", "wellness", ridge),
        ("ridge_wellness_load", "wellness_load", ridge),
        ("boosting_wellness", "wellness", boosting),
        ("boosting_wellness_load", "wellness_load", boosting),
    ]


def metrics(data: pd.DataFrame) -> dict:
    """Oblicza liczebność oraz MAE, RMSE i R² podanej grupy predykcji."""
    return {
        "n": len(data), "participants": int(data.participant.nunique()),
        "MAE": float(mean_absolute_error(data.target, data.prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(data.target, data.prediction))),
        "R2": float(r2_score(data.target, data.prediction)) if len(data) > 1 else None,
    }


def paired_load_gain(predictions: pd.DataFrame) -> list[dict]:
    """Opisowe przedziały z losowania osób, warunkowe względem wytrenowanych modeli."""
    rng = np.random.default_rng(42)
    rows = []
    for validation in predictions.validation.unique():
        for family in ["ridge", "boosting"]:
            for subset in ["all", "intense"]:
                d = predictions.loc[predictions.validation.eq(validation)].copy()
                if subset == "intense":
                    d = d.loc[d.intense_day]
                d["error"] = (d.target - d.prediction).abs()
                wide = d.groupby(["participant", "model"]).error.mean().unstack()
                left, right = f"{family}_wellness", f"{family}_wellness_load"
                gains = (wide[left] - wide[right]).dropna().to_numpy()
                if not len(gains):
                    continue
                boot = rng.choice(gains, size=(2000, len(gains)), replace=True).mean(axis=1)
                rows.append({"validation": validation, "family": family, "subset": subset,
                             "participants": len(gains), "mean_player_MAE_gain_from_load": float(gains.mean()),
                             "bootstrap_95_low": float(np.quantile(boot, .025)), "bootstrap_95_high": float(np.quantile(boot, .975))})
    return rows


def run(input_dir: Path, output_dir: Path, cutoff: str) -> pd.DataFrame:
    """Ocenia modele, zapisuje wyniki i wyświetla podsumowanie."""
    data = pd.read_csv(input_dir / "dataset.csv", parse_dates=["date", "target_date"])
    audit = json.loads((input_dir / "audit.json").read_text())
    parts = []
    split_log = []
    for validation, fold, train, test in make_splits(data, cutoff):
        tr, te = data.iloc[train], data.iloc[test]
        split_log.append({"validation": validation, "fold": fold, "train_rows": len(tr), "test_rows": len(te),
                          "train_participants": int(tr.participant.nunique()), "test_participants": int(te.participant.nunique()),
                          "train_max_target_date": str(tr.target_date.max().date()), "test_min_date": str(te.date.min().date())})
        for name, feature_set, factory in models():
            if name == "persistence":
                pred = te.fatigue.to_numpy()
            elif name == "train_mean":
                pred = np.full(len(te), tr.target.mean())
            else:
                estimator = factory()
                cols = audit["feature_sets"][feature_set]
                estimator.fit(tr[cols], tr.target)
                pred = estimator.predict(te[cols])
            part = te[["participant", "team", "date", "target_date", "fatigue", "target", "intense_day"]].copy()
            part["prediction"] = np.clip(pred, 1, 5)
            part["validation"], part["fold"], part["model"] = validation, fold, name
            parts.append(part)
        print(f"Zakończono podział {fold} ({VALUES.get(validation, validation)}): trening={len(tr)}, test={len(te)}", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    scores = []
    per_player = []
    for (validation, model), group in predictions.groupby(["validation", "model"]):
        for subset in ["all", "intense"]:
            selected = group if subset == "all" else group.loc[group.intense_day]
            if selected.empty:
                continue
            row = {"validation": validation, "model": model, "subset": subset, **metrics(selected)}
            player_maes = []
            for participant, pg in selected.groupby("participant"):
                pm = metrics(pg)
                player_maes.append(pm["MAE"])
                per_player.append({"validation": validation, "model": model, "subset": subset, "participant": participant, **pm})
            row["MAE_player_mean"] = float(np.mean(player_maes))
            scores.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = pd.DataFrame(scores)
    summary.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(per_player).to_csv(output_dir / "metrics_by_player.csv", index=False)
    gains = paired_load_gain(predictions)
    pd.DataFrame(gains).to_csv(output_dir / "load_gain.csv", index=False)
    (output_dir / "validation.json").write_text(json.dumps({"cutoff": cutoff, "splits": split_log,
        "parameters": {"ridge_alpha": 10, "boosting_max_iter": 100, "boosting_max_leaf_nodes": 8,
                       "boosting_min_samples_leaf": 30, "boosting_l2": 10, "random_state": 42},
        "prediction_clipping": [1, 5], "hyperparameter_search": False,
        "bootstrap_note": "Losowanie osób przy stałych wytrenowanych modelach; bez wnioskowania przyczynowego i pełnej niepewności wyboru modelu."}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(summary)
    print_outputs("SoccerMon — porównanie modeli", output_dir,
                  ["predictions.csv", "metrics.csv", "metrics_by_player.csv", "load_gain.csv", "validation.json"])
    return summary


if __name__ == "__main__":
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SOCCERMON_DIR, help='Folder przygotowanego SoccerMon z dataset.csv i audit.json.')
    parser.add_argument("--output-dir", type=Path, default=SOCCERMON_ARTIFACTS / "comparison", help='Folder plików wynikowych; domyślnie podfolder danego eksperymentu w artifacts/.')
    parser.add_argument("--cutoff", default="2021-07-01", help='Początek testu chronologicznego w formacie RRRR-MM-DD (domyślnie: 2021-07-01).')
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.cutoff)
