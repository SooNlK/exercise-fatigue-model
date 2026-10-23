"""Trening i zapis dwóch metod prognozowania ryzyka SoccerMon.

1. Wczytuje dataset.csv z przygotowanych danych.
2. Sprawdza cel następnego dnia, skale, historię i zestaw cech.
3. Dopasowuje osobniczą metodę częstości oraz klasyfikator boosting.
4. Zapisuje osobne pliki joblib i metadane JSON do artifacts/soccermon/models/.
5. Wczytuje modele ponownie i sprawdza zgodność prawdopodobieństw.

Domyślnie wykorzystuje cały zbiór. Opcja --train-before ogranicza daty celów
do dni wcześniejszych od podanej granicy. To trening do użycia w aplikacji,
nie nowa walidacja jakości; wyników na treningu nie raportujemy jako jakości.
Nie zmienia modelu obecnego API ani wcześniejszych wyników eksperymentów.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data.common import PROJECT_ROOT, SOCCERMON_DIR, PolishArgumentParser
from src.data.soccermon.preparation import WELLNESS, make_features
from src.experiments.common import SOCCERMON_ARTIFACTS, print_outputs
from src.models.soccermon.predictor import daily_date, load_risk_model
from src.models.soccermon.risk import (
    BOOSTING_PARAMS, PIPELINE_VERSION, PRIOR_STRENGTH, TARGET_DEFINITION,
    PersonalIntensityRisk, event_labels, make_boosting, risk_probabilities,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_training_data(input_dir, train_before=None):
    """Odrzuca niezgodny zbiór; nie uzupełnia brakujących celów."""
    data = pd.read_csv(input_dir / "dataset.csv", parse_dates=["date", "target_date"])
    sample = pd.DataFrame({"date": [pd.Timestamp("2000-01-01")], "participant": ["schema"]})
    for name in [*WELLNESS, "daily_srpe"]:
        sample[name] = np.nan
    _, sets = make_features(sample)
    columns = sets["wellness_load"]
    required = ["participant", "date", "target_date", "target", "intense_day", "fatigue_history_count", *columns]
    if not set(required).issubset(data.columns):
        raise ValueError("Brak wymaganych kolumn treningowych.")
    if data.participant.isna().any() or data.duplicated(["participant", "date"]).any():
        raise ValueError("Wymagane identyfikatory osób oraz unikalne pary osoba–data.")
    for name in ["date", "target_date"]:
        data[name] = pd.to_datetime(data[name].map(daily_date))
    if not (data.target_date - data.date).eq(pd.Timedelta(days=1)).all():
        raise ValueError("Cel musi przypadać na następny dzień kalendarzowy.")
    if not data.fatigue_history_count.ge(7).all():
        raise ValueError("Trening wymaga 7 wcześniejszych raportów z 21 dni.")
    if data.intense_day.dtype != bool:
        raise ValueError("intense_day musi zawierać wartości logiczne bez braków.")
    event_labels(data)
    if np.isinf(data[columns].to_numpy(dtype=float)).any():
        raise ValueError("Cechy zawierają nieskończoności.")
    if train_before is not None:
        data = data.loc[data.target_date.lt(daily_date(train_before))].copy()
    if data.empty or event_labels(data).nunique() != 2:
        raise ValueError("Trening wymaga obserwacji obu klas zdarzenia.")
    empty_features = data[columns].columns[data[columns].isna().all()].tolist()
    if empty_features:
        raise ValueError(f"Za mało danych do treningu: cechy bez żadnego pomiaru: {empty_features}.")
    return data.reset_index(drop=True), columns


def train_models(input_dir=SOCCERMON_DIR, output_dir=SOCCERMON_ARTIFACTS / "models", train_before=None):
    """Zapisuje oba modele z identyczną próbą i kontraktem prognozy."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    data, columns = read_training_data(input_dir, train_before)
    y = event_labels(data)
    sources = {str((input_dir / "dataset.csv").resolve()): sha256(input_dir / "dataset.csv")}
    code_paths = ["src/models/soccermon/risk.py", "src/models/soccermon/predictor.py",
                  "src/data/soccermon/preparation.py", "src/experiments/soccermon/train_risk.py"]
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "target_definition": TARGET_DEFINITION,
        "horizon_days": 1,
        "feature_cols": columns,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(data),
        "training_participants": int(data.participant.nunique()),
        "training_events": int(y.sum()),
        "training_min_date": str(data.date.min().date()),
        "training_max_date": str(data.date.max().date()),
        "training_max_target_date": str(data.target_date.max().date()),
        "train_before": str(daily_date(train_before).date()) if train_before is not None else None,
        "training_counts_by_participant": {str(k): int(v) for k, v in data.participant.value_counts().items()},
        "source_sha256": sources,
        "code_sha256": {p: sha256(PROJECT_ROOT / p) for p in code_paths},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "history_contract": "Cała dostępna historia, minimum 29 dni kalendarzowych; dzisiejsze fatigue i minimum 7 wcześniejszych raportów z 21 dni; braki pozostają nieznane; cechy odczytane jak dataset.csv",
        "unknown_intensity_policy": "Jak w eksperymencie: brak progu lub obciążenia daje intense_day=False; odpowiedź oznacza tę sytuację jako nieznaną.",
        "personalization_policy": "Częstości osoby z treningu; dla nieobserwowanej kombinacji fallback do populacji. Historia wejściowa nie aktualizuje częstości.",
        "validation_status": "Trening finalny nie stanowi oceny jakości. Dotychczasowe wyniki są eksploracyjne; przewaga nad mocnym prostym odniesieniem pozostaje niepewna.",
    }
    models = {
        "personal_intensity_frequency": ("personal_intensity_risk", PersonalIntensityRisk().fit(data), {"prior_strength": PRIOR_STRENGTH}),
        "boosting_load": ("boosting_risk", make_boosting().fit(data[columns], y), BOOSTING_PARAMS),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    identity = hashlib.sha256(json.dumps({"sources": sources, "code": metadata["code_sha256"],
                                         "cutoff": metadata["train_before"]}, sort_keys=True).encode()).hexdigest()[:12]
    for method, (filename, model, params) in models.items():
        artifact = {**metadata, "method": method, "model_id": f"{PIPELINE_VERSION}-{method}-{identity}",
                    "parameters": params, "model": model}
        path = output_dir / f"{filename}.joblib"
        joblib.dump(artifact, path)
        restored = load_risk_model(path)
        expected, _ = risk_probabilities(artifact, data)
        actual, _ = risk_probabilities(restored, data)
        np.testing.assert_array_equal(expected, actual)
        sidecar = {k: v for k, v in artifact.items() if k != "model"}
        sidecar.update({"artifact_sha256": sha256(path), "reload_predictions_identical": True})
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.extend([path.name, meta_path.name])
    print(f"Trening: {len(data)} wierszy, {data.participant.nunique()} osób, {int(y.sum())} pogorszeń.")
    print(f"Ostatnia data celu treningowego: {metadata['training_max_target_date']}.")
    print_outputs("Zapis dwóch metod ryzyka SoccerMon", output_dir, paths)
    return paths


def main():
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SOCCERMON_DIR, help="Folder przygotowanych danych SoccerMon.")
    parser.add_argument("--output-dir", type=Path, default=SOCCERMON_ARTIFACTS / "models", help="Folder modeli i metadanych.")
    parser.add_argument("--train-before", help="Użyj wyłącznie celów sprzed daty RRRR-MM-DD.")
    args = parser.parse_args()
    train_models(args.input_dir, args.output_dir, args.train_before)


if __name__ == "__main__":
    main()
