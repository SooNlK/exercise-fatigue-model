"""
Trenowanie i zapis finalnego modelu produkcyjnego
====================================================

Obsługuje dwa tryby. Metadane zbioru określają, czy target jest deltą czy
poziomem. Wyniki ze starszego pipeline'u z interpolacją wymagają ponownej oceny.

1. --predict-delta: model przewiduje zmianę
   samopoczucia (Δwellness = composite_wellness(t+1) - composite_wellness(t)).
   Rekonstruuje jutrzejszy poziom jako dzisiejszy_poziom + przewidziana_delta.
   Zachowuje historyczny format tercyli wskaźnika, który nie jest zwalidowaną
   miarą czasu regeneracji.
2. --no-predict-delta: model przewiduje poziom bezwzględny wprost.

Domyślny zapis jest oddzielony od modelu obecnego API. Nowy model wymaga
zgodnego preprocessingu inferencji; samo przygotowanie danych nie wdraża modelu.
Zapis: artifacts/pmdata/model.joblib, obejmujący model, cechy i metadane.

Użycie:
    python -m src.experiments.pmdata.train_final --dataset ml_dataset_delta.csv --predict-delta --output model.joblib
"""

from pathlib import Path


import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.data.common import PMDATA_DIR, PolishArgumentParser
from src.experiments.common import PMDATA_ARTIFACTS, print_outputs
from src.data.pmdata.preparation import read_dataset_metadata, resolve_target_mode


def main():
    parser = PolishArgumentParser(description="Trenowanie finalnego modelu produkcyjnego")
    parser.add_argument("--dataset", default=str(PMDATA_DIR / "ml_dataset_delta.csv"), help='Ścieżka przygotowanego zbioru CSV.')
    parser.add_argument("--output", default=str(PMDATA_ARTIFACTS / "model.joblib"), help='Ścieżka zapisu modelu wraz z metadanymi.')
    parser.add_argument(
        "--predict-delta",
        action="store_true",
        default=None,
        help="Target typu delta; domyślnie tryb odczytywany z metadanych zbioru",
    )
    parser.add_argument("--no-predict-delta", dest="predict_delta", action="store_false", help='Prognozuj poziom wskaźnika zamiast zmiany.')
    parser.add_argument(
        "--level-col",
        default=None,
        help="Nazwa kolumny reprezentującej dzisiejszy poziom (do rekonstrukcji jutrzejszego poziomu z delty)",
    )
    args = parser.parse_args()

    metadata = read_dataset_metadata(args.dataset)
    try:
        args.predict_delta = resolve_target_mode(metadata, args.predict_delta, default=True)
    except ValueError as exc:
        parser.error(str(exc))
    args.level_col = args.level_col or metadata.get("level_col", "composite_wellness")

    df = pd.read_csv(args.dataset)

    non_feature_cols = {"participant", "date", "target"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    n_before = len(df)
    df = df.dropna(subset=feature_cols + ["target"])
    if df.empty:
        parser.error("Brak kompletnych zaobserwowanych wierszy do treningu")
    print(f"Usunięto {n_before - len(df)} wierszy z brakami. Trenuję na {len(df)} wierszach.")

    X = df[feature_cols]
    y = df["target"]

    if args.predict_delta:
        # Stały wariant badawczy, wymagający ponownej oceny na poprawionych danych.
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=15,
            max_features=0.5, random_state=42,
        )

    model.fit(X, y)
    print(f"Model wytrenowany na pełnym zbiorze danych (tryb: {'delta' if args.predict_delta else 'poziom'}).")

    # progi kategoryzacji (tercyle)
    # niezależnie od tego, czy model przewiduje wprost poziom, czy deltę
    if args.predict_delta:
        reconstructed_level = df[args.level_col] + df["target"]
        tercyle = np.percentile(reconstructed_level.dropna(), [33.33, 66.67])
    else:
        tercyle = np.percentile(y, [33.33, 66.67])

    thresholds = {
        "dluga_regeneracja_ponizej": float(tercyle[0]),
        "krotka_regeneracja_powyzej": float(tercyle[1]),
    }
    print(f"\nProgi kategoryzacji (tercyle rozkładu POZIOMU composite_wellness):")
    print(f"  Dolny tercyl: poziom < {thresholds['dluga_regeneracja_ponizej']:.3f}")
    print(f"  Górny tercyl: poziom > {thresholds['krotka_regeneracja_powyzej']:.3f}")
    print("  Są to progi wskaźnika wellness, bez walidacji jako kategorie czasu regeneracji.")

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "predict_delta": args.predict_delta,
        "level_col": args.level_col,
        "target_definition": (
            "Δ composite_wellness (t+1 minus t)" if args.predict_delta
            else "composite_wellness (poziom) na dzień t+1"
        ),
        "pipeline_version": metadata.get("pipeline_version", "legacy-unversioned"),
        "preprocessing": {key: metadata[key] for key in ["missing_policy", "load_policy", "load_feature_policy", "time_policy", "composite_definition"] if key in metadata},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"\nZapisano model + metadane do: {args.output}")
    print_outputs("PMData — trening modelu", Path(args.output).parent, [Path(args.output).name])


if __name__ == "__main__":
    main()
