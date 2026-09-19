"""
Trenowanie i zapis finalnego modelu produkcyjnego
====================================================

Obsługuje dwa tryby (wybór na podstawie eksperymentu porównawczego, w którym
target typu DELTA wypadł zdecydowanie lepiej - R2=0.37 vs R2=0.01 dla poziomu
bezwzględnego):

1. --predict-delta (ZALECANY, domyślny): model przewiduje zmianę
   samopoczucia (Δwellness = composite_wellness(t+1) - composite_wellness(t)).
   Do kategoryzacji (WF-04) rekonstruujemy przewidywany poziom jutrzejszy jako
   dzisiejszy_poziom + przewidziana_delta, i dopiero względem niego liczy
   progi krótka/umiarkowana/długa regeneracja.
2. bez --predict-delta: model przewiduje poziom bezwzględny wprost (starszy,
   gorszy wariant - zachowany dla porównania/reprodukowalności eksperymentu).

Użycie:
    python train_final.py --dataset ml_dataset_delta.csv --predict-delta --output model.joblib
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def main():
    parser = argparse.ArgumentParser(description="Trenowanie finalnego modelu produkcyjnego")
    parser.add_argument("--dataset", default="ml_dataset_delta.csv")
    parser.add_argument("--output", default="model.joblib")
    parser.add_argument(
        "--predict-delta",
        action="store_true",
        default=True,
        help="Trenuj na targecie typu delta (domyślnie WŁĄCZONE - to zwycięski wariant z eksperymentów)",
    )
    parser.add_argument("--no-predict-delta", dest="predict_delta", action="store_false")
    parser.add_argument(
        "--level-col",
        default="composite_wellness",
        help="Nazwa kolumny reprezentującej dzisiejszy poziom (do rekonstrukcji jutrzejszego poziomu z delty)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    non_feature_cols = {"participant", "date", "target"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    n_before = len(df)
    df = df.dropna(subset=feature_cols + ["target"])
    print(f"Usunięto {n_before - len(df)} wierszy z brakami. Trenuję na {len(df)} wierszach.")

    X = df[feature_cols]
    y = df["target"]

    if args.predict_delta:
        # zwycięski wariant z porównania modeli: Gradient Boosting regularyzowany
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
    print(f"  poziom < {thresholds['dluga_regeneracja_ponizej']:.3f}  -> DŁUGA regeneracja potrzebna")
    print(f"  poziom > {thresholds['krotka_regeneracja_powyzej']:.3f}  -> KRÓTKA regeneracja / gotowy")
    print(f"  pomiędzy                                        -> UMIARKOWANA regeneracja")

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
    }
    joblib.dump(artifact, args.output)
    print(f"\nZapisano model + metadane do: {args.output}")


if __name__ == "__main__":
    main()