"""
Zagnieżdżona walidacja krzyżowa (nested CV) z lekkim grid search dla Gradient Boosting
=========================================================================================

Struktura:
- Zewnętrzna pętla: Leave-One-Participant-Out (12 foldów)
- Wewnętrzna pętla: dla każdego zewnętrznego foldu, mała siatka hiperparametrów
  (max_depth x learning_rate, 9 kombinacji) testowana metodą GroupKFold na
  pozostałych 11 uczestnikach - wybieramy kombinację z najniższym wewnętrznym MAE
- Finalna ocena: model z wybranymi hiperparametrami trenowany na całym
  zewnętrznym zbiorze treningowym, oceniany na niewidzianym uczestniku

Dodatkowo: raportujemy, jak często każda kombinacja hiperparametrów "wygrywa"
w poszczególnych foldach - jeśli wybór jest niestabilny (rozjeżdża się między
foldami), to sam w sobie ważny wniosek o tym, że przy tej wielkości próby
strojenie hiperparametrów nie daje wiarygodnego, jednoznacznego rozstrzygnięcia.

Użycie:
    python nested_cv_gb.py --dataset ml_dataset_delta.csv
"""

import argparse
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

PARAM_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
}
FIXED_PARAMS = dict(n_estimators=100, subsample=0.8, min_samples_leaf=15, random_state=42)


def load_dataset(path: str):
    df = pd.read_csv(path)
    non_feature_cols = {"participant", "date", "target"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    n_before = len(df)
    df = df.dropna(subset=feature_cols + ["target"])
    print(f"Usunięto {n_before - len(df)} wierszy z brakami. Zbiór: {len(df)} wierszy, {df['participant'].nunique()} uczestników.\n")

    X = df[feature_cols].reset_index(drop=True)
    y = df["target"].reset_index(drop=True)
    groups = df["participant"].reset_index(drop=True)
    return X, y, groups


def select_best_params(X_train, y_train, groups_train, n_inner_splits: int = 5):
    """Wewnętrzna walidacja krzyżowa - wybór hiperparametrów WYŁĄCZNIE na danych treningowych foldu."""
    n_splits = min(n_inner_splits, groups_train.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    best_mae = np.inf
    best_params = None

    for max_depth, lr in product(PARAM_GRID["max_depth"], PARAM_GRID["learning_rate"]):
        fold_maes = []
        for tr_idx, val_idx in gkf.split(X_train, y_train, groups_train):
            model = GradientBoostingRegressor(max_depth=max_depth, learning_rate=lr, **FIXED_PARAMS)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = model.predict(X_train.iloc[val_idx])
            fold_maes.append(mean_absolute_error(y_train.iloc[val_idx], pred))

        mean_mae = np.mean(fold_maes)
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_params = {"max_depth": max_depth, "learning_rate": lr}

    return best_params, best_mae


def main():
    parser = argparse.ArgumentParser(description="Nested CV z lekkim grid search dla Gradient Boosting")
    parser.add_argument("--dataset", default="ml_dataset_delta.csv")
    args = parser.parse_args()

    X, y, groups = load_dataset(args.dataset)
    print(f"Siatka hiperparametrów: max_depth={PARAM_GRID['max_depth']}, learning_rate={PARAM_GRID['learning_rate']}")
    print(f"({len(PARAM_GRID['max_depth']) * len(PARAM_GRID['learning_rate'])} kombinacji, testowanych osobno w każdym z {groups.nunique()} zewnętrznych foldów)\n")

    logo = LeaveOneGroupOut()
    outer_maes, outer_rmses, outer_r2s = [], [], []
    chosen_params_log = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        test_participant = groups.iloc[test_idx].iloc[0]

        best_params, inner_mae = select_best_params(X_train, y_train, groups_train)
        chosen_params_log.append(best_params)

        model = GradientBoostingRegressor(**best_params, **FIXED_PARAMS)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        outer_maes.append(mae)
        outer_rmses.append(rmse)
        outer_r2s.append(r2)

        print(
            f"Fold {fold_i:2d} (test={test_participant:>4}): "
            f"wybrane={best_params}, wewn.MAE={inner_mae:.3f}, "
            f"zewn.MAE={mae:.3f}, R2={r2:.3f}"
        )

    print("\n=== Wynik nested CV (Gradient Boosting + lekkie grid search) ===\n")
    print(f"MAE:  {np.mean(outer_maes):.3f} ± {np.std(outer_maes):.3f}")
    print(f"RMSE: {np.mean(outer_rmses):.3f} ± {np.std(outer_rmses):.3f}")
    print(f"R2:   {np.mean(outer_r2s):.3f} ± {np.std(outer_r2s):.3f}")

    print("\n=== Stabilność wyboru hiperparametrów między foldami ===")
    print("(ile razy każda kombinacja wygrała wewnętrzną walidację)\n")
    counts = Counter(tuple(sorted(p.items())) for p in chosen_params_log)
    for params, count in counts.most_common():
        print(f"  {dict(params)}: {count}/{len(chosen_params_log)} foldów")

    if len(counts) > len(chosen_params_log) / 2:
        print(
            "\nUwaga: wybór najlepszych hiperparametrów jest niestabilny (rozjeżdża się "
            "między foldami) - sugeruje to, że przy tej wielkości próby strojenie "
            "hiperparametrów nie daje jednoznacznego, wiarygodnego rozstrzygnięcia, "
            "a różnice między kombinacjami mieszczą się w granicach szumu estymacji."
        )
    else:
        print(
            "\nWybór hiperparametrów jest względnie stabilny między foldami - "
            "to dobry znak, że wybrana konfiguracja nie jest przypadkowa."
        )


if __name__ == "__main__":
    main()