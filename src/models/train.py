"""
Trenowanie i porównanie modeli predykcji next-day wellness (PMData)
=====================================================================

Wczytuje ml_dataset.csv (wygenerowany przez przygotowanie_danych.py) i:
1. Usuwa wiersze z niekompletnymi cechami (początek serii uczestnika).
2. Waliduje modele metodą Leave-One-Participant-Out (LOGO) - każdy fold
   testuje na całym jednym uczestniku, trenując na pozostałych 7. To
   sprawdza, czy model generalizuje na nowego zawodnika, którego nie widział
   w treningu - znacznie bardziej wiarygodne niż losowy podział, bo kolejne
   dni tej samej osoby są mocno skorelowane (data leakage przy losowym CV).
3. Porównuje: baseline "persystencji" (jutro = dziś), regresję liniową,
   Ridge, Random Forest, Gradient Boosting, SVR, XGBoost, LightGBM.
4. Drukuje tabelę porównawczą (MAE, RMSE, R2, uśrednione po foldach).
5. Pokazuje istotność cech (feature importance) dla Random Forest.

Użycie:
    python train.py --dataset ml_dataset.csv
"""

import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


class ZeroChangeBaseline(BaseEstimator, RegressorMixin):
    """
    Bazowe podejście heurystyczne dla targetu typu DELTA (zmiana): zakłada
    brak zmiany samopoczucia z dnia na dzień (przewidywanie = 0). To jest
    poprawny odpowiednik "jutro jak dziś" w przestrzeni zmiany, w
    przeciwieństwie do PersistenceBaseline (który przewiduje surowy poziom
    "dziś" - sensowny tylko, gdy target sam jest poziomem, nie deltą).
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """
    Bazowe podejście heurystyczne (bez uczenia maszynowego): przewiduje, że
    samopoczucie jutro będzie takie samo jak dziś.
    """

    def __init__(self, today_col: str):
        self.today_col = today_col

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.today_col].to_numpy()


def load_dataset(path: str, target_col_name: str = "composite_wellness") -> tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    df = pd.read_csv(path)

    non_feature_cols = {"participant", "date", "target"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    n_before = len(df)
    df = df.dropna(subset=feature_cols + ["target"])
    n_after = len(df)
    print(f"Usunięto {n_before - n_after} wierszy z brakami w cechach/targecie.")
    print(f"Finalny zbiór: {n_after} wierszy, {df['participant'].nunique()} uczestników.\n")

    X = df[feature_cols].reset_index(drop=True)
    y = df["target"].reset_index(drop=True)
    groups = df["participant"].reset_index(drop=True)

    today_col = target_col_name if target_col_name in feature_cols else feature_cols[-1]

    return X, y, groups, today_col


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    logo = LeaveOneGroupOut()
    maes, rmses, r2s = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

    return {
        "MAE_mean": np.mean(maes),
        "MAE_std": np.std(maes),
        "RMSE_mean": np.mean(rmses),
        "RMSE_std": np.std(rmses),
        "R2_mean": np.mean(r2s),
        "R2_std": np.std(r2s),
    }


def print_comparison_table(results: dict) -> None:
    print("\n=== Porównanie modeli (walidacja Leave-One-Participant-Out) ===\n")
    header = f"{'Model':<35} | {'MAE':>14} | {'RMSE':>14} | {'R2':>14}"
    print(header)
    print("-" * len(header))

    for name, r in sorted(results.items(), key=lambda kv: kv[1]["MAE_mean"]):
        print(
            f"{name:<35} | {r['MAE_mean']:>6.3f} ± {r['MAE_std']:<5.3f} | "
            f"{r['RMSE_mean']:>6.3f} ± {r['RMSE_std']:<5.3f} | "
            f"{r['R2_mean']:>6.3f} ± {r['R2_std']:<5.3f}"
        )

    best_model = min(results.items(), key=lambda kv: kv[1]["MAE_mean"])
    baseline_key = next((k for k in results if "baseline" in k.lower()), None)
    baseline_mae = results.get(baseline_key, {}).get("MAE_mean") if baseline_key else None
    print(f"\nNajlepszy model wg MAE: {best_model[0]}")
    if baseline_mae is not None and best_model[0] != baseline_key:
        improvement = (1 - best_model[1]["MAE_mean"] / baseline_mae) * 100
        print(
            f"Poprawa względem baseline'u heurystycznego: {improvement:.1f}% "
            f"({'spełnia' if improvement > 0 else 'NIE spełnia'} wymaganie WNF-02)"
        )


def print_feature_importance(X: pd.DataFrame, y: pd.Series, top_n: int = 10) -> None:
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    print(f"\n=== Istotność cech (Random Forest, top {top_n}) ===\n")
    print(importances.head(top_n).round(4).to_string())
    print(
        "\nUwaga: model trenowany na całym zbiorze wyłącznie do celów interpretacji "
        "(feature importance) - do oceny skuteczności używaj wyników z walidacji LOGO powyżej."
    )


def main():
    parser = argparse.ArgumentParser(description="Trenowanie modeli predykcji next-day wellness")
    parser.add_argument("--dataset", default="ml_dataset.csv", help="Ścieżka do ml_dataset.csv")
    parser.add_argument(
        "--target-col-name",
        default="composite_wellness",
        help="Nazwa kolumny reprezentującej dzisiejszy stan (musi zgadzać się z --target-col z data_preparation.py)",
    )
    parser.add_argument(
        "--predict-delta",
        action="store_true",
        help="Ustaw, jeśli --dataset zawiera target typu DELTA (zmiana) - użyje poprawnego baseline'u (0), a nie persystencji poziomu",
    )
    args = parser.parse_args()

    X, y, groups, today_col = load_dataset(args.dataset, args.target_col_name)
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Cechy: {list(X.columns)}\n")

    baseline_model = ZeroChangeBaseline() if args.predict_delta else PersistenceBaseline(today_col=today_col)
    baseline_name = "Brak zmiany (baseline)" if args.predict_delta else "Persystencja (baseline)"

    models = {
        baseline_name: baseline_model,
        "Regresja liniowa": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Ridge (mocniejsza regularyzacja)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest (domyślny)": RandomForestRegressor(n_estimators=300, random_state=42),
        "Random Forest (regularyzowany)": RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=15,
            max_features=0.5,
            random_state=42,
        ),
        "Gradient Boosting (domyślny)": GradientBoostingRegressor(random_state=42),
        "Gradient Boosting (regularyzowany)": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=15,
            random_state=42,
        ),
        "SVR": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)),
        "XGBoost (domyślny)": XGBRegressor(random_state=42, verbosity=0),
        "XGBoost (regularyzowany)": XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=15,
            random_state=42,
            verbosity=0,
        ),
        "LightGBM (domyślny)": LGBMRegressor(random_state=42, verbose=-1),
        "LightGBM (regularyzowany)": LGBMRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=15,
            random_state=42,
            verbose=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Trenuję: {name}...")
        results[name] = evaluate_model(model, X, y, groups)

    print_comparison_table(results)
    print_feature_importance(X, y)


if __name__ == "__main__":
    main()