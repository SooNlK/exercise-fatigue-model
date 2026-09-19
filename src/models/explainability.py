"""
Analiza interpretowalności modelu metodą SHAP
================================================

Wczytuje wytrenowany model (model.joblib) oraz zbiór danych, i liczy wartości
SHAP (SHapley Additive exPlanations) dla każdej cechy - w przeciwieństwie do
zwykłego feature_importances_ (który mówi tylko "jak bardzo cecha jest
ważna"), SHAP pokazuje też KIERUNEK wpływu (np. czy wysokie rhr_relative_lag1
podnosi, czy obniża przewidywane samopoczucie), co ma bezpośrednią wartość
aplikacyjną/sportową, a nie tylko techniczną.

Użycie:
    python explainability.py --model model.joblib --dataset ml_dataset.csv
"""

import argparse

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # zapis do pliku, bez okna GUI
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Analiza SHAP wytrenowanego modelu")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--dataset", default="ml_dataset.csv")
    parser.add_argument("--output-plot", default="shap_summary.png")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    try:
        import shap
    except ImportError:
        raise SystemExit(
            "Brak pakietu 'shap'. Zainstaluj: pip install shap"
        )

    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    df = pd.read_csv(args.dataset)
    df = df.dropna(subset=feature_cols + ["target"])
    X = df[feature_cols]

    print(f"Liczę SHAP dla {len(X)} obserwacji, {len(feature_cols)} cech...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ranking cech wg średniej |SHAP| (analog feature_importances_, ale spójny z SHAP)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # kierunek: korelacja między wartością cechy a jej wpływem SHAP
    # dodatnia korelacja = "im wyższa wartość cechy, tym wyższa predykcja"
    directions = []
    for i, col in enumerate(feature_cols):
        corr = np.corrcoef(X[col].to_numpy(), shap_values[:, i])[0, 1]
        directions.append(corr)
    importance_df["kierunek_wplywu"] = [
        "im wyżej, tym WYŻSZY target" if directions[feature_cols.index(f)] > 0.05
        else "im wyżej, tym NIŻSZY target" if directions[feature_cols.index(f)] < -0.05
        else "brak jasnego kierunku"
        for f in importance_df["feature"]
    ]

    print(f"\n=== Ranking cech wg istotności SHAP (top {args.top_n}) ===\n")
    print(importance_df.head(args.top_n).to_string(index=False))

    # zapis wykresu podsumowującego
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=args.top_n)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
    print(f"\nZapisano wykres SHAP summary do: {args.output_plot}")

    importance_csv = "shap_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"Zapisano pełną tabelę rankingu do: {importance_csv}")


if __name__ == "__main__":
    main()