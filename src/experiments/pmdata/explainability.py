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
    python -m src.experiments.pmdata.explainability --model model.joblib --dataset ml_dataset.csv

Wyniki: wykres i ranking CSV w artifacts/pmdata/explainability/.
Wartości SHAP opisują model i nie dowodzą przyczynowego wpływu cech.
"""

from pathlib import Path


import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # zapis do pliku, bez okna GUI
import matplotlib.pyplot as plt

from src.data.common import PMDATA_DIR, PolishArgumentParser
from src.experiments.common import PMDATA_ARTIFACTS, print_outputs, print_table


def main():
    parser = PolishArgumentParser(description="Analiza SHAP wytrenowanego modelu")
    parser.add_argument("--model", default=str(PMDATA_ARTIFACTS / "model.joblib"), help='Ścieżka modelu wraz z metadanymi.')
    parser.add_argument("--dataset", default=str(PMDATA_DIR / "ml_dataset_delta.csv"), help='Ścieżka przygotowanego zbioru CSV.')
    parser.add_argument("--output-plot", default=str(PMDATA_ARTIFACTS / "explainability" / "shap_summary.png"), help='Ścieżka zapisu wykresu SHAP.')
    parser.add_argument("--output-importance", default=str(PMDATA_ARTIFACTS / "explainability" / "shap_importance.csv"), help='Ścieżka zapisu rankingu cech CSV.')
    parser.add_argument("--top-n", type=int, default=15, help='Liczba cech prezentowanych w rankingu (domyślnie: 15).')
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
        "im wyżej, tym wyższy target" if directions[feature_cols.index(f)] > 0.05
        else "im wyżej, tym niższy target" if directions[feature_cols.index(f)] < -0.05
        else "brak jasnego kierunku"
        for f in importance_df["feature"]
    ]

    print(f"\n=== Ranking cech wg istotności SHAP (top {args.top_n}) ===\n")
    print_table(importance_df.head(args.top_n))

    # zapis wykresu podsumowującego
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=args.top_n)
    plt.xlabel("Wartość SHAP (wpływ na wynik modelu)")
    plt.title("Wpływ cech na prognozę modelu")
    for axis in plt.gcf().axes:
        if axis.get_ylabel() == "Feature value":
            axis.set_ylabel("Wartość cechy")
            axis.set_yticklabels([label.get_text().replace("Low", "Niska").replace("High", "Wysoka") for label in axis.get_yticklabels()])
    plt.tight_layout()
    Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
    print(f"\nZapisano wykres SHAP summary do: {args.output_plot}")

    importance_csv = args.output_importance
    Path(importance_csv).parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_csv, index=False)
    print(f"Zapisano pełną tabelę rankingu do: {importance_csv}")
    plt.close("all")
    print_outputs("PMData — interpretacja SHAP", Path(args.output_plot).parent, [Path(args.output_plot).name])
    print(f"  {Path(importance_csv).resolve()}")


if __name__ == "__main__":
    main()
