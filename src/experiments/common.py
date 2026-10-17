"""Wspólne foldery i polskie podsumowania eksperymentów obu zbiorów."""

from pathlib import Path

from src.data.common import PROJECT_ROOT


ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
PMDATA_ARTIFACTS = ARTIFACTS_ROOT / "pmdata"
SOCCERMON_ARTIFACTS = ARTIFACTS_ROOT / "soccermon"

LABELS = {
    "validation": "Walidacja", "fold": "Podział", "model": "Model", "subset": "Podzbiór",
    "source": "Źródło obciążenia", "period": "Okres", "mode": "Wariant celu",
    "n": "Wiersze", "participants": "Osoby", "participant": "Osoba", "offset": "Dzień względem wysiłku",
    "pairs": "Pary epizodów", "feature": "Cecha", "mean_abs_shap": "Średnia |SHAP|",
    "kierunek_wplywu": "Kierunek wpływu", "MAE_player_mean": "MAE — średnia osób",
    "mae_macro": "MAE — średnia osób", "target_r2_macro": "R² celu — średnia osób",
    "next_day_r2_macro": "R² poziomu jutro — średnia osób",
    "event_change": "Zmiana po wysiłku", "control_change": "Zmiana w kontroli",
    "contrast": "Różnica zmian", "event_load": "Obciążenie epizodu", "control_load": "Obciążenie kontroli",
    "ci_low": "Dolna granica 95%", "ci_high": "Górna granica 95%",
}
VALUES = {
    "group5": "5 podziałów według osób", "chronological": "Chronologiczna",
    "leave_one_player_out": "Jedna osoba w teście", "all": "Wszystkie dni", "intense": "Po intensywnym dniu",
    "published": "Opublikowane obciążenie", "sessions": "Suma sesji", "delta": "Zmiana", "level": "Poziom",
    "matched": "Dopasowane epizody", "isolated_next3": "Bez intensywnego dnia przez 3 dni",
    "persistence": "Jutro jak dziś", "train_mean": "Średnia celu z treningu",
    "ridge_today": "Ridge — dzisiejsze zmęczenie", "ridge_wellness": "Ridge — wellness",
    "ridge_wellness_load": "Ridge — wellness i obciążenie", "boosting_wellness": "Boosting — wellness",
    "boosting_wellness_load": "Boosting — wellness i obciążenie", "ridge_base": "Ridge — baza",
    "ridge_base_gps": "Ridge — baza i GPS", "boosting_base": "Boosting — baza",
    "boosting_base_gps": "Boosting — baza i GPS", "boosting_base_samples": "Boosting — baza i liczba próbek",
    "boosting_base_signals": "Boosting — baza i statystyki sygnałów",
}


def print_table(table):
    """Tłumaczy tabelę tylko do wyświetlenia; schematy zapisanych CSV pozostają stałe."""
    display = table.copy()
    for column in ["validation", "model", "subset", "source", "mode"]:
        if column in display:
            display[column] = display[column].map(lambda value: VALUES.get(value, value))
            if column == "validation":
                display[column] = display[column].str.replace("from_", "Od ", regex=False)
    print(display.rename(columns=LABELS).round(4).to_string(index=False))


def print_outputs(name, output_dir, filenames):
    """Wypisuje wspólne zakończenie eksperymentu i istniejące pliki wynikowe."""
    print(f"\n=== Zakończono eksperyment: {name} ===")
    print("Zapisane pliki:")
    for filename in filenames:
        print(f"  {(Path(output_dir) / filename).resolve()}")
