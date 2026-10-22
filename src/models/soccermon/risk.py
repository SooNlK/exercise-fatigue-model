"""Metody prognozowania ryzyka pogorszenia zmęczenia SoccerMon.

1. Definiuje zdarzenie: jutrzejsze fatigue niższe o co najmniej jeden punkt.
2. Dopasowuje częstości osobnicze według fatigue i intensywności dnia,
   ze ściąganiem do częstości populacyjnych przez 20 pseudoobserwacji.
3. Tworzy klasyfikator boosting z parametrami użytymi w eksperymencie.
4. Zapewnia wspólne prawdopodobieństwa, w tym zero dla fatigue=1.

Częstości są dopasowywane wyłącznie na treningu. Historia przekazana
do prognozy nie aktualizuje automatycznie wyuczonego modelu osobniczego.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PIPELINE_VERSION = "soccermon-deterioration-v1"
TARGET_DEFINITION = "fatigue_next_day <= fatigue_today - 1"
PRIOR_STRENGTH = 20.0
BOOSTING_PARAMS = dict(max_iter=100, max_leaf_nodes=8, min_samples_leaf=30,
                       l2_regularization=10, early_stopping=False, random_state=42)


def event_labels(data):
    """Zwraca zaobserwowane pogorszenie; brak celu nie jest klasą ujemną."""
    for name in ["fatigue", "target"]:
        if not data[name].isin([1, 2, 3, 4, 5]).all():
            raise ValueError(f"Wymagane zaobserwowane {name} w skali całkowitej 1–5.")
    return (data.target <= data.fatigue - 1).astype(int)


class PersonalIntensityRisk:
    """Stałe częstości z treningu; nieznana osoba korzysta z populacji."""

    def fit(self, data):
        if data.empty or data.participant.isna().any():
            raise ValueError("Wymagany niepusty trening z identyfikatorami osób.")
        if not data.intense_day.isin([True, False]).all():
            raise ValueError("Wymagany logiczny znacznik intense_day.")
        frame = data[["participant", "fatigue", "intense_day"]].copy()
        frame["event"] = event_labels(data).to_numpy()
        self.global_mean_ = (frame.event.sum() + 1) / (len(frame) + 2)
        strength = PRIOR_STRENGTH
        today = frame.groupby("fatigue").event.agg(["sum", "count"])
        self.today_ = (today["sum"] + strength * self.global_mean_) / (today["count"] + strength)
        strata = frame.groupby(["fatigue", "intense_day"]).event.agg(["sum", "count"])
        prior = self.today_.reindex(strata.index.get_level_values("fatigue")).to_numpy()
        self.intensity_ = pd.Series(
            (strata["sum"].to_numpy() + strength * prior) / (strata["count"].to_numpy() + strength),
            index=strata.index,
        )
        personal = frame.groupby(["participant", "fatigue", "intense_day"]).event.agg(["sum", "count"])
        prior = self.intensity_.reindex(personal.index.droplevel("participant")).to_numpy()
        self.personal_ = pd.Series(
            (personal["sum"].to_numpy() + strength * prior) / (personal["count"].to_numpy() + strength),
            index=personal.index,
        )
        return self

    def predict_with_source(self, data):
        """Zwraca ryzyko oraz poziom częstości faktycznie użyty dla wiersza."""
        today = data.fatigue.map(self.today_).fillna(self.global_mean_).to_numpy(dtype=float)
        intensity = self.intensity_.reindex(pd.MultiIndex.from_frame(data[["fatigue", "intense_day"]])).to_numpy()
        personal = self.personal_.reindex(pd.MultiIndex.from_frame(data[["participant", "fatigue", "intense_day"]])).to_numpy()
        probability = np.where(np.isnan(personal), np.where(np.isnan(intensity), today, intensity), personal)
        source = np.where(~np.isnan(personal), "personal_intensity",
                          np.where(~np.isnan(intensity), "population_intensity", "population_today"))
        source = np.where(data.fatigue.eq(1), "scale_floor", source)
        return np.where(data.fatigue.eq(1), 0.0, probability), source


def make_boosting():
    """Tworzy model bez strojenia, zgodny z głównym eksperymentem."""
    return HistGradientBoostingClassifier(**BOOSTING_PARAMS)


def risk_probabilities(artifact, rows):
    """Wspólna obsługa obu metod na już przygotowanych wierszach."""
    if artifact["method"] == "personal_intensity_frequency":
        probability, source = artifact["model"].predict_with_source(rows)
    elif artifact["method"] == "boosting_load":
        model = artifact["model"]
        index = list(model.classes_).index(1)
        probability = model.predict_proba(rows[artifact["feature_cols"]])[:, index]
        source = np.full(len(rows), "boosting", dtype=object)
        source[rows.fatigue.eq(1).to_numpy()] = "scale_floor"
    else:
        raise ValueError("Nieznana metoda prognozowania ryzyka.")
    probability = np.where(rows.fatigue.eq(1), 0.0, probability)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Model zwrócił nieprawidłowe prawdopodobieństwo.")
    return probability, source
