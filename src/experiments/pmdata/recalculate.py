"""Pełna ponowna ocena prognoz następnego dnia (PMData)
=====================================================

1. Wczytuje wariant poziomu i zmiany z data/processed/pmdata/.
2. Porównuje stałe modele i proste punkty odniesienia na tej samej próbie
   kompletnych obserwacji, za każdym razem wyłączając jedną osobę z uczenia.
3. Dla Gradient Boostingu dobiera parametry w wewnętrznej walidacji grupowej.
4. Rozróżnia R² zmiany i R² zrekonstruowanego jutrzejszego poziomu.
5. Zapisuje predykcje, metryki per osoba, podsumowanie i metadane do
   artifacts/pmdata/evaluation/.

Ranking kandydatów jest eksploracyjny; zagnieżdżona walidacja dotyczy tylko
strojenia Gradient Boostingu. Skrypt nie aktualizuje modelu API.
Użycie: python -m src.experiments.pmdata.recalculate
"""


import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline

from src.data.common import PMDATA_DIR, PolishArgumentParser
from src.experiments.common import PMDATA_ARTIFACTS, print_outputs, print_table, VALUES
from src.data.pmdata.preparation import read_dataset_metadata, resolve_target_mode
from src.experiments.pmdata.compare import build_models
from src.experiments.pmdata.nested_cv import FIXED_PARAMS, PARAM_GRID, select_best_params


def prediction_metrics(y_true, y_pred, today, predict_delta):
    """Oblicza metryki tych samych prognoz w skali celu i jutrzejszego poziomu."""
    level_true = np.asarray(y_true) + np.asarray(today) if predict_delta else np.asarray(y_true)
    level_pred = np.asarray(y_pred) + np.asarray(today) if predict_delta else np.asarray(y_pred)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'target_r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        'next_day_r2': float(r2_score(level_true, level_pred)) if len(y_true) > 1 else np.nan,
    }


def held_out_predictions(model, X, y, groups, *, tune=False):
    """Prognozuje każdy wiersz raz, wyłączając całą daną osobę z uczenia."""
    predictions = np.full(len(y), np.nan)
    choices = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        estimator = clone(model)
        participant = str(groups.iloc[test_idx].iloc[0])
        if tune:
            params, inner_mae = select_best_params(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx])
            estimator.set_params(**params)
            choices.append({'participant': participant, 'inner_mae': float(inner_mae), **params})
            print(f'  Walidacja zagnieżdżona: osoba testowa={participant}, parametry={params}', flush=True)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = estimator.predict(X.iloc[test_idx])
    if not np.isfinite(predictions).all():
        raise ValueError('Brakujące lub nieskończone predykcje poza próbą treningową')
    return predictions, choices


def run(dataset_dir, output_dir, nested=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, fold_metrics, all_predictions, tuning, manifests = [], [], [], [], []
    cohort = None
    for mode, filename in [('delta', 'ml_dataset_delta.csv'), ('level', 'ml_dataset.csv')]:
        path = dataset_dir / filename
        metadata = read_dataset_metadata(path)
        is_delta = resolve_target_mode(metadata, mode == 'delta')
        df_raw = pd.read_csv(path)
        features = [c for c in df_raw.columns if c not in {'participant', 'date', 'target'}]
        if 'composite_wellness' not in features:
            raise ValueError('Brak zaobserwowanego bieżącego composite_wellness')
        df = df_raw.dropna(subset=features + ['target']).reset_index(drop=True)
        if df.empty or not np.isfinite(df[features + ['target']].to_numpy()).all():
            raise ValueError('Pusty zbiór kompletnych obserwacji lub wartości nieskończone')
        keys = df[['participant', 'date']]
        if keys.duplicated().any():
            raise ValueError('Powtarzająca się para osoba–data')
        if cohort is not None and not cohort.equals(keys):
            raise ValueError('Wariant poziomu i zmiany musi korzystać z tej samej próby i kolejności wierszy')
        cohort = keys
        X, y, groups = df[features], df['target'], df['participant']
        today = X['composite_wellness'].to_numpy()
        manifests.append({
            'mode': mode, 'dataset': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'metadata_sha256': hashlib.sha256(path.with_suffix('.csv.meta.json').read_bytes()).hexdigest(),
            'pipeline_version': metadata.get('pipeline_version'),
            'rows_with_target': len(df_raw), 'complete_rows': len(df),
            'participants': int(groups.nunique()), 'features': features,
            'participant_counts': groups.value_counts().sort_index().to_dict(),
        })
        print(f'{VALUES[mode]}: {len(df)} wierszy, {groups.nunique()} osób', flush=True)
        models = build_models(is_delta, 'composite_wellness')
        models['Regresja tylko dzisiejszego wellness'] = make_pipeline(
            ColumnTransformer([('today', 'passthrough', ['composite_wellness'])], remainder='drop'),
            LinearRegression(),
        )
        models['Średnia celu z treningu (baseline)'] = DummyRegressor(strategy='mean')
        if nested and is_delta:
            models['Gradient Boosting (nested CV)'] = GradientBoostingRegressor(**FIXED_PARAMS)
        recipes = {}
        for name, model in models.items():
            print(f'{VALUES[mode]}: {name}', flush=True)
            recipes[name] = repr(model)
            pred, choices = held_out_predictions(model, X, y, groups, tune='nested CV' in name)
            tuning.extend({'mode': mode, 'model': name, **choice} for choice in choices)
            pred_df = keys.copy()
            pred_df['target_date'] = (pd.to_datetime(df['date']) + pd.Timedelta(days=metadata.get('horizon', 1))).astype(str)
            pred_df['mode'], pred_df['model'] = mode, name
            pred_df['today'], pred_df['target_true'], pred_df['target_pred'] = today, y.to_numpy(), pred
            pred_df['next_day_true'] = y.to_numpy() + today if is_delta else y.to_numpy()
            pred_df['next_day_pred'] = pred + today if is_delta else pred
            all_predictions.append(pred_df)
            rows = []
            for participant in sorted(groups.unique()):
                mask = (groups == participant).to_numpy()
                row = {'mode': mode, 'model': name, 'participant': participant, 'n': int(mask.sum()),
                       **prediction_metrics(y[mask], pred[mask], today[mask], is_delta)}
                rows.append(row)
                fold_metrics.append(row)
            per_person = pd.DataFrame(rows)
            summary = {'mode': mode, 'model': name, 'n': len(y), 'participants': len(rows)}
            for metric in ['mae', 'rmse', 'target_r2', 'next_day_r2']:
                summary[f'{metric}_macro'] = float(per_person[metric].mean())
                summary[f'{metric}_std'] = float(per_person[metric].std(ddof=0))
            summary.update({f'{k}_pooled': v for k, v in prediction_metrics(y, pred, today, is_delta).items()})
            summaries.append(summary)
            # Zapis przyrostowy chroni wcześniejsze wyniki podczas długiego strojenia.
            pd.DataFrame(summaries).to_csv(output_dir / 'summary.csv', index=False)
            pd.DataFrame(fold_metrics).to_csv(output_dir / 'per_participant.csv', index=False)
            pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / 'predictions.csv', index=False)
        manifests[-1]['model_recipes'] = recipes
    pd.DataFrame(tuning).to_csv(output_dir / 'nested_choices.csv', index=False)
    manifest = {'datasets': manifests, 'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__,
                'validation': 'Wyłączenie jednej osoby; kompletne obserwacje; średnie macro z równymi wagami osób',
                'selection_warning': 'Ranking jest eksploracyjny, bez niezależnego testu końcowego. Zagnieżdżona walidacja stroi tylko Gradient Boosting.',
                'nested_grid': PARAM_GRID if nested else None, 'nested_fixed_params': FIXED_PARAMS if nested else None}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print_table(pd.DataFrame(summaries)[['mode', 'model', 'mae_macro', 'target_r2_macro', 'next_day_r2_macro']])
    print_outputs("PMData — pełna ponowna ocena", output_dir,
                  ['summary.csv', 'per_participant.csv', 'predictions.csv', 'nested_choices.csv', 'manifest.json'])


def main():
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=PMDATA_DIR, help='Folder z wariantami poziomu i zmiany PMData.')
    parser.add_argument('--output-dir', type=Path, default=PMDATA_ARTIFACTS / "evaluation", help='Folder plików wynikowych; domyślnie podfolder danego eksperymentu w artifacts/.')
    parser.add_argument('--skip-nested', action='store_true', help='Pomiń zagnieżdżone strojenie Gradient Boostingu.')
    args = parser.parse_args()
    run(args.dataset_dir, args.output_dir, nested=not args.skip_nested)


if __name__ == '__main__':
    main()
