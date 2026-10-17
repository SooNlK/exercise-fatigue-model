"""Przygotowanie dziennych podsumowań sygnałów GPS (SoccerMon)
============================================================

1. Wyszukuje pliki Parquet na dysku źródłowym.
2. Wybiera osoby występujące w wellness i dni należące do przygotowanej próby.
3. Oblicza statystyki prędkości, impulsu i liczby próbek; sprawdza dostępność
   tętna, ale nie używa go jako cechy modelu.
4. Wykorzystuje pamięć podręczną według ścieżki, rozmiaru i czasu modyfikacji.
5. Zapisuje gps_daily.csv oraz gps_audit.json do
   data/processed/soccermon/objective/.

Użycie: python -m src.data.soccermon.objective --objective-dir /sciezka/do/parquet --subjective-dir /sciezka/do/soccermon
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import SOCCERMON_DIR, SOCCERMON_OBJECTIVE_DIR, PolishArgumentParser, print_summary

GPS_FEATURES = ["gps_speed_mean", "gps_speed_std", "gps_speed_p95", "gps_impulse_mean", "gps_impulse_p95", "gps_sample_count"]


def finite_nonnegative(values):
    """Wybiera skończone i nieujemne próbki sygnału."""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values >= 0)]


def summarize(speed, impulse, sample_count):
    """Oblicza dzienne statystyki prędkości i impulsu w jednostkach źródła."""
    s, a = finite_nonnegative(speed), finite_nonnegative(impulse)
    return {"gps_speed_mean": float(s.mean()) if len(s) else np.nan,
            "gps_speed_std": float(s.std()) if len(s) else np.nan,
            "gps_speed_p95": float(np.quantile(s, .95)) if len(s) else np.nan,
            "gps_impulse_mean": float(a.mean()) if len(a) else np.nan,
            "gps_impulse_p95": float(np.quantile(a, .95)) if len(a) else np.nan,
            "gps_sample_count": int(sample_count), "valid_speed_samples": len(s), "valid_impulse_samples": len(a)}


def extract(root, subjective_root, output, cohort_path=None):
    """Zapisuje podsumowania GPS i audyt, korzystając z dostępnej pamięci podręcznej."""
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    wellness = pd.read_csv(subjective_root / "wellness" / "fatigue.csv")
    people = set(wellness.columns[1:])
    files = sorted(root.rglob("*.parquet"))
    selected = [p for p in files if p.name[11:-8] in people]
    matching_files = len(selected)
    if cohort_path is not None:
        cohort = pd.read_csv(cohort_path, usecols=["participant", "date"])
        keys = set(zip(cohort.participant, cohort.date))
        selected = [p for p in selected if (p.name[11:-8], p.name[:10]) in keys]
    if not selected:
        raise ValueError("Nie znaleziono plików Parquet dla wybranych osób i dni.")
    csv = output / "gps_daily.csv"
    cache = {}
    if csv.exists():
        for row in pd.read_csv(csv).to_dict("records"):
            cache[row["file"]] = row
    rows = []
    schemas = {}
    for index, path in enumerate(selected, 1):
        relative = str(path.relative_to(root))
        stat = path.stat()
        previous = cache.get(relative)
        if previous and previous["bytes"] == stat.st_size and previous["mtime_ns"] == stat.st_mtime_ns:
            rows.append(previous)
            continue
        pf = pq.ParquetFile(path)
        schema = str(pf.schema_arrow)
        schemas[schema] = schemas.get(schema, 0) + 1
        required = ["speed", "inst_acc_impulse"]
        if not set(required).issubset(pf.schema_arrow.names):
            raise ValueError(f"Brak wymaganych kolumn: {path}")
        names = required + (["heart_rate"] if "heart_rate" in pf.schema_arrow.names else [])
        table = pf.read(columns=names)
        row = {"participant": path.name[11:-8], "date": path.name[:10],
               "file": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
               **summarize(table["speed"].to_numpy(), table["inst_acc_impulse"].to_numpy(), pf.metadata.num_rows)}
        if "heart_rate" in names:
            hr = table["heart_rate"].to_numpy().astype(float)
            row["nonzero_heart_rate_samples"] = int((np.isfinite(hr) & (hr > 0)).sum())
        else:
            row["nonzero_heart_rate_samples"] = 0
        rows.append(row)
        if index % 25 == 0:
            pd.DataFrame(rows).to_csv(csv, index=False)
            print(f"Podsumowano pliki Parquet: {index}/{len(selected)}", flush=True)
    result = pd.DataFrame(rows)
    if result.duplicated(["participant", "date"]).any():
        raise ValueError("Wiele plików dla tej samej osoby i dnia: wymagana jawna agregacja sesji")
    result.to_csv(csv, index=False)
    audit = {"all_files": len(files), "files_matching_wellness_ids": matching_files,
             "files_selected_for_observed_cohort": len(selected),
             "all_objective_players": len({p.name[11:-8] for p in files}),
             "matching_players": int(result.participant.nunique()),
             "total_source_bytes": sum(p.stat().st_size for p in files),
             "schemas_read_this_run": schemas, "cache_validation": "Ścieżka względna, rozmiar i czas modyfikacji w nanosekundach; bez sumy kontrolnej treści",
             "features": GPS_FEATURES, "units": "Oryginalne jednostki źródłowe; bez zakładania przelicznika prędkości lub impulsu",
             "heart_rate_used_for_prediction": False, "raw_files_modified": False}
    (output / "gps_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary("SoccerMon — GPS", len(result), result.participant.nunique(), {
        "Podsumowania GPS": csv, "Audyt GPS": output / "gps_audit.json",
    })
    print(f"Znalezione pliki Parquet: {len(files)}")
    print(f"Pliki dopasowane do identyfikatorów wellness: {matching_files}")
    return result


def main():
    parser = PolishArgumentParser(description=__doc__)
    parser.add_argument("--objective-dir", type=Path, required=True, help="Folder źródłowy z plikami Parquet")
    parser.add_argument("--subjective-dir", type=Path, required=True, help="Folder źródłowy SoccerMon z raportami wellness.")
    parser.add_argument("--output-dir", type=Path, default=SOCCERMON_OBJECTIVE_DIR, help="Folder wynikowy (domyślnie: data/processed/soccermon/objective).")
    parser.add_argument("--cohort", type=Path, default=SOCCERMON_DIR / "dataset.csv", help="CSV wyznaczający osoby i dni do przetworzenia.")
    args = parser.parse_args()
    extract(args.objective_dir, args.subjective_dir, args.output_dir, args.cohort)


if __name__ == "__main__":
    main()
