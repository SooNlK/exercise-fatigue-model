"""Wspólne ścieżki i polskie komunikaty skryptów przygotowania danych."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
PMDATA_DIR = PROCESSED_ROOT / "pmdata"
SOCCERMON_DIR = PROCESSED_ROOT / "soccermon"
SOCCERMON_OBJECTIVE_DIR = SOCCERMON_DIR / "objective"


class PolishArgumentParser(argparse.ArgumentParser):
    """Wyświetla wspólną pomoc i komunikaty błędów po polsku."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, add_help=False, **kwargs)
        self._positionals.title = "Argumenty pozycyjne"
        self._optionals.title = "Opcje"
        self.add_argument("-h", "--help", action="help", help="Wyświetl pomoc i zakończ.")

    def format_usage(self):
        return super().format_usage().replace("usage: ", "Użycie: ", 1)

    def format_help(self):
        return super().format_help().replace("usage: ", "Użycie: ", 1)

    def error(self, message):
        message = message.replace("the following arguments are required:", "Brakuje wymaganych argumentów:")
        message = message.replace("not allowed with argument", "nie można łączyć z argumentem")
        message = message.replace("unrecognized arguments:", "Nieznane argumenty:")
        self.print_usage(sys.stderr)
        self.exit(2, f"Błąd: {message}\n")


def print_summary(name, rows, participants, paths):
    """Podaje liczebność zbioru i lokalizacje zapisanych plików."""
    print(f"\n=== Przygotowanie danych: {name} ===")
    print(f"Liczba wierszy: {rows}")
    print(f"Liczba osób: {participants}")
    print("Zapisane pliki:")
    for label, path in paths.items():
        print(f"  {label}: {Path(path).resolve()}")
