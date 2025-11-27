"""Evaluate forecasting models and print metrics."""
import json
import sys
from pathlib import Path

from src.time_series import build_monthly_series_from_file
from src.evaluate import evaluate_models

DATA_FILE = "data/inscription.xlsx"


def main():
    p = Path(DATA_FILE)
    if not p.exists():
        print(f"File {DATA_FILE} not found", file=sys.stderr)
        return 2
    
    series = build_monthly_series_from_file(str(p))
    print('Series length:', len(series))
    
    results = evaluate_models(series, initial_arima=24)
    print('\nEvaluation results:')
    print(json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
