"""Runner to evaluate forecasting models (ARIMA & ETS) and print forecast metrics.

Usage examples:
  python run_evaluation.py --file inscription.xlsx
  python run_evaluation.py --synthetic

The script will load a file (Excel expected) and build monthly counts from
`Date inscription` (fallback to `Première venue`) or run on synthetic data.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.ts_utils import build_monthly_series_from_file
from src.models_eval import evaluate_models


def make_synthetic(n=120, seed=0):
    import numpy as np

    rng = np.random.RandomState(seed)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='MS')
    t = np.arange(n)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    trend = 0.05 * t
    noise = rng.normal(scale=1.5, size=n)
    values = np.maximum(10 + seasonal + trend + noise, 0)
    return pd.Series(values, index=idx, name='Monthly Registrations')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='Path to Excel file (optional)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic monthly series')
    parser.add_argument('--initial-arima', type=int, default=24)
    args = parser.parse_args(argv)

    if args.synthetic:
        series = make_synthetic()
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"File {args.file} not found  exiting", file=sys.stderr)
            return 2
        series = build_monthly_series_from_file(str(p))
    else:
        print('No file provided and --synthetic not set; use --synthetic or --file FILE', file=sys.stderr)
        return 2

    print('Series length:', len(series))
    results = evaluate_models(series, initial_arima=args.initial_arima)

    print('\nEvaluation results:')
    print(json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
