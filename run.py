"""Evaluate forecasting models and print metrics."""
import json
import sys
from src.forecast import run as run_forecast

def main():
    try:
        # Call the unified run function with the project-specific defaults
        results = run_forecast(
            file="data/inscription.xlsx",
            initial_train_monthly=24,
            initial_train_yearly=5,
            horizon_monthly=12,
            horizon_yearly=5
        )
        print('\nSummary:')
        print(json.dumps(results, indent=2))
        return 0
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
