"""Evaluate forecasting models and print metrics."""
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
        
        # Clean summary output
        print('╔' + '═'*68 + '╗')
        print('║' + ' FORECASTING SUMMARY '.center(68) + '║')
        print('╠' + '═'*68 + '╣')
        
        for freq_key, data in results.items():
            if 'error' in data:
                print(f'║ {freq_key.upper()}: ERROR - {data["error"]:<54} ║')
            else:
                metrics = data['walk_metrics']
                print(f'║ {freq_key.upper():<10} │ R²={metrics["R2"]:.4f} │ RMSE={metrics["RMSE"]:>7.2f} │ MAPE={metrics["MAPE"]:>5.1f}% ║')
        
        print('╚' + '═'*68 + '╝')
        return 0
    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
