"""Evaluate forecasting models and print metrics."""
import json
import sys
from pathlib import Path

from src.time_series import build_series_from_file
from src.evaluate import evaluate_models
from src.forecast import ensure_out_dir, fit_auto_arima, fit_and_forecast_full
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd

DATA_FILE = "data/inscription.xlsx"


def main():
    # Use fixed defaults so `run.py` can be run without CLI args
    initial_train_monthly = 24
    initial_train_yearly = 5
    horizon_yearly = 5

    p = Path(DATA_FILE)
    if not p.exists():
        print(f"File {DATA_FILE} not found", file=sys.stderr)
        return 2
    
    # Evaluate monthly series (existing behaviour)
    series_monthly = build_series_from_file(str(p), freq='monthly')
    print('Monthly series length:', len(series_monthly))
    results_monthly = evaluate_models(series_monthly, initial_arima=initial_train_monthly)
    print('\nMonthly evaluation results:')
    print(json.dumps(results_monthly, indent=2))

    # Also evaluate yearly aggregation
    series_yearly = build_series_from_file(str(p), freq='yearly')
    print('\nYearly series length:', len(series_yearly))
    try:
        results_yearly = evaluate_models(series_yearly, initial_arima=initial_train_yearly)
    except Exception as ex:
        results_yearly = {'arima': {'error': str(ex)}}
    print('\nYearly evaluation results:')
    print(json.dumps(results_yearly, indent=2))

    # Fit final ARIMA on yearly series and save a forecast CSV
    out = ensure_out_dir()
    try:
        print('\nFitting ARIMA on yearly series (this may take a moment)...')
        am = fit_auto_arima(series_yearly, freq='yearly')
        order = am.order
        print('Selected order for yearly ARIMA:', order)
        res, mean, conf = fit_and_forecast_full(series_yearly, order=order, steps=horizon_yearly)

        idx = pd.date_range(start=series_yearly.index[-1] + pd.offsets.YearBegin(1), periods=horizon_yearly, freq='YS')
        df_f = pd.DataFrame({'forecast': mean.values, 'lower': conf.iloc[:, 0].values, 'upper': conf.iloc[:, 1].values}, index=idx)
        out_csv = out / 'arima_forecast_yearly.csv'
        df_f.to_csv(out_csv)
        print('Saved yearly forecast CSV to', out_csv)

        # Also save plots for yearly forecast (history + forecast, residuals, ACF)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            series_yearly.plot(ax=ax, label='history')
            df_f['forecast'].plot(ax=ax, label='forecast')
            ax.fill_between(df_f.index, df_f['lower'], df_f['upper'], color='gray', alpha=0.3)
            ax.legend()
            ax.set_title(f'ARIMA{order} yearly forecast')
            fig_path = out / 'arima_forecast_yearly.png'
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
            print('Saved yearly forecast plot to', fig_path)

            # Residuals
            resid = res.resid
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            resid.plot(ax=ax[0], title='Residuals')
            resid.hist(ax=ax[1], bins=30)
            ax[1].set_title('Residuals histogram')
            resid_path = out / 'arima_residuals_yearly.png'
            fig.savefig(resid_path, bbox_inches='tight')
            plt.close(fig)
            print('Saved yearly residuals plot to', resid_path)

            # ACF — use a safe number of lags for short yearly series
            resid_clean = resid.dropna()
            if len(resid_clean) >= 2:
                fig = plt.figure(figsize=(8, 4))
                lags = min(36, max(1, len(resid_clean) - 1))
                plot_acf(resid_clean, lags=lags, ax=plt.gca())
                acf_path = out / 'arima_acf_yearly.png'
            else:
                raise RuntimeError('Not enough residuals to plot ACF')
            plt.tight_layout()
            plt.savefig(acf_path)
            plt.close()
            print('Saved yearly ACF plot to', acf_path)
        except Exception as ex:
            print('Failed to create yearly plots:', ex)
    except Exception as ex:
        print('Failed to produce yearly forecast CSV:', ex)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
