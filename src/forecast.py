"""ARIMA workflow: auto-select order, evaluate, diagnostics, and forecast.

Produces:
- printed summary (order, walk-forward metrics)
- saved forecast CSV at `outputs/arima_forecast.csv`
- saved plots at `outputs/arima_forecast.png`, `outputs/arima_residuals.png`, `outputs/arima_acf.png`
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning
from statsmodels.graphics.tsaplots import plot_acf

from src.time_series import build_series_from_file
from src.evaluate import walk_forward_arima, compute_metrics


def ensure_out_dir():
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    return out


def fit_auto_arima(series, freq: str = 'monthly'):
    # choose seasonal behavior depending on frequency
    if str(freq).lower() in ('monthly', 'm', 'ms'):
        model = pm.auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True,
                              error_action='ignore', max_p=5, max_q=5, max_P=2, max_Q=2)
    else:
        # yearly or other low-frequency series usually don't have intra-period seasonality
        model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True,
                              error_action='ignore', max_p=5, max_q=5)
    return model


def fit_and_forecast_full(series, order, steps=12):
    m = ARIMA(series, order=order)
    # Suppress known harmless startup warnings and use a robust optimizer
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")
        warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.*")
        warnings.filterwarnings("ignore", category=SMConvergenceWarning)
        # Use default estimator but silence known benign startup warnings
        res = m.fit()
    forecast = res.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    conf = forecast.conf_int()
    return res, mean, conf

#Run forecasting workflow for both monthly and yearly frequencies.
def run(file: str = 'data/inscription.xlsx',
        initial_train_monthly: int = 24,
        initial_train_yearly: int = 5,
        horizon_monthly: int = 12,
        horizon_yearly: int = 5):

    if not file:
        raise SystemExit('file argument required')
    out = ensure_out_dir()
    results = {}

    for freq in ('monthly', 'yearly'):
        # Select parameters based on frequency
        if freq == 'monthly':
            initial_train = initial_train_monthly
            horizon = horizon_monthly
        else:
            initial_train = initial_train_yearly
            horizon = horizon_yearly

        series = build_series_from_file(file, freq=freq)

        print(f'\n--- Processing {freq} series ---')
        print('Running auto_arima (this may take a minute)...')
        try:
            am = fit_auto_arima(series, freq=freq)
            order = am.order
            seasonal_order = am.seasonal_order
            print('Selected order:', order, 'seasonal_order:', seasonal_order)

            # Walk-forward with selected order
            print('Evaluating with walk-forward...')
            actuals, preds = walk_forward_arima(series, order=order, initial_train=initial_train)
            metrics = compute_metrics(actuals, preds)
            print('Walk-forward metrics:', json.dumps(metrics, indent=2))

            # Fit on full series and forecast
            print(f'Fitting final ARIMA{order} on full series and forecasting {horizon} steps...')
            res, mean, conf = fit_and_forecast_full(series, order=order, steps=horizon)

            # Save forecast CSV — build index according to frequency
            if str(freq).lower() in ('monthly', 'm', 'ms'):
                start_offset = pd.offsets.MonthBegin(1)
                idx = pd.date_range(start=series.index[-1] + start_offset, periods=horizon, freq='MS')
                freq_key = 'monthly'
            else:
                # yearly forecasting — use YearBegin / 'YS' (year start alignment)
                start_offset = pd.offsets.YearBegin(1)
                idx = pd.date_range(start=series.index[-1] + start_offset, periods=horizon, freq='YS')
                freq_key = 'yearly'

            df_f = pd.DataFrame({'forecast': mean.values, 'lower': conf.iloc[:, 0].values, 'upper': conf.iloc[:, 1].values}, index=idx)
            out_csv = out / f'arima_forecast_{freq_key}.csv'
            df_f.to_csv(out_csv)
            print('Saved forecast CSV to', out_csv)

            # Plot historical + forecast
            fig, ax = plt.subplots(figsize=(10, 5))
            series.plot(ax=ax, label='history')
            df_f['forecast'].plot(ax=ax, label='forecast')
            ax.fill_between(df_f.index, df_f['lower'], df_f['upper'], color='gray', alpha=0.3)
            ax.legend()
            ax.set_title(f'ARIMA{order} forecast ({freq})')
            fig_path = out / f'arima_forecast_{freq_key}.png'
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
            print('Saved forecast plot to', fig_path)

            # Residuals
            resid = res.resid
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            resid.plot(ax=ax[0], title='Residuals')
            resid.hist(ax=ax[1], bins=30)
            ax[1].set_title('Residuals histogram')
            resid_path = out / f'arima_residuals_{freq_key}.png'
            fig.savefig(resid_path, bbox_inches='tight')
            plt.close(fig)
            print('Saved residuals plot to', resid_path)

            # ACF
            resid_clean = resid.dropna()
            if len(resid_clean) >= 2:
                fig = plt.figure(figsize=(8, 4))
                # Adjust lags for short series
                lags = min(36, max(1, len(resid_clean) - 1))
                plot_acf(resid_clean, lags=lags, ax=plt.gca())
                acf_path = out / f'arima_acf_{freq_key}.png'
                plt.tight_layout()
                plt.savefig(acf_path)
                plt.close()
                print('Saved ACF plot to', acf_path)
            
            # store per-frequency result summary
            results[freq_key] = {'order': order, 'seasonal_order': seasonal_order, 'walk_metrics': metrics, 'forecast_csv': str(out_csv)}

        except Exception as ex:
            print(f"Error processing {freq} series: {ex}")
            results[freq_key] = {'error': str(ex)}

    return results
