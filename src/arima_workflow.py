"""ARIMA workflow: auto-select order, evaluate, diagnostics, and forecast.

Produces:
- printed summary (order, walk-forward metrics)
- saved forecast CSV at `outputs/arima_forecast.csv`
- saved plots at `outputs/arima_forecast.png`, `outputs/arima_residuals.png`, `outputs/arima_acf.png`
"""
import os
import argparse
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

from src.ts_utils import build_monthly_series_from_file
from src.models_eval import walk_forward_arima
from src.metrics import compute_all


def ensure_out_dir():
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    return out


def fit_auto_arima(series):
    # use seasonal=True with m=12 (monthly data)
    model = pm.auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True,
                         error_action='ignore', max_p=5, max_q=5, max_P=2, max_Q=2)
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


def run(args):
    if args.file:
        series = build_monthly_series_from_file(args.file)
    else:
        raise SystemExit('file argument required')

    out = ensure_out_dir()

    print('Running auto_arima (this may take a minute)...')
    am = fit_auto_arima(series)
    order = am.order
    seasonal_order = am.seasonal_order
    print('Selected order:', order, 'seasonal_order:', seasonal_order)

    # Walk-forward with selected order
    print('Evaluating with walk-forward...')
    actuals, preds = walk_forward_arima(series, order=order, initial_train=args.initial_train)
    metrics = compute_all(actuals, preds)
    print('Walk-forward metrics:', json.dumps(metrics, indent=2))

    # Fit on full series and forecast
    print(f'Fitting final ARIMA{order} on full series and forecasting {args.horizon} steps...')
    res, mean, conf = fit_and_forecast_full(series, order=order, steps=args.horizon)

    # Save forecast CSV
    idx = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=args.horizon, freq='MS')
    df_f = pd.DataFrame({'forecast': mean.values, 'lower': conf.iloc[:, 0].values, 'upper': conf.iloc[:, 1].values}, index=idx)
    out_csv = out / 'arima_forecast.csv'
    df_f.to_csv(out_csv)
    print('Saved forecast CSV to', out_csv)

    # Plot historical + forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(ax=ax, label='history')
    df_f['forecast'].plot(ax=ax, label='forecast')
    ax.fill_between(df_f.index, df_f['lower'], df_f['upper'], color='gray', alpha=0.3)
    ax.legend()
    ax.set_title(f'ARIMA{order} forecast')
    fig_path = out / 'arima_forecast.png'
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print('Saved forecast plot to', fig_path)

    # Residuals
    resid = res.resid
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    resid.plot(ax=ax[0], title='Residuals')
    resid.hist(ax=ax[1], bins=30)
    ax[1].set_title('Residuals histogram')
    resid_path = out / 'arima_residuals.png'
    fig.savefig(resid_path, bbox_inches='tight')
    plt.close(fig)
    print('Saved residuals plot to', resid_path)

    # ACF
    fig = plt.figure(figsize=(8, 4))
    plot_acf(resid.dropna(), lags=36, ax=plt.gca())
    acf_path = out / 'arima_acf.png'
    plt.tight_layout()
    plt.savefig(acf_path)
    plt.close()
    print('Saved ACF plot to', acf_path)

    return {'order': order, 'seasonal_order': seasonal_order, 'walk_metrics': metrics, 'forecast_csv': str(out_csv)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Excel path to build monthly series')
    parser.add_argument('--initial-train', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=12)
    args = parser.parse_args()
    res = run(args)
    print('\nSummary:')
    print(json.dumps(res, indent=2))
