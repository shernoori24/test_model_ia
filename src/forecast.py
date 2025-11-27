"""ARIMA workflow: auto-select order, evaluate, diagnostics, and forecast."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

from src.time_series import build_series_from_file
from src.evaluate import walk_forward_arima, compute_metrics
from src.utils import suppress_arima_warnings


def ensure_out_dir():
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    return out


def fit_auto_arima(series, freq: str = 'monthly'):
    # choose seasonal behavior depending on frequency
    if str(freq).lower() in ('monthly', 'm', 'ms'):
        model = pm.auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True,
                              error_action='ignore', max_p=6, max_q=6, max_P=2, max_Q=2)
    else:
        # yearly or other low-frequency series usually don't have intra-period seasonality
        model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True,
                              error_action='ignore', max_p=6, max_q=6)
    return model


def fit_and_forecast_full(series, order, steps=12):
    m = ARIMA(series, order=order)
    with suppress_arima_warnings():
        res = m.fit()
    forecast = res.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    conf = forecast.conf_int()
    return res, mean, conf


def run(file: str = 'data/inscription.xlsx',
        initial_train_monthly: int = 24,
        initial_train_yearly: int = 5,
        horizon_monthly: int = 12,
        horizon_yearly: int = 5):
    """Run forecasting workflow for both monthly and yearly frequencies.

    Parameters are defaults so the function can be called without CLI arguments.
    """
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
        # Apply Log-Transformation to stabilize variance
        series_log = np.log1p(series)

        print(f'\n{"="*70}')
        print(f'  {freq.upper()} FORECASTING'.center(70))
        print(f'{"="*70}')
        print(f'\n⏳ Selecting optimal ARIMA model...')
        try:
            # Fit on log series
            am = fit_auto_arima(series_log, freq=freq)
            order = am.order
            seasonal_order = am.seasonal_order
            print(f'✓ Model selected: ARIMA{order}')
            if seasonal_order != (0, 0, 0, 0):
                print(f'  Seasonal component: {seasonal_order}')

            # Walk-forward with selected order (on log series)
            print(f'\n🔄 Evaluating with walk-forward validation...')
            actuals_log, preds_log = walk_forward_arima(series_log, order=order, initial_train=initial_train)
            
            # Inverse transform for metrics (back to original scale)
            actuals = np.expm1(actuals_log)
            preds = np.expm1(preds_log)
            metrics = compute_metrics(actuals, preds)
            print(f'\n📊 Performance Metrics:')
            print(f'  ├─ MAE:  {metrics["MAE"]:.2f}')
            print(f'  ├─ RMSE: {metrics["RMSE"]:.2f}')
            print(f'  ├─ MAPE: {metrics["MAPE"]:.1f}%')
            print(f'  └─ R²:   {metrics["R2"]:.4f}')

            # Fit on full series and forecast (on log series)
            print(f'\n🔮 Generating {horizon}-step forecast...')
            res, mean_log, conf_log = fit_and_forecast_full(series_log, order=order, steps=horizon)

            # Inverse transform forecast (back to original scale)
            mean = np.expm1(mean_log)
            conf = np.expm1(conf_log)

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
            out_json = out / f'arima_forecast_{freq_key}.json'
            df_f.to_json(out_json, orient='index', date_format='iso', indent=2)

            # Plot historical + forecast (original scale)
            fig, ax = plt.subplots(figsize=(10, 5))
            series.plot(ax=ax, label='history')
            df_f['forecast'].plot(ax=ax, label='forecast')
            ax.fill_between(df_f.index, df_f['lower'], df_f['upper'], color='gray', alpha=0.3)
            ax.legend()
            ax.set_title(f'ARIMA{order} Forecast - {freq.title()}')
            fig_path = out / f'arima_forecast_{freq_key}.png'
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)

            # Residuals (from log model)
            resid = res.resid
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            resid.plot(ax=ax[0], title='Residuals (Log Scale)')
            resid.hist(ax=ax[1], bins=30)
            ax[1].set_title('Residuals histogram (Log Scale)')
            resid_path = out / f'arima_residuals_{freq_key}.png'
            fig.savefig(resid_path, bbox_inches='tight')
            plt.close(fig)

            # ACF (from log model residuals)
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
            
            # Output saved files
            print(f'\n💾 Outputs saved:')
            print(f'  ├─ Forecast: {out_json.name}')
            print(f'  ├─ Plot: {fig_path.name}')
            print(f'  ├─ Residuals: {resid_path.name}')
            print(f'  └─ ACF: {acf_path.name}')
            
            # store per-frequency result summary
            results[freq_key] = {'order': order, 'seasonal_order': seasonal_order, 'walk_metrics': metrics, 'forecast_json': str(out_json)}

        except Exception as ex:
            print(f'\n❌ Error processing {freq} series: {ex}')
            results[freq_key] = {'error': str(ex)}

    print(f'\n{"="*70}\n')
    return results
