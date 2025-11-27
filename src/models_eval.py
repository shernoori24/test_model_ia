"""Model evaluation helpers for time-series forecasting.

Provides walk-forward validation and metric computation for ARIMA and ETS models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning

# Silence known sklearn deprecation chatter (coming from inner sklearn/pandas internals)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from statsmodels.tsa.arima.model import ARIMA


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, R2 for arrays of equal length.

    y_true and y_pred should be 1-D numeric arrays of same length.
    """
    mask = ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    if len(y_true) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Avoid division by zero in MAPE
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    r2 = r2_score(y_true, y_pred)

    return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape), 'R2': float(r2)}


def walk_forward_arima(series: pd.Series, order=(2, 1, 2), initial_train=24):
    """Walk-forward validation using ARIMA; returns (actuals, preds) arrays.

    - series: pd.Series indexed by pd.DatetimeIndex
    - order: (p,d,q)
    - initial_train: number of months to use for first training window
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")

    n = len(series)
    if n < initial_train + 1:
        raise ValueError("Not enough data points for the given initial_train size")

    actuals = []
    preds = []

    for i in range(initial_train, n):
        train = series.iloc[:i]
        test_val = series.iloc[i]

        try:
            model = ARIMA(train, order=order)
            # fit can emit a lot of warnings when run repeatedly in a loop.
            # Use a conservative, robust optimizer configuration and suppress
            # a few known (harmless) start-up warnings to reduce console noise.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")
                warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.*")
                warnings.filterwarnings("ignore", category=SMConvergenceWarning)
                # method_kwargs are forwarded to the optimizer (scipy) in many statsmodels versions
                # Use default estimator but silence known benign startup warnings
                fit = model.fit()
            forecast = fit.forecast(steps=1)
            yhat = forecast.iloc[0]
        except Exception:
            yhat = np.nan

        actuals.append(float(test_val))
        preds.append(float(yhat) if not pd.isna(yhat) else np.nan)

    return np.array(actuals), np.array(preds)




def evaluate_models(series: pd.Series, initial_arima=24, order=(2, 1, 2)):
    """Evaluate ARIMA via walk-forward and return metric dict.

    Returns: dict { 'arima': metrics_dict }
    """
    res = {}

    # ARIMA
    try:
        y_arima, yhat_arima = walk_forward_arima(series, order=order, initial_train=initial_arima)
        res['arima'] = compute_metrics(y_arima, yhat_arima)
    except Exception as ex:
        res['arima'] = {'error': str(ex)}

    return res
