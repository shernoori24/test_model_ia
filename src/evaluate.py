"""Model evaluation for time-series forecasting."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from statsmodels.tsa.arima.model import ARIMA
from src.utils import suppress_arima_warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, R2 for arrays of equal length."""
    mask = ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    if len(y_true) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    # r2_score requires at least two samples and non-constant y_true; return nan when undefined
    try:
        if len(y_true) < 2 or np.nanstd(y_true) == 0:
            r2 = np.nan
        else:
            r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape), 'R2': float(r2)}


def walk_forward_arima(series: pd.Series, order=(2, 1, 2), initial_train=24):
    """Walk-forward validation using ARIMA; returns (actuals, preds) arrays."""
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
            with suppress_arima_warnings():
                fit = model.fit()
            forecast = fit.forecast(steps=1)
            yhat = forecast.iloc[0]
        except Exception:
            yhat = np.nan

        actuals.append(float(test_val))
        preds.append(float(yhat) if not pd.isna(yhat) else np.nan)

    return np.array(actuals), np.array(preds)


def evaluate_models(series: pd.Series, initial_arima=24, order=(2, 1, 2)):
    """Evaluate ARIMA via walk-forward and return metric dict."""
    try:
        y_arima, yhat_arima = walk_forward_arima(series, order=order, initial_train=initial_arima)
        return {'arima': compute_metrics(y_arima, yhat_arima)}
    except Exception as ex:
        return {'arima': {'error': str(ex)}}
