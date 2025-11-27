"""
Small metrics helpers for evaluating forecasts.
Provides MAE, RMSE, MAPE and R2 calculation helpers.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (in percent).

    Uses a safe divide: where y_true == 0 we exclude those points from the calculation.
    If all y_true are zero, returns np.nan.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if not np.any(mask):
        return float(np.nan)

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dictionary with MAE / RMSE / MAPE / R2."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter out any points where prediction is NaN (e.g. failed fit during walk-forward)
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # If we dropped all points, return NaN metrics to indicate evaluation failed
    if len(y_true) == 0:
        return {"MAE": float(np.nan), "RMSE": float(np.nan), "MAPE": float(np.nan), "R2": float(np.nan)}
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
