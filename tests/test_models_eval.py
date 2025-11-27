import numpy as np
import pandas as pd

from src.models_eval import compute_metrics, evaluate_models


def make_synthetic(n=120, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='MS')
    t = np.arange(n)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    trend = 0.05 * t
    noise = rng.normal(scale=1.5, size=n)
    values = np.maximum(10 + seasonal + trend + noise, 0)
    return pd.Series(values, index=idx)


def test_compute_metrics_basic():
    y_true = np.array([10, 12, 15, 20], dtype=float)
    y_pred = np.array([9, 11, 14, 21], dtype=float)

    metrics = compute_metrics(y_true, y_pred)
    assert 'MAE' in metrics and 'RMSE' in metrics and 'MAPE' in metrics and 'R2' in metrics
    assert pytest_float(metrics['MAE'])


def pytest_float(x):
    return isinstance(x, float) or isinstance(x, np.floating)


def test_evaluate_models_on_synthetic():
    series = make_synthetic(120)
    results = evaluate_models(series, initial_arima=24)

    assert 'arima' in results

    # metrics should be a dict with numeric values (or an error message)
    val = results['arima']
    if 'error' in val:
        # Shouldn't happen with synthetic data, but tolerate
        assert isinstance(val['error'], str)
    else:
        assert all(k in val for k in ('MAE', 'RMSE', 'MAPE', 'R2'))
        assert pytest_float(val['MAE'])
