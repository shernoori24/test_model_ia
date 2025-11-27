import numpy as np
from src.metrics import mae, rmse, mape, r2, compute_all


def test_basic_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    metrics = compute_all(y_true, y_pred)

    # MAE = mean(|[2,2,3]|) = 7/3 ~= 2.333
    assert pytest_approx(metrics['MAE'], 2.333, tol=1e-3)
    # RMSE sqrt(mean([4,4,9])) = sqrt(17/3) ~= 2.378
    assert pytest_approx(metrics['RMSE'], 2.3804761428476167, tol=1e-3)


def test_mape_zero_handling():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([0.0, 12.0, 18.0])

    # MAPE should ignore zero true values (first entry) and compute for the rest
    m = mape(y_true, y_pred)
    # For the two non-zero: |(10-12)/10|=0.2 , |(20-18)/20|=0.1 -> mean=0.15 -> *100 = 15%
    assert pytest_approx(m, 15.0, tol=1e-6)


def test_mape_all_zero():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])

    m = mape(y_true, y_pred)
    assert np.isnan(m)


def pytest_approx(a, b, tol=1e-6):
    return abs(a - b) <= tol
