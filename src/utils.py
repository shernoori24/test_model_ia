"""Utility functions for the forecasting project."""
import warnings
from contextlib import contextmanager
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning


@contextmanager
def suppress_arima_warnings():
    """Suppress common ARIMA warnings that are harmless during model fitting."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")
        warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.*")
        warnings.filterwarnings("ignore", category=SMConvergenceWarning)
        yield
