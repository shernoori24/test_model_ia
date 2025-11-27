"""Time series helper functions for this project."""
from __future__ import annotations

import pandas as pd
from typing import Optional


def build_monthly_series_from_df(df: pd.DataFrame, date_col='Date inscription', fallback_col='Première venue') -> pd.Series:
    """Return a monthly-count time series (pd.Series) from a DataFrame.

    - df: dataframe containing date columns
    - date_col: primary date column to use
    - fallback_col: fallback date column when primary is missing

    The returned series is indexed by a DatetimeIndex (month start) and contains integer counts.
    """
    # Ensure date parsing for any known date columns
    for c in (date_col, fallback_col):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)

    # Build effective date using named columns if available
    if date_col in df.columns and fallback_col in df.columns:
        df['Effective Date'] = df[date_col].fillna(df[fallback_col])
    elif date_col in df.columns:
        df['Effective Date'] = df[date_col]
    elif fallback_col in df.columns:
        df['Effective Date'] = df[fallback_col]
    else:
        # Auto-detect a date column: pick the column with the most parseable dates
        best_col = None
        best_count = 0
        best_recent_year = 0
        for col in df.columns:
            # try to parse column values
            parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            cnt = int(parsed.notna().sum())
            recent_year = 0
            if cnt > 0:
                try:
                    recent_year = int(pd.Series(parsed.dropna()).dt.year.max())
                except Exception:
                    recent_year = 0
            # prefer columns with more parseable dates and more recent years
            if cnt > best_count or (cnt == best_count and recent_year > best_recent_year):
                best_col = col
                best_count = cnt
                best_recent_year = recent_year

        if best_col is None or best_count == 0:
            raise ValueError('No date column found in DataFrame (auto-detect failed)')
        # assign parsed dates back to df under that column name
        df[best_col] = pd.to_datetime(df[best_col], errors='coerce', dayfirst=True)
        df['Effective Date'] = df[best_col]

    # Drop rows without an effective date
    df = df[df['Effective Date'].notna()].copy()
    if df.empty:
        raise ValueError('No valid Effective Date values found')

    # Set index and resample monthly (month-start), count per month
    monthly = df.set_index('Effective Date').resample('MS').size().rename('Monthly Registrations').astype(int)
    return monthly


def build_monthly_series_from_file(path: str, **kwargs) -> pd.Series:
    df = pd.read_excel(path)
    return build_monthly_series_from_df(df, **kwargs)
