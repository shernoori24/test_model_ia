"""ETL helpers for registration time series preprocessing.

Functions are written so they are deterministic and testable without the original dataset.
"""
from __future__ import annotations

import pandas as pd


def load_inscription_excel(path: str) -> pd.DataFrame:
    """Load glossary data from an Excel file.

    This function intentionally doesn't try to install packages or alter environment.
    It returns an unmodified DataFrame.
    """
    return pd.read_excel(path)


def prepare_dates(df: pd.DataFrame, date_cols=None) -> pd.DataFrame:
    """Parse important date columns and create an 'Effective Date'.

    - Parses columns in `date_cols` to datetimes (errors='coerce', dayfirst=True).
    - Creates 'Effective Date' from 'Date inscription' falling back to 'Première venue'.
    - Adds integer year/month columns and a Year-Month string.
    - Drops rows with no Effective Date and resets index.

    Returns a new DataFrame (does not mutate input).
    """
    df = df.copy()
    if date_cols is None:
        date_cols = ['Date de naissance', 'Date inscription', 'Première venue']

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # Prefer 'Date inscription' but fall back to 'Première venue'
    if 'Date inscription' in df.columns and 'Première venue' in df.columns:
        df['Effective Date'] = df['Date inscription'].fillna(df['Première venue'])
    elif 'Date inscription' in df.columns:
        df['Effective Date'] = df['Date inscription']
    elif 'Première venue' in df.columns:
        df['Effective Date'] = df['Première venue']
    else:
        # Nothing to do — create a column of NaT
        df['Effective Date'] = pd.NaT

    # Drop rows without an effective date
    df = df[df['Effective Date'].notna()].reset_index(drop=True)

    df['Effective_Year'] = df['Effective Date'].dt.year.astype('Int64')
    df['Effective_Month'] = df['Effective Date'].dt.month.astype('Int64')
    df['Year-Month'] = df['Effective_Year'].astype(str) + '-' + df['Effective_Month'].astype(str).str.zfill(2)

    return df


def build_monthly_series(df: pd.DataFrame, date_index='Effective Date') -> pd.Series:
    """Create a continuous monthly series (MS - month start) of registration counts.

    The returned Series has a DatetimeIndex at month start, integer counts, and no missing months.
    """
    if date_index not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"date_index {date_index} not found in DataFrame columns or DateTimeIndex")

    # If date_index is a column, set as index
    if date_index in df.columns:
        ts = df.set_index(date_index).index
    else:
        ts = df.index

    # Build counts per month and resample to ensure continuity
    s = (pd.Series(1, index=ts)
         .resample('MS')
         .sum()
         .rename('Monthly Registrations')
         .fillna(0)
         .astype(int))

    return s
