import pandas as pd
import numpy as np

from src.time_series import build_series_from_df


def make_dates_years(start_year=2000, n=10):
    # create n timestamps at Jan 1 of each year (use 'YS' alias)
    years = pd.date_range(start=pd.Timestamp(f'{start_year}-01-01'), periods=n, freq='YS')
    return years


def test_build_yearly_series_from_df():
    # build a frame with one event per year and some extra events
    years = make_dates_years(2000, 6)
    # create some duplicate events within certain years
    dates = list(years) + [years[0] + pd.Timedelta(days=10), years[2] + pd.Timedelta(days=100)]
    df = pd.DataFrame({'Date inscription': dates})

    s = build_series_from_df(df, date_col='Date inscription', freq='yearly')

    # expect 6 unique year bins
    assert isinstance(s, pd.Series)
    assert s.name == 'Yearly Registrations'
    # index should be year-start aligned and a yearly frequency (YS/AS variants)
    assert isinstance(s.index, pd.DatetimeIndex)
    # accomodate variants in pandas representation (YS, YS-JAN, YearBegin...)
    freq_repr = getattr(s.index, 'freqstr', None) or str(s.index.freq)
    freq_repr = str(freq_repr).upper()
    assert freq_repr.startswith('YS') or freq_repr.startswith('AS') or 'YEAR' in freq_repr
    # counts should reflect duplicates
    assert s.iloc[0] >= 2
    assert len(s) == 6
