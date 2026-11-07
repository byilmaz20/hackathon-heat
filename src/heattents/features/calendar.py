"""Calendar features: US federal holidays (rule-based, no dependency), school summer break, COVID period."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _nth_weekday(year, month, weekday, n):
    d = pd.Timestamp(year, month, 1)
    d += pd.Timedelta(days=(weekday - d.day_of_week) % 7)
    return d + pd.Timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    d = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.day_of_week - weekday) % 7)


def _observed(d):
    return d - pd.Timedelta(days=1) if d.day_of_week == 5 else d + pd.Timedelta(days=1) if d.day_of_week == 6 else d


def federal_holidays(years) -> pd.DatetimeIndex:
    """Warm-season federal holidays (observed dates): Memorial Day, Juneteenth (2021+), Independence Day, Labor Day."""
    out = []
    for y in years:
        out += [_last_weekday(y, 5, 0), _observed(pd.Timestamp(y, 7, 4)), _nth_weekday(y, 9, 0, 1)]
        if y >= 2021:
            out.append(_observed(pd.Timestamp(y, 6, 19)))
    return pd.DatetimeIndex(out)


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    years = range(d.dt.year.min(), d.dt.year.max() + 1)
    hol = federal_holidays(years)
    # NYC public schools: last day ~ 26 June, reopen the Thursday after Labor Day
    school_off = np.zeros(len(d), bool)
    for y in years:
        start = pd.Timestamp(y, 6, 27)
        end = _nth_weekday(y, 9, 0, 1) + pd.Timedelta(days=3)
        school_off |= ((d >= start) & (d < end)).to_numpy()
    return pd.DataFrame({
        "dow": d.dt.day_of_week.to_numpy(),
        "weekend": (d.dt.day_of_week >= 5).astype(int).to_numpy(),
        "holiday": d.isin(hol).astype(int).to_numpy(),
        "july4_window": ((d.dt.month == 7) & d.dt.day.between(3, 5)).astype(int).to_numpy(),
        "school_break": school_off.astype(int),
        "covid": d.dt.year.isin([2020, 2021]).astype(int).to_numpy(),
        "year": d.dt.year.to_numpy(),
        "day_of_year": d.dt.day_of_year.to_numpy(),
    }, index=dates.index)
