"""Daily heat features for one location's weather series (observed, forecast or user-specified scenario).

All temperatures in deg C. Input columns: date, tmax, tmin, tmean, atmax (apparent max), atmean, rh, plus optional
dew, rad, precip, wind. The series must be daily and contiguous for the look-backs to be valid.

Heat-health features
    atmax_l1..l3        lagged apparent max (distributed-lag structure, cf. DLNM)
    ehi_sig, ehi_accl   Excess-heat indices (Nairn & Fawcett): 3-day mean temp vs the climatological 95th percentile,
                        and vs the preceding 30 days;  ehf = max(0, ehi_sig) * max(1, ehi_accl)
    streak              consecutive days with apparent max >= 32 C (capped at 7)
    season_hot_days     log(1 + earlier days >= 32 C this year) - adaptation over the season
    tropical_night      tmin >= 24 C
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOT_APPARENT_C = 32.0
TROPICAL_NIGHT_C = 24.0


def heat_features(w: pd.DataFrame, t95: float) -> pd.DataFrame:
    """``t95``: 95th percentile of daily mean temperature over a fixed climatology (passed in to avoid leakage)."""
    w = w.sort_values("date").reset_index(drop=True)
    out = w[["date"]].copy()
    for c in ("tmax", "tmin", "tmean", "atmax", "atmean", "rh", "dew", "rad", "precip", "wind"):
        if c in w:
            out[c] = w[c].to_numpy()
    for k in (1, 2, 3):
        out[f"atmax_l{k}"] = w.atmax.shift(k)
    out["tmin_l1"] = w.tmin.shift(1)
    m3 = w.tmean.rolling(3, min_periods=1).mean()
    out["ehi_sig"] = m3 - t95
    out["ehi_accl"] = m3 - w.tmean.shift(3).rolling(30, min_periods=10).mean()
    out["ehf"] = np.maximum(out.ehi_sig, 0) * np.maximum(1, out.ehi_accl.fillna(0))
    hot = (w.atmax >= HOT_APPARENT_C).astype(int)
    spell = (hot != hot.shift()).cumsum()
    out["streak"] = np.minimum(hot.groupby(spell).cumsum() * hot, 7)
    out["season_hot_days"] = np.log1p(hot.groupby(w.date.dt.year).cumsum() - hot)
    out["tropical_night"] = (w.tmin >= TROPICAL_NIGHT_C).astype(int)
    out["atmax_x_streak"] = np.maximum(w.atmax - HOT_APPARENT_C, 0) * out.streak
    return out


RENAME = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "temperature_2m_mean": "tmean",
          "apparent_temperature_max": "atmax", "apparent_temperature_mean": "atmean", "apparent_temperature_min": "atmin",
          "relative_humidity_2m_mean": "rh", "dew_point_2m_mean": "dew", "shortwave_radiation_sum": "rad",
          "precipitation_sum": "precip", "wind_speed_10m_max": "wind"}


def openmeteo_daily(df: pd.DataFrame, loc_id: str) -> pd.DataFrame:
    d = df[df.loc_id == loc_id].rename(columns=RENAME)
    return d[["date"] + [c for c in RENAME.values() if c in d]].sort_values("date").reset_index(drop=True)
