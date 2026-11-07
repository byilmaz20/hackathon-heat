"""Neighbourhood daily temperature from Daymet V4 (NASA / ORNL DAAC, 1 km grid), at the 42 UHF42 neighbourhood centres.

Source: Daymet single-pixel API (https://daymet.ornl.gov/single-pixel/), no key and no daily quota.
Variables: tmax, tmin (deg C), vp (water vapour pressure, Pa). Heat index ("apparent max") is derived from tmax and
the relative humidity implied by vp at tmax (Steadman / NWS Rothfusz regression, as in the NWS heat index).
Citation: Thornton, M. M. et al. (2022). Daymet: Daily Surface Weather Data on a 1-km Grid for North America,
Version 4 R1. ORNL DAAC. doi:10.3334/ORNLDAAC/2129
Used for training the local-heat share model (HRRR estimates via Open-Meteo hit a daily quota).
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .http import get
from .weather_local import centroids

URL = "https://daymet.ornl.gov/single-pixel/api/data"


def heat_index_c(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """NWS heat index (Rothfusz regression with the simple formula below 80 F), returned in deg C."""
    t = np.asarray(t_c) * 9 / 5 + 32
    rh = np.clip(np.asarray(rh), 0, 100)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 6.83783e-3 * t ** 2
            - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t ** 2 * rh + 8.5282e-4 * t * rh ** 2 - 1.99e-6 * t ** 2 * rh ** 2)
    hi = np.where((simple + t) / 2 >= 80, full, simple)
    return (hi - 32) * 5 / 9


def _one(row) -> pd.DataFrame:
    txt = get(URL, {"lat": f"{row.lat:.4f}", "lon": f"{row.lon:.4f}", "vars": "tmax,tmin,vp",
                    "start": "2005-01-01", "end": "2025-12-31"}, timeout=300).decode()
    body = txt[txt.index("year,yday"):]
    d = pd.read_csv(io.StringIO(body))
    d.columns = ["year", "yday", "tmax", "tmin", "vp"]
    d["date"] = pd.to_datetime(d.year.astype(str), format="%Y") + pd.to_timedelta(d.yday - 1, unit="D")
    es = 610.94 * np.exp(17.625 * d.tmax / (d.tmax + 243.04))  # saturation vapour pressure at tmax (Pa)
    d["rh_at_tmax"] = np.clip(100 * d.vp / es, 0, 100)
    d["atmax"] = heat_index_c(d.tmax.to_numpy(), d.rh_at_tmax.to_numpy())
    d["uhf"] = row.uhf
    return d[d.date.dt.month.isin([5, 6, 7, 8, 9])][["uhf", "date", "tmax", "tmin", "rh_at_tmax", "atmax"]]


def fetch(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "weather_uhf42_daymet.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    c = centroids(paths)
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = list(ex.map(_one, c.itertuples()))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"[ingest:daymet] {df.uhf.nunique()} neighbourhoods, {df.date.min():%Y-%m-%d}..{df.date.max():%Y-%m-%d} -> {out}")
    return df
