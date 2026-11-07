"""Weather inputs from Open-Meteo (ERA5 / ERA5-Land reanalysis and archived forecasts).

borough_daily   daily weather (ERA5-Land, 0.1 deg) at one representative point per borough, April-September;
                within-city microclimate is carried by the static surface-temperature layer instead (keeps the
                download inside Open-Meteo's free quota)
upstream_daily  daily weather for cities upstream of NYC in typical summer flow (lead-lag features)
hourly_central  hourly temperature / heat index at Central Park (intra-day profile)
forecasts       archived real forecasts issued 1 and 3 days ahead (Previous Runs API, 2022 onwards)
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import PATHS, Paths
from .http import get_json

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
TZ = "America/New_York"
START, END = "2005-01-01", "2026-09-30"
DAILY = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_max",
         "apparent_temperature_mean", "apparent_temperature_min", "relative_humidity_2m_mean", "dew_point_2m_mean",
         "shortwave_radiation_sum", "sunshine_duration", "precipitation_sum", "wind_speed_10m_max"]
HOURLY = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m", "shortwave_radiation"]
UPSTREAM = {  # south-west of NYC: summer heat usually arrives along this corridor
    "newark": (40.7357, -74.1724), "trenton": (40.2206, -74.7597), "philadelphia": (39.9526, -75.1652),
    "baltimore": (39.2904, -76.6122), "washington_dc": (38.9072, -77.0369), "central_park": (40.7812, -73.9665),
}


def _daily_frames(resp, ids) -> pd.DataFrame:
    resp = resp if isinstance(resp, list) else [resp]
    frames = []
    for loc_id, r in zip(ids, resp):
        d = pd.DataFrame(r["daily"]).rename(columns={"time": "date"})
        d.insert(0, "loc_id", loc_id)
        d["grid_lat"], d["grid_lon"] = r["latitude"], r["longitude"]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _query_daily(points: dict[str, tuple[float, float]], model: str | None, batch: int = 10) -> pd.DataFrame:
    ids = list(points)
    frames = []
    for k in range(0, len(ids), batch):
        chunk = ids[k:k + batch]
        params = {"latitude": ",".join(f"{points[i][0]:.4f}" for i in chunk),
                  "longitude": ",".join(f"{points[i][1]:.4f}" for i in chunk),
                  "start_date": START, "end_date": END, "daily": ",".join(DAILY), "timezone": TZ}
        if model:
            params["models"] = model
        frames.append(_daily_frames(get_json(ARCHIVE, params, timeout=300), chunk))
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df.date)
    return df


BOROUGH_POINTS = {"Manhattan": (40.7831, -73.9712), "Bronx": (40.8448, -73.8648), "Brooklyn": (40.6782, -73.9442),
                  "Queens": (40.7282, -73.7949), "Staten Island": (40.5795, -74.1502)}


def _query_season(points: dict[str, tuple[float, float]], model: str | None, years=range(2005, 2027)) -> pd.DataFrame:
    """April-September of each year (keeps 30-day look-backs for heat indices inside the window)."""
    ids = list(points)
    frames = []
    for y in years:
        params = {"latitude": ",".join(f"{points[i][0]:.4f}" for i in ids),
                  "longitude": ",".join(f"{points[i][1]:.4f}" for i in ids),
                  "start_date": f"{y}-04-01", "end_date": min(f"{y}-09-30", END), "daily": ",".join(DAILY), "timezone": TZ}
        if model:
            params["models"] = model
        frames.append(_daily_frames(get_json(ARCHIVE, params, timeout=300), ids))
        time.sleep(2.0)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df.date)
    return df


def borough_daily(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "weather_borough_daily.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    df = _query_season(BOROUGH_POINTS, "era5_land")
    df.to_parquet(out, index=False)
    print(f"[ingest:weather] boroughs x {df.date.nunique()} days -> {out}")
    return df


def upstream_daily(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "weather_upstream_daily.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    df = _query_daily(UPSTREAM, None)
    df.to_parquet(out, index=False)
    print(f"[ingest:weather] upstream cities {list(UPSTREAM)} -> {out}")
    return df


def hourly_central(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "weather_central_park_hourly.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    lat, lon = UPSTREAM["central_park"]
    frames = []
    for y in range(2005, 2027):
        r = get_json(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": f"{y}-04-01",
                               "end_date": min(f"{y}-09-30", END), "hourly": ",".join(HOURLY), "timezone": TZ}, timeout=300)
        frames.append(pd.DataFrame(r["hourly"]))
        time.sleep(2.0)
    df = pd.concat(frames, ignore_index=True).rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df.datetime)
    df.to_parquet(out, index=False)
    print(f"[ingest:weather] Central Park hourly {len(df):,} rows -> {out}")
    return df


def forecasts(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    """Daily max/mean of archived lead-1 and lead-3 hourly forecasts at Central Park (2022-01 .. 2026-09)."""
    out = paths.raw / "weather_forecasts_central_park.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    lat, lon = UPSTREAM["central_park"]
    vars_ = [f"{v}_previous_day{k}" for v in ("temperature_2m", "apparent_temperature", "relative_humidity_2m") for k in (1, 3)]
    r = get_json(PREVIOUS_RUNS, {"latitude": lat, "longitude": lon, "start_date": "2022-01-01", "end_date": END,
                                 "hourly": ",".join(vars_), "timezone": TZ}, timeout=300)
    h = pd.DataFrame(r["hourly"]).rename(columns={"time": "datetime"})
    h["date"] = pd.to_datetime(h.datetime).dt.normalize()
    agg = {}
    for k in (1, 3):
        agg[f"tmax_f{k}"] = (f"temperature_2m_previous_day{k}", "max")
        agg[f"tmin_f{k}"] = (f"temperature_2m_previous_day{k}", "min")
        agg[f"tmean_f{k}"] = (f"temperature_2m_previous_day{k}", "mean")
        agg[f"atmax_f{k}"] = (f"apparent_temperature_previous_day{k}", "max")
        agg[f"rh_f{k}"] = (f"relative_humidity_2m_previous_day{k}", "mean")
    df = h.groupby("date").agg(**agg).reset_index()
    df.to_parquet(out, index=False)
    print(f"[ingest:weather] archived forecasts {df.date.min():%Y-%m-%d}..{df.date.max():%Y-%m-%d} -> {out}")
    return df


def fetch_all(paths: Paths = PATHS, refresh: bool = False):
    upstream_daily(paths, refresh)
    forecasts(paths, refresh)
    hourly_central(paths, refresh)
    borough_daily(paths, refresh)
