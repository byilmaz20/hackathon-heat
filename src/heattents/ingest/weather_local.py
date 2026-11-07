"""Neighbourhood-level weather from NOAA's HRRR model (3 km grid) via Open-Meteo, at the 42 UHF42 neighbourhood centres.

estimates   Open-Meteo Historical Forecast API, model ncep_hrrr_conus: the stitched short-range HRRR series, i.e. the
            model's best estimate of each hour's weather. Available from 2018. Used as "observed" local weather for
            training (the city-wide ERA5 series cannot see differences between neighbourhoods; ERA5-Land, at ~9 km,
            puts the 42 neighbourhoods in 13 cells and has no heat index).
forecasts   Open-Meteo Previous Runs API, model ncep_hrrr_conus, variables *_previous_day1: the forecast for each hour
            issued the day before. Available from 2024. Used for honest next-day testing.
Both are aggregated per neighbourhood and day (New York time): max temperature, max apparent temperature (heat index),
min temperature, mean relative humidity. May-September only.
Probe (20-26 June 2025): HRRR puts the 42 centres in 38 grid cells; the hottest-minus-coolest neighbourhood
difference in daily max heat index reached 8.8 C (ERA5-Land: 4.8 C in plain temperature).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .geography import load_polygons
from .http import get_json

HIST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ncep_hrrr_conus"
TZ = "America/New_York"
VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m"]


def centroids(paths: Paths = PATHS) -> pd.DataFrame:
    f = paths.raw / "uhf42_centroids.csv"
    if not f.exists():
        P = load_polygons("UHF42", paths)
        P = P[P.geo_id != "9999"]
        pts = [g.representative_point() for g in P.geometry]
        pd.DataFrame({"uhf": P.geo_id.to_numpy(), "lat": [p.y for p in pts], "lon": [p.x for p in pts]}).to_csv(f, index=False)
    return pd.read_csv(f, dtype={"uhf": str})


def _daily(resp: list, uhfs, suffix: str = "") -> pd.DataFrame:
    frames = []
    for u, r in zip(uhfs, resp):
        h = pd.DataFrame(r["hourly"])
        h["time"] = pd.to_datetime(h.time)
        h["date"] = h.time.dt.normalize()
        g = h.groupby("date").agg(tmax=(f"temperature_2m{suffix}", "max"), tmin=(f"temperature_2m{suffix}", "min"),
                                  atmax=(f"apparent_temperature{suffix}", "max"), rh=(f"relative_humidity_2m{suffix}", "mean"))
        frames.append(g.reset_index().assign(uhf=u))
    return pd.concat(frames, ignore_index=True)


def _fetch(url: str, years, suffix: str, paths: Paths) -> pd.DataFrame:
    """One summer at a time; finished summers are kept in a .partial file so a quota stop can resume."""
    c = centroids(paths)
    part = paths.raw / f"_hrrr_{'prev' if 'previous' in url else 'hist'}.partial.parquet"
    done = pd.read_parquet(part) if part.exists() else pd.DataFrame()
    params = {"latitude": ",".join(f"{x:.4f}" for x in c.lat), "longitude": ",".join(f"{x:.4f}" for x in c.lon),
              "hourly": ",".join(v + suffix for v in VARS), "models": MODEL, "timezone": TZ}
    frames = [done] if len(done) else []
    for y in years:
        if len(done) and (pd.to_datetime(done.date).dt.year == y).any():
            continue
        for start, end in ((f"{y}-05-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-09-30")):  # two requests per summer
            r = get_json(url, {**params, "start_date": start, "end_date": min(end, "2026-09-30")}, timeout=600)
            frames.append(_daily(r, c.uhf.to_numpy(), suffix))
            time.sleep(3)
        pd.concat(frames, ignore_index=True).to_parquet(part, index=False)
        print(f"[ingest:hrrr] {url.split('//')[1].split('-api')[0]} {y} done", flush=True)
    return pd.concat(frames, ignore_index=True)


def estimates(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "weather_uhf42_hrrr_daily.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    df = _fetch(HIST, range(2018, 2027), "", paths)
    df.to_parquet(out, index=False)
    return df


def forecasts(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "weather_uhf42_hrrr_forecast_day1.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    df = _fetch(PREV, range(2024, 2027), "_previous_day1", paths)
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    f = forecasts()
    e = estimates()
    for name, d in (("estimates", e), ("forecasts", f)):
        print(name, len(d), d.date.min(), d.date.max(), "missing atmax", int(d.atmax.isna().sum()))
