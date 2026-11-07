"""MODZCTA x day modelling panel (May-September, 2005-2026) and the citywide daily series.

Row = (modzcta, date). Columns:
    y                     EMS heat calls in the area that day (label)
    static features       see features.static (joined by modzcta)
    city weather          Central Park heat features (features.weather)
    local anomaly         borough apparent max minus Central Park (ERA5-Land), if downloaded
    upstream              apparent max yesterday in Newark, Trenton, Philadelphia, Baltimore, Washington DC and the
                          south-west gradient (upstream mean - NYC), i.e. "is heat on its way?"
    calendar              weekend, federal holiday, 4 July window, school break, COVID years, year
    lead-lag (known d-1)  yesterday's citywide and borough calls, citywide syndromic ED visits
Outputs: data/processed/panel.parquet, data/processed/city_daily.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from ..ingest import context, ems, weather as wx
from ..ingest.geography import load_modzcta, zip_to_modzcta
from .calendar import calendar_features
from .weather import heat_features, openmeteo_daily

SEASON_MONTHS = (5, 6, 7, 8, 9)
ZIP_FIXES = {"11249": "11211"}  # Williamsburg ZIP created after the MODZCTA vintage
UPSTREAM = ["newark", "trenton", "philadelphia", "baltimore", "washington_dc"]
CLIMATOLOGY_YEARS = (2005, 2019)  # fixed reference period for ehi_sig (no look-ahead into test years)


def city_weather(paths: Paths = PATHS) -> pd.DataFrame:
    up = wx.upstream_daily(paths)
    cp = openmeteo_daily(up, "central_park")
    clim = cp[cp.date.dt.year.between(*CLIMATOLOGY_YEARS) & cp.date.dt.month.isin(SEASON_MONTHS)]
    t95 = float(np.quantile(clim.tmean, 0.95))
    f = heat_features(cp, t95)
    for city in UPSTREAM:
        u = openmeteo_daily(up, city).set_index("date").atmax
        f[f"up_{city}_atmax_l1"] = f.date.map(u.shift(1))
    f["up_sw_gradient_l1"] = f[[f"up_{c}_atmax_l1" for c in ("philadelphia", "baltimore", "washington_dc")]].mean(1) - f.atmax_l1
    f.attrs["t95"] = t95
    return f


def load_ems(paths: Paths = PATHS) -> pd.DataFrame:
    """EMS heat calls in season with date and MODZCTA (NaN when the ZIP cannot be mapped)."""
    zmap = zip_to_modzcta(load_modzcta(paths))
    e = ems.fetch(paths)
    e["date"] = e.incident_datetime.dt.normalize()
    e["modzcta"] = e.zipcode.astype("string").str[:5].replace(ZIP_FIXES).map(zmap)
    return e[e.date.dt.month.isin(SEASON_MONTHS)].copy()


def build_city(paths: Paths = PATHS) -> pd.DataFrame:
    """Citywide daily series: heat features, upstream, calendar, EMS total (+ lag), syndromic ED (+ lag)."""
    e = load_ems(paths)
    W = city_weather(paths)
    W = W[W.date.dt.month.isin(SEASON_MONTHS) & (W.date <= e.date.max())].reset_index(drop=True)
    city = W.join(calendar_features(W.date))
    counts = e.groupby("date").size()
    city["city_y"] = city.date.map(counts).fillna(0)
    city["city_y_l1"] = city.date.map(counts.reindex(counts.index.union(city.date)).fillna(0).shift(1, freq="D")).fillna(0)
    ed = context.syndromic(paths).set_index("date").heat_ed_visits
    city["ed_visits"] = city.date.map(ed)
    city["ed_visits_l1"] = city.date.map(ed.shift(1, freq="D"))
    city["censored_peak"] = (city.city_y >= 150).astype(int)
    city.to_parquet(paths.processed / "city_daily.parquet", index=False)
    print(f"[features:city] {len(city)} days, {city.city_y.sum():,.0f} calls -> {paths.processed / 'city_daily.parquet'}")
    return city


def build(paths: Paths = PATHS) -> pd.DataFrame:
    S = pd.read_csv(paths.processed / "modzcta_static.csv", dtype={"modzcta": str})
    e = load_ems(paths)
    borough_of = e.dropna(subset=["modzcta"]).groupby("modzcta").borough.agg(lambda s: s.mode().iloc[0])
    city = build_city(paths)

    # area x day
    areas = S.modzcta.to_numpy()
    P = pd.DataFrame({"modzcta": np.repeat(areas, len(city)), "date": np.tile(city.date.to_numpy(), len(areas))})
    y = e.dropna(subset=["modzcta"]).groupby(["modzcta", "date"]).size().rename("y")
    P = P.merge(y, on=["modzcta", "date"], how="left").fillna({"y": 0})
    P["borough"] = P.modzcta.map(borough_of).fillna("UNKNOWN")
    bor = e.groupby(["borough", "date"]).size().rename("borough_y").reset_index()
    bor["date"] = bor.date + pd.Timedelta(days=1)
    P = P.merge(bor.rename(columns={"borough_y": "borough_y_l1"}), on=["borough", "date"], how="left").fillna({"borough_y_l1": 0})
    if (paths.raw / "weather_borough_daily.parquet").exists():  # local apparent-temperature anomaly (ERA5-Land)
        bd = wx.borough_daily(paths)
        bd = bd.rename(columns={"apparent_temperature_max": "b_atmax"})[["loc_id", "date", "b_atmax"]]
        bd["borough"] = bd.loc_id.str.upper().replace({"STATEN ISLAND": "RICHMOND / STATEN ISLAND"})
        P = P.merge(bd[["borough", "date", "b_atmax"]], on=["borough", "date"], how="left")
        P = P.merge(city[["date", "atmax"]], on="date", how="left")
        P["local_atmax_anomaly"] = P.b_atmax - P.atmax
        P = P.drop(columns=["b_atmax", "atmax"])
    else:
        print("[features:panel] borough weather not downloaded yet; skipping local anomaly")
    P = P.merge(city.drop(columns=["city_y"]), on="date", how="left").merge(S, on="modzcta", how="left")
    P = P.sort_values(["modzcta", "date"]).reset_index(drop=True)
    out = paths.processed / "panel.parquet"
    P.to_parquet(out, index=False)
    print(f"[features:panel] {len(P):,} rows ({len(areas)} areas x {len(city)} days), {P.y.sum():,.0f} calls "
          f"of {city.city_y.sum():,.0f} citywide -> {out}")
    return P
