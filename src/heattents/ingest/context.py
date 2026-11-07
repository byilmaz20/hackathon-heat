"""Neighbourhood context and secondary labels from the NYC Environment & Health Data Portal and NYC Open Data.

indicators   static heat-vulnerability features (surface temperature, vegetation, tree canopy, home AC)
syndromic    citywide daily heat-syndrome ED visits (100% of NYC EDs since 2016) - a second, independent label
hvi          Heat Vulnerability Index (1-5) by ZCTA
cooling      "Cool It! NYC" cooling sites (existing supply)
"""
from __future__ import annotations

import io
import json

import pandas as pd

from ..config import PATHS, Paths
from .http import get, socrata

EHDP = "https://raw.githubusercontent.com/nychealth/EHDP-data/production/"
# indicator id -> (column name, measure id); the finest geography we have polygons for is used
INDICATORS = {
    2141: ("surface_temp_f", 688),
    2143: ("vegetative_cover_pct", 690),
    2157: ("tree_canopy_pct", 708),
    2185: ("households_with_ac_pct", 781),
    103: ("poverty_pct", 221),
    2146: ("older_living_alone_pct", 693),
    2325: ("black_pct", 988),
}
GEO_PREFERENCE = ["NTA2010", "UHF42", "CD"]  # polygons available in ingest.geography


def indicators(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    """Long table: variable, geo_type, geo_id, value (latest time period at the finest available geography)."""
    out = paths.raw / "ehdp_indicators.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, dtype={"geo_id": str})
    frames = []
    for iid, (name, measure) in INDICATORS.items():
        d = pd.DataFrame(json.loads(get(f"{EHDP}indicators/data/{iid}.json")))
        d = d[d.MeasureID == measure]
        geo = next(g for g in GEO_PREFERENCE if (d.GeoType == g).any())
        d = d[d.GeoType == geo]
        d = d[d.TimePeriodID == d.TimePeriodID.max()]
        frames.append(pd.DataFrame({"variable": name, "geo_type": geo, "geo_id": d.GeoID.astype(str),
                                    "value": pd.to_numeric(d.Value, errors="coerce"), "time_period_id": d.TimePeriodID}))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    return df


def syndromic(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    """Citywide daily heat-syndrome ED visits with the LaGuardia max temperature / heat index (F)."""
    out = paths.raw / "nyc_heat_syndrome_ed_daily.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, parse_dates=["date"])
    base = EHDP + "key-topics/heat-syndrome/"
    files = ["previous_years.csv", "edheat1720_supp.csv"] + [f"edheat{y}_live.csv" for y in range(2021, 2027)]
    frames = []
    for f in files:
        d = pd.read_csv(io.BytesIO(get(base + f)))
        d.columns = [c.strip().upper() for c in d.columns]
        date_col = next(c for c in d.columns if "DATE" in c)
        frames.append(pd.DataFrame({"date": pd.to_datetime(d[date_col]), "heat_ed_visits": d.filter(like="COUNT").iloc[:, 0],
                                    "lga_max_temp_f": d.filter(like="TEMP").iloc[:, 0], "source": f}))
    df = pd.concat(frames).sort_values(["date", "source"]).drop_duplicates("date", keep="last").reset_index(drop=True)
    df.to_csv(out, index=False)
    return df


def hvi(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "nyc_hvi_zcta.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, dtype={"zcta": str})
    df = pd.DataFrame(socrata("4mhf-duep", {"$order": "zcta20"})).rename(columns={"zcta20": "zcta"})
    df["hvi"] = pd.to_numeric(df.hvi)
    df.to_csv(out, index=False)
    return df


def cooling_sites(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "nyc_cool_it_sites_2020.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out)
    df = pd.DataFrame(socrata("h2bn-gu9k", {"$order": "propertyname"}))
    df = df.rename(columns={"x": "lon", "y": "lat"})
    df[["lon", "lat"]] = df[["lon", "lat"]].apply(pd.to_numeric)
    df.to_csv(out, index=False)
    return df
