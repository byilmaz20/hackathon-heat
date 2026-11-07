"""Daytime presence - where people *are* during the day (the EDA shows calls follow daytime/visitor activity).

jobs        workplace jobs per ZCTA (LEHD LODES 8 WAC, all jobs, 2021) - census-block counts summed via the LODES
            block-to-ZCTA crosswalk
subway      mean daily subway entries per station complex in June-August 2023-2025 (MTA monthly ridership),
            with station coordinates
"""
from __future__ import annotations

import gzip
import io

import pandas as pd

from ..config import PATHS, Paths
from .http import get, socrata

LODES = "https://lehd.ces.census.gov/data/lodes/LODES8/ny/"
LODES_YEAR = 2021


def jobs(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / f"lodes{LODES_YEAR}_jobs_zcta_ny.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, dtype={"zcta": str})
    wac = pd.read_csv(io.BytesIO(gzip.decompress(get(f"{LODES}wac/ny_wac_S000_JT00_{LODES_YEAR}.csv.gz", timeout=600))),
                      usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
    xw = pd.read_csv(io.BytesIO(gzip.decompress(get(f"{LODES}ny_xwalk.csv.gz", timeout=600))),
                     usecols=["tabblk2020", "zcta"], dtype={"tabblk2020": str, "zcta": str})
    df = wac.merge(xw, left_on="w_geocode", right_on="tabblk2020").groupby("zcta", as_index=False).C000.sum()
    df = df.rename(columns={"C000": "jobs"})
    df.to_csv(out, index=False)
    print(f"[ingest:lodes] {df.jobs.sum():,.0f} jobs in {len(df)} ZCTAs -> {out}")
    return df


def subway(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    """Mean daily entries in June-August 2023-2025 per station complex (MTA monthly station ridership, ak4z-sape)."""
    out = paths.raw / "mta_subway_summer_station_ridership.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out)
    q = {"$select": "station_complex_id, station_complex, avg(latitude) as lat, avg(longitude) as lon, "
                    "sum(ridership) as entries, count(*) as months",
         "$where": "date_extract_m(month) in (6, 7, 8) and date_extract_y(month) between 2023 and 2025",
         "$group": "station_complex_id, station_complex", "$order": "station_complex_id"}
    df = pd.DataFrame(socrata("ak4z-sape", q, domain="data.ny.gov"))
    df[["lat", "lon", "entries", "months"]] = df[["lat", "lon", "entries", "months"]].apply(pd.to_numeric)
    df["summer_entries_per_day"] = df.entries / (df.months * 30.67)
    df = df[["station_complex_id", "station_complex", "lat", "lon", "summer_entries_per_day"]]
    df.to_csv(out, index=False)
    print(f"[ingest:mta] {len(df)} station complexes -> {out}")
    return df
