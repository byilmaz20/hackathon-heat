"""Population and vulnerability by ZCTA (US Census ACS 5-year), and park land (NYC Parks Properties).

ACS variables (aggregated to MODZCTA in the feature step):
    pop_total            B01003_001E
    pop_65plus           sum of B01001 age-sex cells 65+ (male 020-025, female 044-049)
    households           B11007_001E
    hh_65plus_alone      B11007_003E  one-person households aged 65+
    median_income        B19013_001E
    black_nh, hispanic   B03002_004E, B03002_012E (components of the NYC Heat Vulnerability Index)
    disability_pct       S1810_C03_001E
    poverty_pct          S1701_C03_001E
"""
from __future__ import annotations

import json
import os

import pandas as pd
from shapely.geometry import shape

from ..config import PATHS, Paths
from .geography import _to_km, load_modzcta
from .http import get, get_json

ACS_YEAR = 2022
DETAIL = {"B01003_001E": "pop_total", "B11007_001E": "households", "B11007_003E": "hh_65plus_alone",
          "B19013_001E": "median_income", "B03002_001E": "pop_race_base", "B03002_004E": "black_nh", "B03002_012E": "hispanic"}
AGE65 = [f"B01001_{i:03d}E" for i in list(range(20, 26)) + list(range(44, 50))]
SUBJECT = {"S1810_C03_001E": "disability_pct", "S1701_C03_001E": "poverty_pct"}
ZCTA = "zip code tabulation area"


def _acs(table_url: str, variables: list[str]) -> pd.DataFrame:
    params = {"get": ",".join(variables), "for": f"{ZCTA}:*"}
    if os.environ.get("CENSUS_API_KEY"):
        params["key"] = os.environ["CENSUS_API_KEY"]
    rows = get_json(table_url, params, timeout=300)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.rename(columns={ZCTA: "zcta"})


def acs(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame | None:
    """Needs a free Census API key in the CENSUS_API_KEY environment variable; returns None without one."""
    out = paths.raw / f"acs{ACS_YEAR}_zcta_nyc.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, dtype={"zcta": str})
    if not os.environ.get("CENSUS_API_KEY"):
        print("[ingest:acs] skipped - set CENSUS_API_KEY (free: https://api.census.gov/data/key_signup.html)")
        return None
    nyc = {z for zs in load_modzcta(paths).zips for z in zs}
    base = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
    d = _acs(base, list(DETAIL) + AGE65)
    s = _acs(base + "/subject", list(SUBJECT))
    df = d.merge(s, on="zcta", how="left")
    df = df[df.zcta.isin(nyc)].copy()
    num = [c for c in df.columns if c != "zcta"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df[df[num] < -1e5] = None  # Census missing-value sentinels (-666666666 etc.)
    df["pop_65plus"] = df[AGE65].sum(axis=1)
    df = df.drop(columns=AGE65).rename(columns={**DETAIL, **SUBJECT})
    df.to_csv(out, index=False)
    print(f"[ingest:acs] {len(df)} NYC ZCTAs -> {out}")
    return df


def park_share(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    """Share of each MODZCTA's land area that is NYC Parks property, and distance-relevant park count."""
    out = paths.raw / "nyc_parks_share_modzcta.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, dtype={"modzcta": str})
    src = paths.raw / "nyc_parks_properties.geojson"
    if refresh or not src.exists():
        src.write_bytes(get("https://data.cityofnewyork.us/api/geospatial/enfh-gkve?method=export&format=GeoJSON", timeout=600))
    gj = json.loads(src.read_text())
    parks = []
    for f in gj["features"]:
        if f.get("geometry"):
            try:
                parks.append(_to_km(shape(f["geometry"]).buffer(0)))
            except Exception:
                continue
    Z = load_modzcta(paths)
    rows = []
    for r in Z.itertuples():
        zg = _to_km(r.geometry)
        inter = [p for p in parks if p.intersects(zg)]
        area = sum(zg.intersection(p).area for p in inter)
        rows.append(dict(modzcta=r.modzcta, park_area_km2=area, park_share=area / zg.area, n_parks=len(inter)))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"[ingest:parks] {len(parks)} park polygons -> {out}")
    return df
