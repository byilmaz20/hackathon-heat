"""Population by ZCTA and year (US Census ACS 5-year), and park land (NYC Parks Properties).

ACS 5-year estimates by ZIP code tabulation area exist from 2011. The estimate labelled year Y covers the survey years
Y-4..Y; we use it as the population of year Y. Variables:
    pop_total    B01003_001E
    pop_65plus   sum of the B01001 age-sex cells aged 65+ (male 020-025, female 044-049)
Needs a free Census API key in the CENSUS_API_KEY environment variable (https://api.census.gov/data/key_signup.html).
2011-2019 must be requested within a state (in=state:36); from 2020 ZCTAs are not nested in states.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from shapely.geometry import shape

from ..config import PATHS, Paths
from .geography import _to_km, load_modzcta
from .http import get, get_json

ACS_YEARS = range(2011, 2025)
AGE65 = [f"B01001_{i:03d}E" for i in list(range(20, 26)) + list(range(44, 50))]
ZCTA = "zip code tabulation area"


def _acs_year(year: int, variables: list[str]) -> pd.DataFrame:
    params = {"get": ",".join(variables), "for": f"{ZCTA}:*", "key": os.environ["CENSUS_API_KEY"]}
    if year <= 2019:
        params["in"] = "state:36"
    rows = get_json(f"https://api.census.gov/data/{year}/acs/acs5", params, timeout=300)
    return pd.DataFrame(rows[1:], columns=rows[0]).rename(columns={ZCTA: "zcta"})


def acs(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame | None:
    """Long table zcta, year, pop_total, pop_65plus for NYC ZCTAs; None without a Census key."""
    out = paths.raw / "acs5_zcta_nyc_by_year.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, dtype={"zcta": str})
    if not os.environ.get("CENSUS_API_KEY"):
        print("[ingest:acs] skipped - set CENSUS_API_KEY (free: https://api.census.gov/data/key_signup.html)")
        return None
    nyc = {z for zs in load_modzcta(paths).zips for z in zs}
    frames = []
    for year in ACS_YEARS:
        try:
            d = _acs_year(year, ["B01003_001E"] + AGE65)
        except Exception as err:  # year not yet published
            print(f"[ingest:acs] {year}: unavailable ({err})")
            continue
        d = d[d.zcta.isin(nyc)].copy()
        num = ["B01003_001E"] + AGE65
        d[num] = d[num].apply(pd.to_numeric, errors="coerce").where(lambda v: v >= 0)  # negative = Census missing code
        frames.append(pd.DataFrame({"zcta": d.zcta, "year": year, "pop_total": d.B01003_001E,
                                    "pop_65plus": d[AGE65].sum(axis=1, min_count=1)}))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[ingest:acs] {df.zcta.nunique()} NYC ZCTAs x {df.year.nunique()} years ({df.year.min()}-{df.year.max()}) -> {out}")
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
