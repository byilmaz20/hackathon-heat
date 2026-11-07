"""Static (slowly varying) features per MODZCTA - built once from the raw downloads.

Exposure / presence  pop_est, density, jobs (LEHD), summer subway entries per day (MTA), cooling sites
Vulnerability        poverty, older adults living alone, Black share, household AC, HVI (1-5), ACS extras if present
Urban heat / green   daytime summer surface temperature, vegetative cover, tree canopy, park share
Indicators published for NTA2010 / UHF42 polygons are moved to MODZCTA by area weighting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..config import PATHS, Paths
from ..ingest import activity, context, demographics
from ..ingest.geography import area_weights, load_modzcta, load_polygons


def _points_in(Z: pd.DataFrame, lat, lon, values: dict[str, np.ndarray]) -> pd.DataFrame:
    """Sum point values (e.g. station entries) inside each MODZCTA polygon."""
    pts = [Point(x, y) for x, y in zip(lon, lat)]
    out = {k: np.zeros(len(Z)) for k in values}
    count = np.zeros(len(Z))
    for zi, g in enumerate(Z.geometry):
        inside = np.array([g.contains(p) for p in pts])
        count[zi] = inside.sum()
        for k, v in values.items():
            out[k][zi] = np.nansum(np.asarray(v)[inside])
    return pd.DataFrame({"modzcta": Z.modzcta, "n_points": count, **out})


def build(paths: Paths = PATHS) -> pd.DataFrame:
    Z = load_modzcta(paths)
    S = Z[["modzcta", "lat", "lon", "area_km2", "pop_est"]].copy()
    S["pop_density"] = S.pop_est / S.area_km2
    member = Z[["modzcta", "zips"]].explode("zips").rename(columns={"zips": "zcta"})

    # EH Data Portal indicators -> area-weighted mean over the polygons covering each MODZCTA
    ind = context.indicators(paths)
    for geo in ind.geo_type.unique():
        w = area_weights(Z, load_polygons(geo, paths))
        wide = ind[ind.geo_type == geo].pivot_table(index="geo_id", columns="variable", values="value")
        m = w.merge(wide, left_on="geo_id", right_index=True)
        for v in wide.columns:
            ok = m[v].notna()
            S = S.merge((m[ok].assign(x=m[v] * m.weight).groupby("modzcta").x.sum()
                         / m[ok].groupby("modzcta").weight.sum()).rename(v), on="modzcta", how="left")

    hvi = member.merge(context.hvi(paths), on="zcta").groupby("modzcta").hvi.mean()
    S = S.merge(hvi, on="modzcta", how="left")
    jobs = member.merge(activity.jobs(paths), on="zcta").groupby("modzcta").jobs.sum()
    S = S.merge(jobs, on="modzcta", how="left").fillna({"jobs": 0})
    S["jobs_density"] = S.jobs / S.area_km2
    sw = activity.subway(paths)
    sub = _points_in(Z, sw.lat, sw.lon, {"subway_entries": sw.summer_entries_per_day})
    S = S.merge(sub.rename(columns={"n_points": "n_subway_stations"}), on="modzcta")
    cs = context.cooling_sites(paths)
    S = S.merge(_points_in(Z, cs.lat, cs.lon, {}).rename(columns={"n_points": "n_cool_it_sites"}), on="modzcta")
    S = S.merge(demographics.park_share(paths), on="modzcta", how="left")
    acs = demographics.acs(paths)
    if acs is not None:
        a = member.merge(acs, on="zcta").groupby("modzcta").sum(numeric_only=True)
        S = S.merge(pd.DataFrame({"pop_65plus_pct": 100 * a.pop_65plus / a.pop_total,
                                  "hh_65plus_alone_pct": 100 * a.hh_65plus_alone / a.households}), on="modzcta", how="left")
    # presence proxy: residents + workers + visitors (subway entries) per km2
    S["daytime_presence"] = S.pop_est + S.jobs + S.subway_entries
    S["log_daytime_presence"] = np.log1p(S.daytime_presence)
    out = paths.processed / "modzcta_static.csv"
    S.to_csv(out, index=False)
    print(f"[features:static] {len(S)} areas x {S.shape[1]} columns -> {out}")
    return S
