r"""Demand scenarios for the deployment optimiser: coherent city -> area -> hour samples.

For scenario s on day t:
    N^s_t      ~ citywide predictive distribution (city model; NB + conformal / forecast-error uncertainty)
    D^s_{i,h}  = N^s_t * m * pi_{it} * phi_{b(i), weekend(t), h}
pi_{it}  area shares (beach & event share model, models.context_share)
m        share of calls with a mappable location (calibrated on history)
phi      hour-of-day profile by borough and weekend (empirical; the EDA shows it does not depend on temperature).
Each area's demand is spread over a few points inside its polygon so that walking catchments are realistic.
Output columns match heattents.deploy: [area_id, lat, lon, hour, scenario, expected_cases].
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..config import PATHS, Paths
from ..features.panel import load_ems
from ..ingest.geography import load_modzcta
from .context_share import ContextShare, prepare


def hour_profile(paths: Paths = PATHS, smoothing: float = 5.0) -> pd.DataFrame:
    """phi[borough, weekend, hour] (rows sum to 1), Laplace-smoothed towards the citywide profile."""
    e = load_ems(paths).dropna(subset=["modzcta"])
    e["hour"] = e.incident_datetime.dt.hour
    e["weekend"] = (e.incident_datetime.dt.day_of_week >= 5).astype(int)
    city = e.groupby("hour").size().reindex(range(24), fill_value=0)
    city = city / city.sum()
    counts = e.groupby(["borough", "weekend", "hour"]).size().unstack("hour", fill_value=0).reindex(columns=range(24), fill_value=0)
    prof = (counts + smoothing * 24 * city.to_numpy()) .div(counts.sum(1) + smoothing * 24, axis=0)
    return prof


def interior_points(geom, k: int) -> list[tuple[float, float]]:
    """Up to k well-spread points inside a polygon (grid points, thinned), always at least one."""
    minx, miny, maxx, maxy = geom.bounds
    n = 12
    grid = [Point(x, y) for x in np.linspace(minx, maxx, n) for y in np.linspace(miny, maxy, n)]
    inside = [p for p in grid if geom.contains(p)] or [geom.representative_point()]
    idx = np.linspace(0, len(inside) - 1, min(k, len(inside))).round().astype(int)
    return [(inside[i].y, inside[i].x) for i in sorted(set(idx))]


@lru_cache(maxsize=4)
def demand_points(points_per_area: int = 4) -> pd.DataFrame:
    Z = load_modzcta()
    rows = []
    for r in Z.itertuples():
        pts = interior_points(r.geometry, points_per_area)
        for j, (lat, lon) in enumerate(pts):
            rows.append(dict(modzcta=r.modzcta, area_id=f"{r.modzcta}#{j}", lat=lat, lon=lon, weight=1 / len(pts)))
    return pd.DataFrame(rows)


def build_demand(area_rows: pd.DataFrame, totals: np.ndarray, area_model: ContextShare, profile: pd.DataFrame,
                 mapped_fraction: float, points_per_area: int = 4, min_cases: float = 1e-3) -> pd.DataFrame:
    """area_rows: one day's panel rows (all 177 areas, incl. borough & weekend); totals: (n_scenarios,) city totals."""
    shares = area_model.shares(area_rows)
    weekend = int(area_rows.weekend.iloc[0])
    phi = np.vstack([profile.loc[(b, weekend)].to_numpy() if (b, weekend) in profile.index else profile.mean().to_numpy()
                     for b in area_rows.borough])  # (areas, 24)
    base = shares[:, None] * phi * mapped_fraction  # (areas, 24), per unit of city total
    pts = demand_points(points_per_area).merge(pd.DataFrame({"modzcta": area_rows.modzcta.to_numpy(),
                                                             "row": np.arange(len(area_rows))}), on="modzcta")
    out = []
    for s, N in enumerate(totals):
        exp_cases = N * base[pts.row.to_numpy()] * pts.weight.to_numpy()[:, None]  # (points, 24)
        df = pd.DataFrame(exp_cases, columns=range(24))
        df[["area_id", "lat", "lon"]] = pts[["area_id", "lat", "lon"]].to_numpy()
        long = df.melt(id_vars=["area_id", "lat", "lon"], var_name="hour", value_name="expected_cases")
        long["scenario"] = s
        out.append(long[long.expected_cases > min_cases])
    D = pd.concat(out, ignore_index=True)
    D["hour"] = D.hour.astype(int)
    D[["lat", "lon"]] = D[["lat", "lon"]].astype(float)
    return D[["area_id", "lat", "lon", "hour", "scenario", "expected_cases"]]


def mapped_fraction(paths: Paths = PATHS) -> float:
    e = load_ems(paths)
    return float(e.modzcta.notna().mean())


def area_rows_for_date(date, paths: Paths = PATHS) -> pd.DataFrame:
    P = pd.read_parquet(paths.processed / "panel.parquet", filters=[("date", "==", pd.Timestamp(date))])
    return prepare(P)


def stratified_totals(samples: np.ndarray, k: int) -> np.ndarray:
    """k equally weighted representative totals: the (i + 0.5) / k quantiles of the predictive samples.

    Scenarios differ only in the citywide level (the spatial-temporal pattern is shared), so stratified
    quantiles are a low-variance substitute for k i.i.d. draws in the sample-average approximation.
    """
    return np.quantile(np.asarray(samples, float), (np.arange(k) + 0.5) / k)
