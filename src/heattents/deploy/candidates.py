"""Candidate tent sites: NYC Parks properties where a pop-up tent can stand, pre-screened by reachable demand.

Every usable park property (plazas, playgrounds, neighbourhood / community / flagship parks, malls, waterfront)
contributes one point. Large parks contribute up to a few points spread inside the polygon, because the
entrance used matters more than the centroid for a 1 km walk. Cemeteries, gardens, nature areas, parkways,
operations sites and undeveloped lots are excluded. To keep the MILP small, only the ``top_n`` sites that can
reach the most expected demand within the walking radius are passed to the optimiser.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from shapely.geometry import shape

from ..config import PATHS, Paths
from ..models.scenarios import interior_points
from .model import Site

USABLE = {"Triangle/Plaza", "Neighborhood Park", "Jointly Operated Playground", "Playground", "Community Park",
          "Recreational Field/Courts", "Mall", "Waterfront Facility", "Flagship Park", "Historic House Park"}


def park_points(paths: Paths = PATHS, min_acres: float = 0.05) -> pd.DataFrame:
    out = paths.processed / "candidate_park_points.csv"
    if out.exists():
        return pd.read_csv(out)
    gj = json.loads((paths.raw / "nyc_parks_properties.geojson").read_text())
    rows = []
    for f in gj["features"]:
        p = f["properties"]
        if p.get("typecategory") not in USABLE or not f.get("geometry"):
            continue
        acres = float(p.get("acres") or 0)
        if acres < min_acres:
            continue
        geom = shape(f["geometry"]).buffer(0)
        k = 1 if acres < 20 else 3 if acres < 200 else 6
        for j, (lat, lon) in enumerate(interior_points(geom, k)):
            rows.append(dict(site_id=f"{p.get('gispropnum')}#{j}", name=p.get("signname") or p.get("name311") or "Park",
                             kind=p.get("typecategory"), acres=acres, borough=p.get("borough"), lat=lat, lon=lon))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def screen(points: pd.DataFrame, demand: pd.DataFrame, radius_km: float, top_n: int = 250) -> pd.DataFrame:
    """Keep the top_n candidates by expected demand within radius (mean over scenarios), one per 150 m cluster."""
    d = demand.groupby(["area_id", "lat", "lon"], as_index=False).expected_cases.sum()
    d["expected_cases"] /= demand.scenario.nunique()
    k = np.cos(np.radians(40.7))
    dist = np.hypot((points.lat.to_numpy()[:, None] - d.lat.to_numpy()[None]) * 110.57,
                    (points.lon.to_numpy()[:, None] - d.lon.to_numpy()[None]) * 111.32 * k)
    reach = (dist <= radius_km) @ d.expected_cases.to_numpy()
    order = np.argsort(-reach)
    kept = []
    for i in order:
        if reach[i] <= 0 or len(kept) >= top_n:
            break
        if all(np.hypot((points.lat.iat[i] - points.lat.iat[j]) * 110.57, (points.lon.iat[i] - points.lon.iat[j]) * 111.32 * k) > 0.15
               for j in kept):
            kept.append(i)
    return points.iloc[kept].assign(reachable_cases=reach[kept]).reset_index(drop=True)


def to_sites(df: pd.DataFrame, max_doctors: int = 4) -> list[Site]:
    return [Site(r.site_id, r.name, float(r.lat), float(r.lon), max_doctors=max_doctors, kind=str(r.kind)) for r in df.itertuples()]
