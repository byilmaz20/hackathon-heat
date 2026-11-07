"""Spatial units and crosswalks.

Modelling unit: MODZCTA (178 "modified ZIP code tabulation areas" used by the NYC Health Department; each groups
one or more raw ZIP codes, so every EMS ``zipcode`` maps to exactly one MODZCTA).
Context indicators are published for NTA2010 / UHF42 / CD polygons; they are moved to MODZCTA by
area-weighted overlay (:func:`area_weights`).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.ops import transform

from ..config import PATHS, Paths
from .http import get

MODZCTA_URL = "https://data.cityofnewyork.us/api/geospatial/pri4-ifjk?method=export&format=GeoJSON"
EHDP_GEO = "https://raw.githubusercontent.com/nychealth/EHDP-data/production/geography/"


def _cached(paths: Paths, name: str, url: str, refresh: bool) -> dict:
    f = paths.raw / name
    if refresh or not f.exists():
        f.write_bytes(get(url))
    return json.loads(f.read_text())


def topojson_to_geojson(topo: dict) -> dict:
    """Decode a (quantized) TopoJSON topology into a GeoJSON FeatureCollection (Polygon / MultiPolygon only)."""
    tr = topo.get("transform")
    arcs = []
    for arc in topo["arcs"]:
        pts = np.array(arc, dtype=float)
        if tr:
            pts = np.cumsum(pts, axis=0) * tr["scale"] + tr["translate"]
        arcs.append(pts)

    def ring(idx):
        coords = []
        for i in idx:
            pts = arcs[i] if i >= 0 else arcs[~i][::-1]
            coords.extend(pts.tolist() if not coords else pts[1:].tolist())
        return coords

    feats = []
    for obj in topo["objects"].values():
        for g in obj["geometries"]:
            if g["type"] == "Polygon":
                geom = {"type": "Polygon", "coordinates": [ring(r) for r in g["arcs"]]}
            elif g["type"] == "MultiPolygon":
                geom = {"type": "MultiPolygon", "coordinates": [[ring(r) for r in p] for p in g["arcs"]]}
            else:
                continue
            feats.append({"type": "Feature", "properties": g.get("properties", {}), "geometry": geom})
    return {"type": "FeatureCollection", "features": feats}


def _to_km(geom):
    """Local equirectangular projection (km) - adequate for area ratios within NYC."""
    k = np.cos(np.radians(40.7))
    return transform(lambda x, y, z=None: (np.asarray(x) * 111.32 * k, np.asarray(y) * 110.57), geom)


def load_modzcta(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    """One row per MODZCTA: id, raw ZIP list, population estimate, centroid, area, shapely geometry (lon/lat)."""
    gj = _cached(paths, "nyc_modzcta.geojson", MODZCTA_URL, refresh)
    rows = []
    for f in gj["features"]:
        p, geom = f["properties"], shape(f["geometry"]).buffer(0)
        if p["modzcta"] == "99999":
            continue
        c = geom.representative_point()
        rows.append(dict(modzcta=p["modzcta"], zips=[z.strip() for z in p["zcta"].split(",")],
                         pop_est=float(p["pop_est"]), lat=c.y, lon=c.x, area_km2=_to_km(geom).area, geometry=geom))
    return pd.DataFrame(rows)


def zip_to_modzcta(Z: pd.DataFrame) -> dict[str, str]:
    m = {z: r.modzcta for r in Z.itertuples() for z in r.zips}
    m.update({r.modzcta: r.modzcta for r in Z.itertuples()})
    return m


def load_polygons(kind: str, paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    """Context polygons from the NYC Environment & Health Data Portal: kind in {NTA2010, UHF42, CD}."""
    src = {"NTA2010": "NTA2010.topo.json", "UHF42": "UHF42.topo.json", "CD": "CD.topo.json"}[kind]
    topo = _cached(paths, f"ehdp_{kind}.topo.json", EHDP_GEO + src, refresh)
    gj = topojson_to_geojson(topo)
    rows = []
    for f in gj["features"]:
        p = f["properties"]
        gid = p.get("GEOCODE") or p.get("GeoID") or p.get("BoroCD") or p.get("UHFCODE") or p.get("NTACode") or p.get("id")
        rows.append(dict(geo_id=str(gid), props=p, geometry=shape(f["geometry"]).buffer(0)))
    return pd.DataFrame(rows)


UHF_BOROUGH = {"1": "BRONX", "2": "BROOKLYN", "3": "MANHATTAN", "4": "QUEENS", "5": "RICHMOND / STATEN ISLAND"}


def load_neighbourhoods(paths: Paths = PATHS) -> pd.DataFrame:
    """The 42 United Hospital Fund (UHF42) neighbourhoods of the NYC Health Department: uhf code, name, borough
    (EMS spelling, from the code's first digit), representative point, area and geometry."""
    U = load_polygons("UHF42", paths)
    U = U[U.geo_id != "9999"].reset_index(drop=True)
    pts = [g.representative_point() for g in U.geometry]
    return pd.DataFrame({"uhf": U.geo_id, "name": [p["GEONAME"] for p in U.props],
                         "borough": U.geo_id.str[0].map(UHF_BOROUGH), "lat": [p.y for p in pts],
                         "lon": [p.x for p in pts], "area_km2": [_to_km(g).area for g in U.geometry],
                         "geometry": U.geometry})


def modzcta_to_neighbourhood(paths: Paths = PATHS) -> pd.Series:
    """MODZCTA -> UHF42 code (NYC Health Department ZIP crosswalk; every MODZCTA lies in exactly one neighbourhood)."""
    x = pd.read_csv(paths.raw / "ehdp_zcta_to_uhf.csv", dtype=str)
    x = x[x.uhfcode != "9999"].drop_duplicates("zcta")
    return pd.Series(x.uhfcode.to_numpy(), index=x.zcta.to_numpy(), name="uhf")


def area_weights(Z: pd.DataFrame, polys: pd.DataFrame) -> pd.DataFrame:
    """Long table (modzcta, geo_id, weight): share of each MODZCTA's area lying in each polygon."""
    zk = [_to_km(g) for g in Z.geometry]
    pk = [_to_km(g) for g in polys.geometry]
    rows = []
    for zi, zg in enumerate(zk):
        for pi, pg in enumerate(pk):
            if zg.intersects(pg):
                a = zg.intersection(pg).area
                if a > 1e-6:
                    rows.append((Z.modzcta.iloc[zi], polys.geo_id.iloc[pi], a))
    w = pd.DataFrame(rows, columns=["modzcta", "geo_id", "area"])
    w["weight"] = w.area / w.groupby("modzcta").area.transform("sum")
    return w[["modzcta", "geo_id", "weight"]]
