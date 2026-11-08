"""MOCK data for UI work: synthetic NYC geography, forecasts and candidate sites. No downloads, no trained models.

Every function returns the same shapes as the real pipeline, so a UI built on it runs unchanged on the real models:

    areas()                   -> like data_layer.areas() (one row per ZIP area, shapely geometry in lon/lat)
    forecast(scenario, seed)  -> like heattents.service.DayForecast (city, samples, areas, demand, neighbourhoods,
                                 scenarios, demand_neighbourhood, observed_total, observed_areas)
    candidate_sites(...)      -> like data_layer.candidate_sites() (site_id, name, lat, lon, kind, borough, ...)

What is fake:
- Geography: ~177 Voronoi "ZIP areas" inside hand-drawn, approximate borough outlines (NOT the real MODZCTA polygons),
  grouped into the 42 UHF neighbourhoods by nearest neighbourhood centre.
- Forecast: heat-index curve x hot-spell / weekend / upstream / persistence multipliers with gamma-Poisson spread.
  Area shares follow daytime presence, with weekend, beach-season and event moves. Plausible numbers, not predictions.
- Historical dates: weather is synthesised deterministically from the date, with a heatwave on 29 Jun - 4 Jul 2026.

data_layer switches to this module when HEAT_MOCK=1, or automatically when the trained models are missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import shapely
from shapely import affinity
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

SEED = 7
KM_LAT, KM_LON = 110.57, 111.32 * float(np.cos(np.radians(40.7)))
MAPPED_FRACTION = 0.96  # share of calls with a known location
PERCENTILES = [10, 30, 50, 70, 90]
SCENARIO_LABELS = ["quiet", "below median", "median", "above median", "busy"]
N_SAMPLES = 4000
CUSTOM_DATE = pd.Timestamp("2026-07-15")  # a mid-July weekday, used for custom scenarios

# --- geography --------------------------------------------------------------------------------------------------------
# Approximate outlines (lon, lat). Hand-drawn: a recognisable map for UI work, not for analysis.
OUTLINES = {
    "Manhattan": [(-74.019, 40.701), (-74.013, 40.716), (-74.010, 40.740), (-74.002, 40.760), (-73.990, 40.783),
                  (-73.968, 40.812), (-73.950, 40.836), (-73.934, 40.860), (-73.927, 40.877), (-73.910, 40.873),
                  (-73.928, 40.846), (-73.934, 40.820), (-73.929, 40.801), (-73.943, 40.776), (-73.962, 40.751),
                  (-73.972, 40.737), (-73.972, 40.722), (-73.978, 40.711), (-73.999, 40.706), (-74.008, 40.701)],
    "Bronx": [(-73.910, 40.875), (-73.918, 40.913), (-73.860, 40.905), (-73.785, 40.886), (-73.780, 40.845),
              (-73.815, 40.812), (-73.855, 40.808), (-73.890, 40.800), (-73.915, 40.796), (-73.932, 40.808),
              (-73.933, 40.835), (-73.917, 40.858)],
    "Brooklyn": [(-74.042, 40.617), (-74.010, 40.595), (-74.012, 40.576), (-73.960, 40.572), (-73.930, 40.578),
                 (-73.900, 40.590), (-73.865, 40.612), (-73.870, 40.650), (-73.856, 40.665), (-73.897, 40.682),
                 (-73.905, 40.698), (-73.925, 40.712), (-73.940, 40.735), (-73.962, 40.738), (-73.970, 40.718),
                 (-73.988, 40.705), (-74.000, 40.692), (-74.018, 40.672), (-74.020, 40.655), (-74.035, 40.640)],
    "Queens": [(-73.962, 40.740), (-73.940, 40.735), (-73.925, 40.712), (-73.905, 40.698), (-73.897, 40.682),
               (-73.856, 40.665), (-73.870, 40.650), (-73.830, 40.640), (-73.790, 40.640), (-73.745, 40.635),
               (-73.725, 40.665), (-73.730, 40.720), (-73.700, 40.740), (-73.705, 40.765), (-73.750, 40.785),
               (-73.775, 40.795), (-73.810, 40.795), (-73.860, 40.785), (-73.890, 40.775), (-73.915, 40.790),
               (-73.940, 40.775), (-73.945, 40.755), (-73.958, 40.745)],
    "Rockaway": [(-73.940, 40.545), (-73.880, 40.560), (-73.800, 40.585), (-73.740, 40.595), (-73.745, 40.607),
                 (-73.800, 40.598), (-73.880, 40.573), (-73.940, 40.558)],
    "Staten Island": [(-74.255, 40.500), (-74.200, 40.510), (-74.140, 40.535), (-74.060, 40.595), (-74.070, 40.640),
                      (-74.100, 40.648), (-74.155, 40.643), (-74.200, 40.632), (-74.205, 40.600), (-74.240, 40.560),
                      (-74.250, 40.520)],
}
# (region, borough, number of ZIP areas, first synthetic ZIP code)
REGIONS = [("Manhattan", "Manhattan", 40, 10001), ("Bronx", "Bronx", 25, 10451), ("Brooklyn", "Brooklyn", 38, 11201),
           ("Queens", "Queens", 55, 11354), ("Rockaway", "Queens", 5, 11691), ("Staten Island", "Staten Island", 14, 10301)]
DENSITY = {"Manhattan": 28000, "Bronx": 13000, "Brooklyn": 14000, "Queens": 8000, "Staten Island": 3000}  # people/km2
HVI_BASE = {"Manhattan": 2.4, "Bronx": 4.2, "Brooklyn": 3.2, "Queens": 2.5, "Staten Island": 1.8}

# The 42 UHF neighbourhoods: (uhf, name, borough, approximate centre lat, lon)
UHF42 = [
    (101, "Kingsbridge - Riverdale", "Bronx", 40.885, -73.905), (102, "Northeast Bronx", "Bronx", 40.880, -73.845),
    (103, "Fordham - Bronx Park", "Bronx", 40.862, -73.885), (104, "Pelham - Throgs Neck", "Bronx", 40.840, -73.825),
    (105, "Crotona - Tremont", "Bronx", 40.845, -73.895), (106, "High Bridge - Morrisania", "Bronx", 40.832, -73.918),
    (107, "Hunts Point - Mott Haven", "Bronx", 40.810, -73.905),
    (201, "Greenpoint", "Brooklyn", 40.718, -73.950), (202, "Downtown - Heights - Slope", "Brooklyn", 40.685, -73.985),
    (203, "Bedford Stuyvesant - Crown Heights", "Brooklyn", 40.680, -73.940), (204, "East New York", "Brooklyn", 40.665, -73.885),
    (205, "Sunset Park", "Brooklyn", 40.648, -74.010), (206, "Borough Park", "Brooklyn", 40.630, -73.985),
    (207, "East Flatbush - Flatbush", "Brooklyn", 40.645, -73.950), (208, "Canarsie - Flatlands", "Brooklyn", 40.628, -73.910),
    (209, "Bensonhurst - Bay Ridge", "Brooklyn", 40.612, -74.022), (210, "Coney Island - Sheepshead Bay", "Brooklyn", 40.585, -73.965),
    (211, "Williamsburg - Bushwick", "Brooklyn", 40.702, -73.930),
    (301, "Washington Heights - Inwood", "Manhattan", 40.850, -73.935),
    (302, "Central Harlem - Morningside Heights", "Manhattan", 40.810, -73.950),
    (303, "East Harlem", "Manhattan", 40.795, -73.940), (304, "Upper West Side", "Manhattan", 40.787, -73.975),
    (305, "Upper East Side", "Manhattan", 40.773, -73.955), (306, "Chelsea - Clinton", "Manhattan", 40.755, -73.998),
    (307, "Gramercy Park - Murray Hill", "Manhattan", 40.745, -73.978), (308, "Greenwich Village - SoHo", "Manhattan", 40.728, -74.002),
    (309, "Union Square - Lower East Side", "Manhattan", 40.720, -73.983), (310, "Lower Manhattan", "Manhattan", 40.708, -74.010),
    (401, "Long Island City - Astoria", "Queens", 40.762, -73.925), (402, "West Queens", "Queens", 40.745, -73.880),
    (403, "Flushing - Clearview", "Queens", 40.770, -73.820), (404, "Bayside - Little Neck", "Queens", 40.765, -73.765),
    (405, "Ridgewood - Forest Hills", "Queens", 40.710, -73.880), (406, "Fresh Meadows", "Queens", 40.735, -73.790),
    (407, "Southwest Queens", "Queens", 40.680, -73.840), (408, "Jamaica", "Queens", 40.695, -73.790),
    (409, "Southeast Queens", "Queens", 40.705, -73.745), (410, "Rockaway", "Queens", 40.590, -73.810),
    (501, "Port Richmond", "Staten Island", 40.635, -74.130), (502, "Stapleton - St. George", "Staten Island", 40.625, -74.085),
    (503, "Willowbrook", "Staten Island", 40.600, -74.140), (504, "South Beach - Tottenville", "Staten Island", 40.545, -74.180),
]
# daytime-presence hot spots (lon, lat, weight, scale km) and beaches (lon, lat)
PRESENCE = [(-73.985, 40.755, 8.0, 1.8), (-74.009, 40.708, 4.0, 1.2), (-73.990, 40.692, 1.5, 1.0),
            (-73.925, 40.745, 1.0, 1.2), (-73.830, 40.758, 1.0, 1.0), (-73.790, 40.645, 1.5, 1.5)]
BEACHES = [(-73.990, 40.573), (-73.960, 40.576), (-73.935, 40.580), (-73.790, 40.868), (-74.075, 40.585),
           (-74.100, 40.570)]
PARKS = [(-73.966, 40.782, 1.6), (-73.969, 40.662, 1.2), (-73.845, 40.745, 1.4), (-73.890, 40.890, 1.2),
         (-73.810, 40.870, 1.6), (-74.150, 40.590, 1.8)]  # (lon, lat, radius km): Central, Prospect, Flushing Meadows, ...

# recurring permitted events: route points (lon, lat) and how much the area's share is multiplied
EVENTS = {
    "pride": ([(-74.003, 40.735), (-73.990, 40.742)], 6.1),
    "puerto_rican_day": ([(-73.973, 40.765), (-73.962, 40.778)], 4.1),
    "dominican_day": ([(-73.983, 40.752), (-73.980, 40.760)], 3.0),
    "west_indian_day": ([(-73.945, 40.670), (-73.958, 40.672)], 15.9),
    "mermaid_parade": ([(-73.980, 40.574)], 5.0),
    "us_open": ([(-73.846, 40.750)], 3.0),
}


def _km(geom):
    return affinity.scale(geom, xfact=KM_LON, yfact=KM_LAT, origin=(0, 0))


def _deg(geom):
    return affinity.scale(geom, xfact=1 / KM_LON, yfact=1 / KM_LAT, origin=(0, 0))


def _dist_km(lat, lon, plat, plon) -> np.ndarray:
    """Distances (km) between points (lat, lon arrays) and one point."""
    return np.hypot((np.asarray(lat) - plat) * KM_LAT, (np.asarray(lon) - plon) * KM_LON)


@lru_cache(maxsize=1)
def boroughs() -> dict:
    """Region name -> polygon (lon/lat). Overlaps between neighbouring outlines are removed."""
    g = {name: Polygon(c).buffer(0) for name, c in OUTLINES.items()}
    g["Bronx"] = g["Bronx"].difference(g["Manhattan"])
    g["Queens"] = g["Queens"].difference(g["Brooklyn"])
    return g


def _random_points(poly, n: int, rng) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
    return np.array(pts)


def _voronoi_cells(poly_deg, n: int, rng, iters: int = 5) -> list:
    """n roughly equal-area cells covering the polygon (Lloyd-relaxed Voronoi, computed in km)."""
    poly = _km(poly_deg)
    pts, env = _random_points(poly, n, rng), poly.envelope.buffer(5)
    for it in range(iters + 1):
        vor = shapely.voronoi_polygons(MultiPoint(pts), extend_to=env, ordered=True)
        cells = [c.intersection(poly) for c in vor.geoms]
        if it < iters:
            pts = np.array([[c.centroid.x, c.centroid.y] if not c.is_empty else p for c, p in zip(cells, pts)])
    return [_deg(c) for c in cells if not c.is_empty and c.area > 0.01]


@lru_cache(maxsize=1)
def areas() -> pd.DataFrame:
    """One row per synthetic ZIP area: area_id, borough, uhf, lat, lon, area_km2, pop_est, presence, hvi,
    park_share, visitor_ratio, beach, usual_share, geometry (shapely, lon/lat)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for region, borough, n, zip0 in REGIONS:
        cells = _voronoi_cells(boroughs()[region], n, rng)
        cells.sort(key=lambda c: (-round(c.representative_point().y, 3), c.representative_point().x))
        for i, c in enumerate(cells):
            p = c.representative_point()
            rows.append(dict(area_id=str(zip0 + i), borough=borough, region=region, lat=p.y, lon=p.x,
                             area_km2=_km(c).area, geometry=c))
    A = pd.DataFrame(rows)
    A["pop_est"] = (A.area_km2 * A.borough.map(DENSITY) * rng.lognormal(0, 0.35, len(A))).round(-2)
    boost = sum(w * np.exp(-_dist_km(A.lat, A.lon, la, lo) / s) for lo, la, w, s in PRESENCE)
    A["presence"] = A.pop_est * (1 + boost)
    A["visitor_ratio"] = (A.presence / A.pop_est).round(2)
    north_manhattan = (A.borough == "Manhattan") & (A.lat > 40.795)
    A["hvi"] = np.clip(np.rint(A.borough.map(HVI_BASE) + 1.5 * north_manhattan + rng.normal(0, 0.8, len(A))), 1, 5).astype(int)
    near_park = sum(np.exp(-(_dist_km(A.lat, A.lon, la, lo) / r) ** 2) for lo, la, r in PARKS)
    A["park_share"] = np.clip(rng.beta(2, 9, len(A)) + 0.35 * near_park, 0, 0.8).round(3)
    near_beach = np.min([_dist_km(A.lat, A.lon, la, lo) for lo, la in BEACHES], axis=0) < 1.6
    A["beach"] = near_beach | (A.region == "Rockaway")
    anchors = pd.DataFrame(UHF42, columns=["uhf", "uhf_name", "borough", "lat", "lon"])
    A["uhf"] = 0
    for b, g in A.groupby("borough"):
        U = anchors[anchors.borough == b]
        d = np.hypot((g.lat.to_numpy()[:, None] - U.lat.to_numpy()[None]) * KM_LAT,
                     (g.lon.to_numpy()[:, None] - U.lon.to_numpy()[None]) * KM_LON)
        A.loc[g.index, "uhf"] = U.uhf.to_numpy()[d.argmin(1)]
    A = A.merge(anchors[["uhf", "uhf_name"]], on="uhf", how="left")
    base = A.presence * (0.85 + 0.08 * A.hvi)
    A["base_weight"] = base / base.sum()
    A["usual_share"] = (5 * _shares(A, leisure=False, beach_open=True, heat=90)
                        + 2 * _shares(A, leisure=True, beach_open=True, heat=90)) / 7
    cols = ["area_id", "borough", "uhf", "uhf_name", "lat", "lon", "area_km2", "pop_est", "presence", "visitor_ratio",
            "hvi", "park_share", "beach", "base_weight", "usual_share", "geometry"]
    return A[cols]


def neighbourhoods() -> pd.DataFrame:
    """The 42 UHF neighbourhoods: uhf, name, borough, lat, lon (approximate centres)."""
    return pd.DataFrame(UHF42, columns=["uhf", "name", "borough", "lat", "lon"])


# --- day context ------------------------------------------------------------------------------------------------------
HEATWAVE = {"2026-06-29": 95, "2026-06-30": 99, "2026-07-01": 104, "2026-07-02": 108, "2026-07-03": 106,
            "2026-07-04": 102, "2026-07-05": 96}


def _nth_weekday(y: int, m: int, wd: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(wd - d.day_of_week) % 7 + 7 * (n - 1))


def _last_weekday(y: int, m: int, wd: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.day_of_week - wd) % 7)


def holidays(y: int) -> set:
    return {_last_weekday(y, 5, 0), pd.Timestamp(y, 6, 19), pd.Timestamp(y, 7, 4), _nth_weekday(y, 9, 0, 1)}


def event_on(d: pd.Timestamp) -> str | None:
    y = d.year
    june18 = pd.Timestamp(y, 6, 18)
    parades = {_last_weekday(y, 6, 6): "pride", _nth_weekday(y, 6, 6, 2): "puerto_rican_day",
               _nth_weekday(y, 8, 6, 2): "dominican_day", _nth_weekday(y, 9, 0, 1): "west_indian_day",
               june18 + pd.Timedelta(days=(5 - june18.day_of_week) % 7): "mermaid_parade"}
    if d in parades:
        return parades[d]
    us_open = _last_weekday(y, 8, 0)
    return "us_open" if us_open <= d < us_open + pd.Timedelta(days=14) else None


def beach_season(d: pd.Timestamp) -> bool:
    return _last_weekday(d.year, 5, 0) - pd.Timedelta(days=2) <= d <= _nth_weekday(d.year, 9, 0, 1)


@lru_cache(maxsize=4096)
def heat_index_on(d: pd.Timestamp) -> float:
    """Synthetic daily max heat index (F): seasonal curve + autocorrelated noise, deterministic in the date."""
    key = f"{d:%Y-%m-%d}"
    if key in HEATWAVE:
        return float(HEATWAVE[key])
    clim = 72 + 17 * np.exp(-0.5 * ((d.day_of_year - 200) / 45) ** 2)
    e = [np.random.default_rng((d - pd.Timedelta(days=k)).toordinal()).normal() for k in range(3)]
    return float(round(clim + 6.5 * (0.7 * e[0] + 0.5 * e[1] + 0.3 * e[2]) / 0.91, 1))


@dataclass(frozen=True)
class Day:
    date: pd.Timestamp
    heat_index_f: float
    hot_days_before: int
    weekend: bool
    holiday: bool
    upstream_gradient_f: float
    calls_by18: float
    event: str | None
    beach_open: bool

    @property
    def leisure(self) -> bool:
        return self.weekend or self.holiday


def day_for_date(date) -> Day:
    d = pd.Timestamp(date).normalize()
    hot = 0
    while hot < 7 and heat_index_on(d - pd.Timedelta(days=hot + 1)) >= 90:
        hot += 1
    hi, hi_prev = heat_index_on(d), heat_index_on(d - pd.Timedelta(days=1))
    up = round(0.6 * (hi - hi_prev) + np.random.default_rng(d.toordinal() + 99).normal(0, 2), 1)
    prev = d - pd.Timedelta(days=1)
    calls_by18 = 0.55 * _base_mean(hi_prev, min(hot, 6), prev.day_of_week >= 5 or prev in holidays(prev.year), 0.0)
    return Day(d, hi, hot, d.day_of_week >= 5, d in holidays(d.year), up, round(calls_by18, 1), event_on(d), beach_season(d))


def day_for_scenario(sc) -> Day:
    return Day(CUSTOM_DATE, float(sc.heat_index_f), int(sc.hot_days_before), bool(sc.weekend), bool(sc.holiday),
               float(sc.upstream_gradient_f), float(sc.calls_yesterday_by18), sc.event, True)


# --- forecast ---------------------------------------------------------------------------------------------------------
def _base_mean(hi: float, hot: int, leisure: bool, upstream: float) -> float:
    m = 30 * np.exp(0.17 * (hi - 100)) * (1 + 0.12 * min(hot, 5)) * (0.9 if leisure else 1.0) * np.exp(0.035 * upstream)
    return float(max(m, 0.3))


def city_mean(day: Day) -> float:
    m = _base_mean(day.heat_index_f, day.hot_days_before, day.leisure, day.upstream_gradient_f)
    return m * ((day.calls_by18 + 5) / (0.55 * m + 5)) ** 0.25  # yesterday's calls pull the forecast a little


DISPERSION = 6.0  # gamma-Poisson shape of the predictive distribution


def model_means(day: Day) -> dict:
    """City mean of each model kind, like heattents.models.city.CityModel.means: glm, gbm, ens = (glm + gbm) / 2.
    The GLM runs hotter on record heat (it extrapolates), the GBM flatter; they disagree a little day to day."""
    ens = city_mean(day)
    e = 0.06 * (day.heat_index_f - 98) / 10 + np.random.default_rng(day.date.toordinal() + 7).normal(0, 0.12)
    e = float(np.clip(e, -0.5, 0.5))
    return dict(glm=ens * (1 + e), gbm=ens * (1 - e), ens=ens)


def _observed_total(day: Day, mean: float) -> float:
    """What 'happened' on a historical date: one draw from the forecast distribution (fixed per date)."""
    rng = np.random.default_rng(day.date.toordinal() + 12345)
    return float(rng.poisson(rng.gamma(DISPERSION, mean / DISPERSION)))


def _observed_areas(day: Day, observed_total: float, share: np.ndarray) -> np.ndarray:
    """Where the observed calls of a historical date happened (located calls only, fixed per date)."""
    rng = np.random.default_rng(day.date.toordinal() + 54321)
    return rng.multinomial(int(round(observed_total * MAPPED_FRACTION)), share)


def neighbourhood_history(uhf, start, end) -> pd.DataFrame:
    """Ensemble forecast vs observed calls in one neighbourhood, day by day: date, observed, forecast, q05, q95."""
    from scipy.stats import nbinom
    A = areas()
    in_nb = (A.uhf.astype(str) == str(uhf)).to_numpy()
    rows = []
    for d in pd.date_range(start, end):
        day = day_for_date(d)
        mean = city_mean(day)
        share = _shares(A, day.leisure, day.beach_open, day.heat_index_f, day.event)
        obs = _observed_areas(day, _observed_total(day, mean), share)[in_nb].sum()
        rows.append(dict(date=d, observed=float(obs), forecast=mean * MAPPED_FRACTION * share[in_nb].sum()))
    H = pd.DataFrame(rows)
    p = DISPERSION / (DISPERSION + H.forecast.to_numpy())
    return H.assign(q05=nbinom.ppf(0.05, DISPERSION, p), q95=nbinom.ppf(0.95, DISPERSION, p))


@lru_cache(maxsize=1)
def city_predictions() -> pd.DataFrame:
    """Like outputs/models/city/oos_predictions.csv: one row per (test day, model) for May-Sep 2024-2026:
    season, model (glm/gbm/ens), date, y (observed), mean, q05, q50, q95."""
    from scipy.stats import nbinom
    rows = []
    for y in (2024, 2025, 2026):
        for d in pd.date_range(f"{y}-05-01", min(pd.Timestamp(f"{y}-09-30"), pd.Timestamp("2026-09-21"))):
            day = day_for_date(d)
            means = model_means(day)
            obs = _observed_total(day, means["ens"])
            rows += [dict(season=y, model=kind, date=d, y=obs, mean=m) for kind, m in means.items()]
    P = pd.DataFrame(rows)
    p = DISPERSION / (DISPERSION + P["mean"].to_numpy())
    for q in (5, 50, 95):
        P[f"q{q:02d}"] = nbinom.ppf(q / 100, DISPERSION, p)
    return P


def _shares(A: pd.DataFrame, leisure: bool, beach_open: bool, heat: float, event: str | None = None) -> np.ndarray:
    w = A.base_weight.to_numpy(float).copy()
    if leisure:  # beaches and parks gain, office districts lose
        w *= np.where(A.beach, 1.5, 1.0) * (1 + 0.5 * A.park_share.to_numpy()) * A.visitor_ratio.to_numpy() ** -0.35
    if beach_open:
        w *= np.where(A.beach, 1.6 * (1.5 if leisure else 1.0) * (1 + 0.04 * max(heat - 90, 0)), 1.0)
    if event:
        pts, mult = EVENTS[event]
        near = np.min([_dist_km(A.lat, A.lon, la, lo) for lo, la in pts], axis=0) < 0.9
        w *= np.where(near, mult, 1.0)
    return w / w.sum()


def hour_profile(heat: float) -> np.ndarray:
    """Share of the day's calls by hour: peak at 14:00, ~62% between 12:00 and 18:00, wider on very hot days."""
    h = np.arange(24)
    sd = 3.1 + (0.5 if heat >= 97 else 0.0)
    p = np.exp(-0.5 * ((h - 14.3) / sd) ** 2) + 0.02
    return p / p.sum()


@dataclass
class DayForecast:
    """Same fields as heattents.service.DayForecast."""
    label: str
    city: pd.Series  # mean, q05..q95, p_ge50, p_ge100
    samples: np.ndarray  # predictive samples of the citywide total
    areas: pd.DataFrame  # modzcta, borough, share, usual_share, change_vs_usual, expected, p_ge1
    demand: pd.DataFrame  # optimiser input: area_id, lat, lon, hour, scenario, expected_cases
    observed_total: float | None = None
    observed_areas: pd.Series | None = None
    neighbourhoods: pd.DataFrame | None = None  # uhf, name, borough, share, usual_share, change_vs_usual, expected, q05, q95, p_ge1 (, observed)
    scenarios: pd.DataFrame | None = None  # scenario, label, percentile, city_total, located_total, probability
    demand_neighbourhood: pd.DataFrame | None = None  # uhf, scenario, probability, expected_cases
    day: Day | None = None  # mock only: the (synthetic) inputs behind this forecast
    model_means: dict | None = None  # city mean per model kind: glm, gbm, ens


def forecast(sc, seed: int = 0) -> DayForecast:
    """Mock forecast for a data_layer.Scenario (historical date or custom inputs)."""
    day = day_for_date(sc.date) if sc.date is not None else day_for_scenario(sc)
    rng = np.random.default_rng([seed, day.date.toordinal(), int(day.heat_index_f * 10)])
    A = areas()
    mean = city_mean(day)
    k = DISPERSION
    samples = rng.poisson(rng.gamma(k, mean / k, N_SAMPLES)).astype(float)
    q = np.quantile(samples, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    city = pd.Series(dict(mean=samples.mean(), q05=q[0], q10=q[1], q25=q[2], q50=q[3], q75=q[4], q90=q[5], q95=q[6],
                          p_ge50=(samples >= 50).mean(), p_ge100=(samples >= 100).mean()))

    share = _shares(A, day.leisure, day.beach_open, day.heat_index_f, day.event)
    located = samples * MAPPED_FRACTION
    p_ge1 = 1 - np.exp(-np.outer(located[:500], share)).mean(0)
    areas_df = pd.DataFrame(dict(modzcta=A.area_id, borough=A.borough, uhf=A.uhf, share=share, usual_share=A.usual_share,
                                 change_vs_usual=share / A.usual_share, expected=city["mean"] * MAPPED_FRACTION * share,
                                 p_ge1=p_ge1))

    # five equally likely versions of the day: the 10/30/50/70/90th percentiles of the citywide total
    totals = np.percentile(samples, PERCENTILES)
    scen = pd.DataFrame(dict(scenario=range(len(PERCENTILES)), label=SCENARIO_LABELS, percentile=PERCENTILES,
                             city_total=totals, located_total=totals * MAPPED_FRACTION, probability=1 / len(PERCENTILES)))
    prof = hour_profile(day.heat_index_f)
    n_a, n_s = len(A), len(scen)
    demand = pd.DataFrame(dict(
        area_id=np.tile(np.repeat(A.area_id.to_numpy(), 24), n_s),
        lat=np.tile(np.repeat(A.lat.to_numpy(), 24), n_s),
        lon=np.tile(np.repeat(A.lon.to_numpy(), 24), n_s),
        hour=np.tile(np.arange(24), n_a * n_s),
        scenario=np.repeat(scen.scenario.to_numpy(), n_a * 24),
        expected_cases=(scen.located_total.to_numpy()[:, None, None] * share[None, :, None] * prof[None, None, :]).ravel(),
    ))

    observed_total = observed_areas = None
    if sc.date is not None:  # a "historical" day: draw what happened from the forecast itself
        observed_total = _observed_total(day, mean)
        observed_areas = pd.Series(_observed_areas(day, observed_total, share), index=A.area_id.to_numpy(),
                                   name="observed")

    nb = areas_df.groupby("uhf")[["share", "usual_share", "expected"]].sum()
    draws = rng.binomial(located[:2000].astype(int)[:, None], nb.share.to_numpy()[None, :])
    nb = nb.assign(change_vs_usual=nb.share / nb.usual_share, q05=np.quantile(draws, 0.05, axis=0),
                   q95=np.quantile(draws, 0.95, axis=0), p_ge1=(draws >= 1).mean(0)).reset_index()
    nb = neighbourhoods()[["uhf", "name", "borough"]].merge(nb, on="uhf", how="left").fillna(0)
    if observed_areas is not None:
        nb["observed"] = nb.uhf.map(observed_areas.groupby(A.uhf.to_numpy()).sum()).fillna(0)
    demand_nb = pd.DataFrame(dict(uhf=np.tile(nb.uhf.to_numpy(), n_s), scenario=np.repeat(scen.scenario.to_numpy(), len(nb)),
                                  probability=1 / n_s,
                                  expected_cases=(scen.located_total.to_numpy()[:, None] * nb.share.to_numpy()[None]).ravel()))

    label = f"{day.date:%Y-%m-%d} (mock)" if sc.date is not None else "custom scenario (mock)"
    return DayForecast(label, city, samples, areas_df, demand, observed_total, observed_areas, nb, scen, demand_nb, day,
                       model_means(day))


# --- candidate sites --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def park_points(n: int = 600) -> pd.DataFrame:
    """Synthetic park / playground / plaza points, spread by area (like heattents.deploy.candidates.park_points)."""
    rng = np.random.default_rng(SEED + 1)
    A = areas()
    pick = rng.choice(len(A), n, p=(A.area_km2 / A.area_km2.sum()).to_numpy())
    rows = []
    for i, a in enumerate(pick):
        r = A.iloc[a]
        (x, y), = _random_points(r.geometry, 1, rng)
        kind = "Waterfront Facility" if r.beach and rng.random() < 0.4 else \
            rng.choice(["Neighborhood Park", "Playground", "Triangle/Plaza", "Community Park"], p=[0.35, 0.35, 0.2, 0.1])
        rows.append(dict(site_id=f"P{i:04d}#0", name=f"{r.borough} {kind.split('/')[0].lower()} #{i:03d}", kind=kind,
                         acres=round(float(rng.lognormal(0.5, 1.2)), 2), borough=r.borough, lat=y, lon=x))
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def cooling_sites(n: int = 90) -> pd.DataFrame:
    """Synthetic Cool It! NYC-style cooling centres, spread by population."""
    rng = np.random.default_rng(SEED + 2)
    A = areas()
    pick = rng.choice(len(A), n, p=(A.pop_est / A.pop_est.sum()).to_numpy())
    rows = []
    for i, a in enumerate(pick):
        r = A.iloc[a]
        (x, y), = _random_points(r.geometry, 1, rng)
        kind = rng.choice(["Library", "Community Center", "Senior Center", "NYCHA Facility"])
        rows.append(dict(site_id=f"CS{i:03d}", name=f"{kind} {r.area_id}-{i:02d}", lat=y, lon=x, kind=kind, borough=r.borough))
    return pd.DataFrame(rows)


def screen(points: pd.DataFrame, demand: pd.DataFrame, radius_km: float, top_n: int = 250) -> pd.DataFrame:
    """Top_n candidates by expected demand within the radius, one per 150 m (same rule as deploy.candidates.screen)."""
    d = demand.groupby(["area_id", "lat", "lon"], as_index=False).expected_cases.sum()
    d["expected_cases"] /= demand.scenario.nunique()
    dist = np.hypot((points.lat.to_numpy()[:, None] - d.lat.to_numpy()[None]) * KM_LAT,
                    (points.lon.to_numpy()[:, None] - d.lon.to_numpy()[None]) * KM_LON)
    reach = (dist <= radius_km) @ d.expected_cases.to_numpy()
    kept = []
    for i in np.argsort(-reach):
        if reach[i] <= 0 or len(kept) >= top_n:
            break
        if all(_dist_km(points.lat.iat[i], points.lon.iat[i], points.lat.iat[j], points.lon.iat[j]) > 0.15 for j in kept):
            kept.append(i)
    return points.iloc[kept].assign(reachable_cases=reach[kept]).reset_index(drop=True)


def candidate_sites(source: str, demand: pd.DataFrame, radius_km: float = 1.0, top_n: int = 250) -> pd.DataFrame:
    if source.startswith("Cool It"):
        return cooling_sites()[["site_id", "name", "lat", "lon", "kind", "borough"]]
    return screen(park_points(), demand, radius_km, top_n)
