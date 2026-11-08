r"""Is forecast + MILP better than the rules a city would use? A fair backtest on real outcomes.

PLANS (same resources for all: N tents, W health workers, c patients per worker per day, 15-minute walk)
    population        the N most populous neighbourhoods; workers in proportion to population
    vulnerability     the N neighbourhoods with the highest Heat Vulnerability Index (population-weighted)
    historical calls  the N neighbourhoods with the most heat calls in earlier summers (data-based, but fixed)
    forecast + MILP   the evening-before forecast (5 scenarios) and the daily MILP (deploy.daily): a new plan each day
    hindsight         the MILP run on what actually happened: the best possible plan (an upper bound, not a plan)
The three rules place their tent at the park or plaza in each chosen neighbourhood that has the most residents within
a 15-minute walk, and split workers in proportion to their own score (at least 1 per tent).

FAIRNESS
    - every plan is scored on the calls that ACTUALLY happened that day (in people: x 20 per call), spread over the ZIP
      area's demand points, with the same walking and capacity rules (an LP: serve as many as possible);
    - the forecast is out of sample: for each test summer the city and share models are refitted on earlier summers;
    - "historical calls" uses earlier summers only;
    - test days: 2024-26 days whose evening-before forecast heat index is >= 95 F (35 C), when a city would deploy.
Run: python -m heattents.deploy.compare
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .daily import WALK_MINUTES, _km, solve, walk_radius_km, walking_request

TEST_SEASONS = (2024, 2025, 2026)
DEPLOY_HEAT_C = 35.0  # forecast max heat index >= 95 F


def _lp_served(sites: pd.DataFrame, pts: pd.DataFrame, visitors: np.ndarray, radius_km: float, c: float) -> tuple[float, float]:
    """(served, within_reach) of a fixed plan: sites [lat, lon, workers]; visitors per demand point."""
    import gurobipy as gp
    if len(sites) == 0:
        return 0.0, 0.0
    within = _km(sites.lat, sites.lon, pts.lat, pts.lon) <= radius_km
    reach_pts = within.any(0)
    m = gp.Model()
    m.Params.OutputFlag = 0
    arcs = [(i, a) for i in range(len(sites)) for a in np.flatnonzero(within[i]) if visitors[a] > 0]
    f = m.addVars(arcs, lb=0.0)
    for i in range(len(sites)):
        m.addConstr(gp.quicksum(f[i, a] for (ii, a) in arcs if ii == i) <= c * float(sites.workers.iloc[i]))
    by_a = {}
    for i, a in arcs:
        by_a.setdefault(a, []).append(i)
    for a, js in by_a.items():
        m.addConstr(gp.quicksum(f[i, a] for i in js) <= float(visitors[a]))
    m.setObjective(f.sum(), gp.GRB.MAXIMIZE)
    m.optimize()
    return float(m.ObjVal), float(visitors[reach_pts].sum())


def _split(scores: np.ndarray, W: int) -> np.ndarray:
    """W workers over tents in proportion to scores, at least 1 each (largest remainder)."""
    n = len(scores)
    base = np.ones(n, int)
    rest = W - n
    if rest <= 0:
        return base[:W] if W < n else base
    share = scores / scores.sum() * rest
    add = np.floor(share).astype(int)
    for i in np.argsort(-(share - add))[: rest - add.sum()]:
        add[i] += 1
    return base + add


class RulePlans:
    """Fixed rule-based plans (population, vulnerability, historical calls) for a test summer."""

    def __init__(self, paths: Paths, pts: pd.DataFrame, train_calls: pd.Series, radius_km: float,
                 sites: pd.DataFrame | None = None):
        """sites: optional candidate sites [name, neighbourhood, lat, lon] (default: NYC parks and plazas)."""
        from shapely import contains_xy
        from ..ingest.geography import load_neighbourhoods, modzcta_to_neighbourhood
        from .candidates import park_points
        S = pd.read_csv(paths.processed / "modzcta_static.csv", dtype={"modzcta": str})
        S["uhf"] = S.modzcta.map(modzcta_to_neighbourhood(paths))
        pop = S.groupby("uhf").pop_est.sum()
        hvi = S.groupby("uhf").apply(lambda g: np.average(g.hvi, weights=g.pop_est), include_groups=False)
        self.scores = {"population": pop, "vulnerability (HVI)": hvi, "historical calls": train_calls}
        # each neighbourhood's site with the most residents within reach
        if sites is not None:
            P = sites.rename(columns={"neighbourhood": "uhf"})[["name", "uhf", "lat", "lon"]].reset_index(drop=True)
        else:
            P = park_points(paths).copy()
            P["uhf"] = None
            for r in load_neighbourhoods(paths).itertuples():
                P.loc[contains_xy(r.geometry, P.lon.to_numpy(), P.lat.to_numpy()) & P.uhf.isna(), "uhf"] = r.uhf
            P = P.dropna(subset=["uhf"]).reset_index(drop=True)
        residents = pts.modzcta.map(S.set_index("modzcta").pop_est).to_numpy() * pts.weight.to_numpy()
        P["residents_in_reach"] = (_km(P.lat, P.lon, pts.lat, pts.lon) <= radius_km) @ residents
        self.best_park = P.sort_values("residents_in_reach", ascending=False).drop_duplicates("uhf").set_index("uhf")

    def plan(self, rule: str, N: int, W: int) -> pd.DataFrame:
        sc = self.scores[rule].reindex(self.best_park.index).dropna().sort_values(ascending=False).head(N)
        sites = self.best_park.loc[sc.index, ["name", "lat", "lon"]].reset_index()
        sites["workers"] = _split(sc.to_numpy(float), W)
        return sites


def seasonal_service(season: int, paths: Paths = PATHS):
    """A ForecastService whose city and share models were trained on summers before ``season`` only."""
    from ..models import context_share as cs
    from ..models.city import Context, fit_city_model, observed_features
    from ..service import ForecastService
    svc = ForecastService(paths)
    ctx = svc.ctx
    F = observed_features(ctx, ctx.city)
    tr = (ctx.city.year < season).to_numpy()
    svc.city_model = fit_city_model(F[tr], ctx.city.city_y.to_numpy()[tr], t95=ctx.t95)
    P = cs.load(paths)
    svc.area_model = cs.fit_tuned(P[P.year < season])
    return svc


def run(n_tents: int = 10, n_workers: int = 40, patients_per_worker_day: float = 30.0,
        paths: Paths = PATHS, max_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ..ingest.geography import modzcta_to_neighbourhood
    from ..models.scenarios import demand_points
    out = paths.out("models/deploy_compare")
    radius = walk_radius_km()
    pts = demand_points(4).reset_index(drop=True)
    nb_of = modzcta_to_neighbourhood(paths)
    rows = []
    for season in TEST_SEASONS:
        svc = seasonal_service(season, paths)
        ctx = svc.ctx
        C = ctx.city[ctx.city.year == season]
        days = [d for d in C.date if ctx.fc.atmax.get(d, -99) >= DEPLOY_HEAT_C]
        P = pd.read_parquet(paths.processed / "panel.parquet", columns=["modzcta", "date", "y"])
        hist = P[P.date.dt.year < season].assign(uhf=lambda d: d.modzcta.map(nb_of)).groupby("uhf").y.sum()
        rules = RulePlans(paths, pts, hist, radius)
        for d in days[:max_days]:
            fc = svc.forecast_date(d)
            actual = pts.modzcta.map(fc.observed_areas).fillna(0).to_numpy() * pts.weight.to_numpy()  # people
            plans = {r: rules.plan(r, n_tents, n_workers) for r in rules.scores}
            mp = solve(walking_request(fc, n_tents, n_workers,
                                       patients_per_worker_day=patients_per_worker_day))
            plans["forecast + MILP"] = mp.tents[["lat", "lon", "workers", "name"]]
            # hindsight: the MILP on the actual day (one scenario with probability 1)
            hs_fc = _actual_forecast(fc)
            hp = solve(walking_request(hs_fc, n_tents, n_workers,
                                       patients_per_worker_day=patients_per_worker_day))
            plans["hindsight (upper bound)"] = hp.tents[["lat", "lon", "workers", "name"]]
            for name, sites in plans.items():
                served, reach = _lp_served(sites, pts, actual, radius, patients_per_worker_day)
                rows.append(dict(season=season, date=d, plan=name, actual_people=float(fc.observed_total),
                                 visitors=float(actual.sum()), within_reach=reach, served=served))
        print(f"[compare] {season}: {len(days[:max_days])} deployment days", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "daily_results.csv", index=False)
    order = ["population", "vulnerability (HVI)", "historical calls", "forecast + MILP", "hindsight (upper bound)"]
    S = R.groupby("plan").agg(days=("date", "nunique"), visitors=("visitors", "sum"), within_reach=("within_reach", "sum"),
                              served=("served", "sum")).reindex(order)
    S["served_pct"] = 100 * S.served / S.visitors
    S["within_reach_pct"] = 100 * S.within_reach / S.visitors
    base = R[R.plan == "forecast + MILP"].set_index("date").served
    for p in order[:3]:
        other = R[R.plan == p].set_index("date").served
        S.loc[p, "days_MILP_better"] = int((base > other + 1e-6).sum())
        S.loc[p, "days_MILP_worse"] = int((base < other - 1e-6).sum())
    S.round(2).to_csv(out / "summary.csv")
    print(S.round(1).to_string())
    return S, R


COVERAGE_RULES = dict(radius_km=2.0, site_mode="anywhere", max_per_neighbourhood=3, unlimited_capacity=True)
PLAN_ORDER = ["population", "vulnerability (HVI)", "historical calls", "forecast + MILP", "hindsight (upper bound)"]


def run_coverage(dates=("2024-06-21", "2025-06-29"), n_tents: int = 10, paths: Paths = PATHS,
                 rules: dict = COVERAGE_RULES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparison under the app's active planning rules (maximum coverage): a hospital helps everyone within
    rules['radius_km'] (no worker limit), hospitals anywhere on land (parks + 400 m grid), up to 3 per neighbourhood.
    The three rules use the same candidate sites (their neighbourhood's site with the most residents within reach).
    Score = people who actually needed help that day (calls x 20) at demand points within reach of an open hospital.
    Returns (results per day and plan, sites per day and plan) and writes them to deploy_compare/coverage/."""
    from ..ingest.geography import modzcta_to_neighbourhood
    from ..models.scenarios import demand_points
    from .daily import candidate_grid, candidate_parks
    out = paths.out("models/deploy_compare/coverage")
    radius = float(rules["radius_km"])
    pts = demand_points(4).reset_index(drop=True)
    nb_of = modzcta_to_neighbourhood(paths)
    S = pd.read_csv(paths.processed / "modzcta_static.csv", dtype={"modzcta": str})
    residents = pts.modzcta.map(S.set_index("modzcta").pop_est).fillna(0).to_numpy() * pts.weight.to_numpy()
    cand = pd.concat([candidate_parks(pts.assign(neighbourhood=pts.modzcta.map(nb_of)), residents, radius, 5, paths),
                      candidate_grid(pts, radius, 0.4, paths)], ignore_index=True)
    panel = pd.read_parquet(paths.processed / "panel.parquet", columns=["modzcta", "date", "y"])
    kw = dict(max_per_neighbourhood=rules["max_per_neighbourhood"], site_mode=rules["site_mode"],
              radius_km=radius, unlimited_capacity=rules["unlimited_capacity"])
    rows, site_rows = [], []
    for d in pd.to_datetime(list(dates)):
        season = d.year
        svc = seasonal_service(season, paths)
        hist = panel[panel.date.dt.year < season].assign(uhf=lambda t: t.modzcta.map(nb_of)).groupby("uhf").y.sum()
        rp = RulePlans(paths, pts, hist, radius, sites=cand)
        fc = svc.forecast_date(d)
        actual = pts.modzcta.map(fc.observed_areas).fillna(0).to_numpy() * pts.weight.to_numpy()  # people
        plans = {r: rp.plan(r, n_tents, n_tents)[["name", "lat", "lon"]] for r in rp.scores}
        plans["forecast + MILP"] = solve(walking_request(fc, n_tents, 0, **kw)).tents[["name", "lat", "lon"]]
        plans["hindsight (upper bound)"] = solve(walking_request(_actual_forecast(fc), n_tents, 0, **kw)).tents[
            ["name", "lat", "lon"]]
        for name, sites in plans.items():
            within = (_km(sites.lat, sites.lon, pts.lat, pts.lon) <= radius).any(0) if len(sites) else np.zeros(len(pts), bool)
            rows.append(dict(date=d.date(), plan=name, needed=float(actual.sum()), helped=float(actual[within].sum())))
            site_rows.append(sites.assign(date=d.date(), plan=name))
        print(f"[compare] coverage {d.date()} done", flush=True)
    R = pd.DataFrame(rows)
    R["helped_pct"] = 100 * R.helped / R.needed
    Sx = pd.concat(site_rows, ignore_index=True)[["date", "plan", "name", "lat", "lon"]]
    R.to_csv(out / "daily_results.csv", index=False)
    Sx.to_csv(out / "sites.csv", index=False)
    print(R.round(1).to_string())
    return R, Sx


def _actual_forecast(fc):
    """A copy of the day's forecast whose single scenario is what actually happened (for the hindsight plan)."""
    import copy
    h = copy.copy(fc)
    obs = fc.observed_areas.reindex(fc.areas.modzcta).fillna(0).to_numpy()
    h.areas = fc.areas.assign(share=obs / max(obs.sum(), 1e-9))
    h.scenarios = pd.DataFrame({"scenario": [0], "probability": [1.0], "located_total": [float(obs.sum())],
                                "city_total": [float(fc.observed_total)]})
    return h


if __name__ == "__main__":
    from heattents.deploy.compare import run as _run
    _run()
