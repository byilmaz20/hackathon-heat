r"""Daily tent and health-worker placement: a two-stage stochastic MILP that minimises expected unanswered visitors.

Input is the prediction output (docs/forecast_model.md, "Output"): daily expected heat EMS calls in 5 equally likely
scenarios, per neighbourhood (and per ZIP area inside it). The planner fixes the number of tents N and of health
workers W in advance.

DEMAND = PEOPLE, NOT CALLS. EMS calls are the tip of heat illness (the busiest neighbourhood typically has 1 call a day,
>10 on 0.5% of days). A tent serves the many overheated people who never call 911, so the forecast is used as an
urgency signal and scaled to people: the forecast service converts its output once (service.to_people, 20 people
per EMS heat call), so the demand D_as reaching this model is already in people and visitors_per_call = 1.

WHO CAN REACH A TENT (default, walking_request): people walk at most 15 minutes. Demand is placed at points inside
each ZIP area (its forecast share, spread over up to 4 interior points); tents stand at NYC parks and plazas (up to
5 best candidates per neighbourhood, at most one tent per neighbourhood). A point can use a tent if the straight-line
distance is <= walk_minutes/60 * walk_speed_kmh / detour  (15 min, 5 km/h, detour 1.3 -> 0.96 km), whatever
neighbourhood it is in. A simpler neighbourhood version (neighbourhood_request: a tent serves its own neighbourhood,
optionally its bordering ones) uses the same model.

Sets
    J     candidate tent sites, each in one neighbourhood g(j);   A  demand points;   S  the 5 scenarios, p_s = 0.2
    R(j)  points within reach of site j
Parameters
    D_as  expected people needing help at point a in scenario s (forecast, in people);   c  people per worker per day (30)
First stage (decided the evening before, the same in every scenario)
    x_j in {0,1}   tent opened at site j;      y_j in Z+   health workers at site j
Second stage (per scenario)
    f_jas >= 0     visitors from point a served at tent j (a in R(j));   z_as >= 0   visitors at a left unanswered

    min   sum_s p_s sum_a z_as                                         expected unanswered visitors
    s.t.  sum_j x_j <= N;   sum_{j: g(j)=u} x_j <= Q   for all u       tents available; at most Q per neighbourhood
          sum_s p_s sum_{a in R(j)} f_jas >= M x_j      for all j       an open tent serves at least M people
                                                                        (M = k x min calls per tent; default 0, Q = 1)
          sum_j y_j <= W;   x_j <= y_j <= W x_j                         workers; an open tent has >= 1, a closed none
          sum_{a in R(j)} f_jas <= c y_j                for all j, s    a tent treats at most c per worker
          sum_{j: a in R(j)} f_jas + z_as = D_as        for all a, s    every visitor is served or unanswered
          f_jas <= D_as x_j                             for all j, a, s only open tents serve (tightens the LP)

THREE-STEP SOLVE (lexicographic). Many plans can serve the same number of visitors (e.g. when every tent is
overwhelmed). Each step keeps the previous optimum and never costs a served visitor:
    1. min  expected unanswered visitors                  (the objective)
    2. max  sum_j (E[reachable visitors_j] / max) x_j      tents where the most people are within reach
    3. max  r,  c y_j >= E[reachable visitors_j] r         workers in proportion to need (max-min); all placed
"Answered" means a visitor within reach of an open tent that still had capacity; the medical effect of the visit is
outside the model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from ..config import PATHS, Paths

STATUS = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible",
          GRB.SUBOPTIMAL: "suboptimal", GRB.INTERRUPTED: "interrupted"}
WALK_MINUTES, WALK_SPEED_KMH, DETOUR = 15.0, 5.0, 1.3
TIEBREAK_TIME_S = 5.0  # per tie-breaking step (steps 2-3)


def walk_radius_km(minutes: float = WALK_MINUTES, speed_kmh: float = WALK_SPEED_KMH, detour: float = DETOUR) -> float:
    """Straight-line distance reachable on foot: street distance / detour factor (NYC grid ~1.3)."""
    return minutes / 60 * speed_kmh / detour


def _km(lat1, lon1, lat2, lon2) -> np.ndarray:
    k = np.cos(np.radians(40.7))
    return np.hypot((np.asarray(lat1)[:, None] - np.asarray(lat2)[None, :]) * 110.57,
                    (np.asarray(lon1)[:, None] - np.asarray(lon2)[None, :]) * 111.32 * k)


# --- request ----------------------------------------------------------------------------------------------------------
@dataclass
class DailyRequest:
    """One day's question: where to open ``n_tents`` tents and how to place ``n_workers`` health workers.

    demand   forecast EMS calls per demand point: [point, scenario, probability, expected_cases]
    sites    candidate tents: [site, name, neighbourhood] (at most one open per neighbourhood)
    reach    site -> points whose visitors can use it
    point_neighbourhood  point -> neighbourhood (for the report);  names: neighbourhood -> name
    Build it with walking_request (15-minute walk, parks) or neighbourhood_request (whole neighbourhoods).
    """
    n_tents: int
    n_workers: int
    demand: pd.DataFrame
    sites: pd.DataFrame
    reach: dict[str, list[str]]
    point_neighbourhood: dict[str, str]
    names: dict[str, str] | None = None
    patients_per_worker_day: float = 30.0
    visitors_per_call: float = 1.0  # demand is already in people (forecast output); >1 only for demand given in calls
    max_per_neighbourhood: int = 1  # Q: tents allowed in the same neighbourhood
    unlimited_capacity: bool = False  # True: a tent helps everyone within reach; workers are ignored (coverage model)
    min_calls_per_hospital: float = 0.0  # an open tent must serve >= this many calls' worth of people (x k), expected
    time_limit_s: float = 60.0
    mip_gap: float = 1e-4
    info: dict = field(default_factory=dict)


def neighbourhood_request(demand: pd.DataFrame, n_tents: int, n_workers: int,
                          adjacency: dict[str, list[str]] | None = None, names: dict[str, str] | None = None,
                          **kw) -> DailyRequest:
    """Whole-neighbourhood version: one candidate tent per neighbourhood; it serves its own neighbourhood and, if
    ``adjacency`` is given, the bordering ones. ``demand``: [uhf, scenario, probability, expected_cases]."""
    U = sorted(demand.uhf.astype(str).unique())
    adj = adjacency or {}
    sites = pd.DataFrame({"site": U, "name": [(names or {}).get(u, u) for u in U], "neighbourhood": U})
    reach = {u: [u] + [v for v in adj.get(u, []) if v in U] for u in U}
    d = demand.rename(columns={"uhf": "point"}).assign(point=lambda x: x.point.astype(str))
    return DailyRequest(n_tents, n_workers, d[["point", "scenario", "probability", "expected_cases"]], sites, reach,
                        {u: u for u in U}, names, **kw)


def neighbourhood_adjacency(paths: Paths = PATHS, tolerance_km: float = 0.05) -> dict[str, list[str]]:
    """uhf -> neighbourhoods sharing a border with it (boundaries within ``tolerance_km`` count as touching)."""
    from ..ingest.geography import _to_km, load_neighbourhoods
    N = load_neighbourhoods(paths)
    G = [_to_km(g) for g in N.geometry]
    ids = N.uhf.tolist()
    return {ids[i]: [ids[k] for k in range(len(ids)) if k != i and G[i].distance(G[k]) <= tolerance_km]
            for i in range(len(ids))}


def candidate_parks(points: pd.DataFrame, visitors: np.ndarray, radius_km: float, per_neighbourhood: int = 5,
                    paths: Paths = PATHS) -> pd.DataFrame:
    """NYC parks and plazas as tent sites: assigned to the neighbourhood they lie in, and the ``per_neighbourhood``
    best per neighbourhood by expected visitors within reach (at least 150 m apart)."""
    from shapely import contains_xy
    from ..ingest.geography import load_neighbourhoods
    from .candidates import park_points
    P = park_points(paths).copy()
    P["neighbourhood"] = None
    for r in load_neighbourhoods(paths).itertuples():
        P.loc[contains_xy(r.geometry, P.lon.to_numpy(), P.lat.to_numpy()) & P.neighbourhood.isna(), "neighbourhood"] = r.uhf
    P = P.dropna(subset=["neighbourhood"]).reset_index(drop=True)
    P["reachable_visitors"] = (_km(P.lat, P.lon, points.lat, points.lon) <= radius_km) @ visitors
    keep = []
    for _, g in P.sort_values("reachable_visitors", ascending=False).groupby("neighbourhood", sort=False):
        chosen = []
        for i, r in g.iterrows():
            if len(chosen) >= per_neighbourhood or r.reachable_visitors <= 0:
                break
            if all(_km([r.lat], [r.lon], [P.lat[j]], [P.lon[j]])[0, 0] > 0.15 for j in chosen):
                chosen.append(i)
        keep += chosen
    S = P.loc[keep].reset_index(drop=True)
    S["site"] = S.site_id.astype(str)
    return S[["site", "name", "neighbourhood", "lat", "lon", "kind", "reachable_visitors"]]


def candidate_grid(points: pd.DataFrame, radius_km: float, spacing_km: float = 0.4, paths: Paths = PATHS) -> pd.DataFrame:
    """Tent sites anywhere on land: a regular grid (every ``spacing_km``) inside the 42 neighbourhoods. Spots that
    reach no demand point, or exactly the same demand points as another spot in the same neighbourhood, are dropped
    (they can never do better), which keeps the MILP small."""
    from shapely import contains_xy
    from ..ingest.geography import load_neighbourhoods
    N = load_neighbourhoods(paths)
    k = np.cos(np.radians(40.7))
    minx, miny, maxx, maxy = np.array([g.bounds for g in N.geometry]).T
    lons = np.arange(minx.min(), maxx.max(), spacing_km / (111.32 * k))
    lats = np.arange(miny.min(), maxy.max(), spacing_km / 110.57)
    LON, LAT = (a.ravel() for a in np.meshgrid(lons, lats))
    rows = []
    for r in N.itertuples():
        inside = contains_xy(r.geometry, LON, LAT)
        rows.append(pd.DataFrame({"lat": LAT[inside], "lon": LON[inside], "neighbourhood": r.uhf, "nb_name": r.name}))
    G = pd.concat(rows, ignore_index=True)
    within = _km(G.lat, G.lon, points.lat, points.lon) <= radius_km
    G["key"] = [tuple(np.flatnonzero(w)) for w in within]
    G = G[G.key.map(len) > 0].drop_duplicates(["neighbourhood", "key"]).reset_index(drop=True)
    G["site"] = [f"grid{i}" for i in range(len(G))]
    G["name"] = [f"{n} · {la:.4f}, {lo:.4f}" for n, la, lo in zip(G.nb_name, G.lat, G.lon)]
    G["kind"] = "open location"
    return G[["site", "name", "neighbourhood", "lat", "lon", "kind"]]


def walking_request(fc, n_tents: int, n_workers: int, walk_minutes: float = WALK_MINUTES,
                    walk_speed_kmh: float = WALK_SPEED_KMH, detour: float = DETOUR, per_neighbourhood: int = 5,
                    visitors_per_call: float = 1.0, site_mode: str = "parks", grid_spacing_km: float = 0.4,
                    radius_km: float | None = None, paths: Paths = PATHS, **kw) -> DailyRequest:
    """15-minute-walk version from a service.DayForecast: demand at points inside each ZIP area (the area's forecast
    share of every scenario total, spread equally over up to 4 interior points). Tent sites: site_mode="parks" (the
    best NYC parks and plazas per neighbourhood) or "anywhere" (a grid every grid_spacing_km over all land)."""
    from ..ingest.geography import modzcta_to_neighbourhood
    from ..models.scenarios import demand_points
    radius = radius_km if radius_km else walk_radius_km(walk_minutes, walk_speed_kmh, detour)
    nb_of = modzcta_to_neighbourhood(paths)
    pts = demand_points(4).merge(fc.areas[["modzcta", "share"]], on="modzcta")
    pts = pts.assign(point=pts.area_id.astype(str), neighbourhood=pts.modzcta.map(nb_of))
    sc = fc.scenarios  # 5 scenarios: located_total = city total x known-location share
    demand = pd.concat([pd.DataFrame({"point": pts.point, "scenario": r.scenario, "probability": r.probability,
                                      "expected_cases": r.located_total * pts.share * pts.weight})
                        for r in sc.itertuples()], ignore_index=True)
    mean_visitors = visitors_per_call * (pts.share * pts.weight).to_numpy() * float((sc.located_total * sc.probability).sum())
    sites = candidate_parks(pts, mean_visitors, radius, per_neighbourhood, paths)
    if site_mode == "anywhere":  # the grid plus the park candidates, so "anywhere" can never do worse than parks
        sites = pd.concat([sites, candidate_grid(pts, radius, grid_spacing_km, paths)], ignore_index=True)
    within = _km(sites.lat, sites.lon, pts.lat, pts.lon) <= radius
    reach = {s: pts.point[within[i]].tolist() for i, s in enumerate(sites.site)}
    names = dict(zip(fc.neighbourhoods.uhf, fc.neighbourhoods.name))
    info = dict(point_latlon=pts[["point", "lat", "lon"]], walk_minutes=walk_minutes, walk_speed_kmh=walk_speed_kmh,
                detour=detour, radius_km=radius,
                demand_points=len(pts), candidate_sites=len(sites),
                visitors_within_reach_of_any_site=float(mean_visitors[within.any(0)].sum() / mean_visitors.sum()))
    return DailyRequest(n_tents, n_workers, demand, sites, reach, dict(zip(pts.point, pts.neighbourhood)), names,
                        visitors_per_call=visitors_per_call, info=info, **kw)


@dataclass
class DailyPlan:
    tents: pd.DataFrame  # site, name, neighbourhood, workers, capacity, reachable, answered (expected), utilisation
    neighbourhoods: pd.DataFrame  # neighbourhood, name, forecast_calls, expected_visitors, answered, unanswered
    scenarios: pd.DataFrame  # scenario, probability, visitors, answered, unanswered, answered_pct
    kpis: dict
    status: str
    mip_gap: float
    runtime_s: float
    stats: dict = field(default_factory=dict)


# --- model ------------------------------------------------------------------------------------------------------------
def solve(req: DailyRequest, env: gp.Env | None = None, verbose: bool = False) -> DailyPlan:
    if req.unlimited_capacity:
        return solve_coverage(req, env, verbose)
    t0 = time.time()
    calls = req.demand.pivot_table(index="point", columns="scenario", values="expected_cases", aggfunc="sum").fillna(0.0)
    prob = req.demand.groupby("scenario").probability.first().reindex(calls.columns).to_numpy()
    if not np.isclose(prob.sum(), 1.0):
        raise ValueError(f"scenario probabilities sum to {prob.sum()}, expected 1")
    Dm = calls * req.visitors_per_call  # tent visitors
    A, S = Dm.index.astype(str).tolist(), Dm.columns.tolist()
    J = req.sites.site.astype(str).tolist()
    group = dict(zip(J, req.sites.neighbourhood.astype(str)))
    Aset = set(A)
    reach = {j: [a for a in req.reach.get(j, []) if a in Aset] for j in J}
    arcs = [(j, a) for j in J for a in reach[j]]
    d = {(a, s): float(Dm.loc[a, s]) for a in A for s in S}
    p = dict(zip(S, prob))
    c, W = req.patients_per_worker_day, req.n_workers

    m = gp.Model("daily_tents", env=env) if env is not None else gp.Model("daily_tents")
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit, m.Params.MIPGap = req.time_limit_s, req.mip_gap
    x = m.addVars(J, vtype=GRB.BINARY, name="x")
    y = m.addVars(J, vtype=GRB.INTEGER, lb=0, ub=W, name="y")
    f = m.addVars([(j, a, s) for (j, a) in arcs for s in S], lb=0.0, name="f")
    z = m.addVars(A, S, lb=0.0, name="z")
    by_point = {a: [] for a in A}
    for j, a in arcs:
        by_point[a].append(j)

    expected_unanswered = gp.quicksum(p[s] * z[a, s] for a in A for s in S)
    m.setObjective(expected_unanswered, GRB.MINIMIZE)
    m.addConstr(x.sum() <= req.n_tents, "tents")
    for g in set(group.values()):
        m.addConstr(gp.quicksum(x[j] for j in J if group[j] == g) <= req.max_per_neighbourhood,
                    f"max_per_neighbourhood[{g}]")
    min_people = req.min_calls_per_hospital * req.visitors_per_call
    if min_people > 0:  # an open tent must be worth opening: expected people served >= M
        m.addConstrs((gp.quicksum(p[s] * f[j, a, s] for a in reach[j] for s in S) >= min_people * x[j] for j in J),
                     "min_people_per_tent")
    m.addConstr(y.sum() <= W, "workers")
    m.addConstrs((y[j] >= x[j] for j in J), "min_one_worker")
    m.addConstrs((y[j] <= W * x[j] for j in J), "workers_only_at_open_tents")
    m.addConstrs((gp.quicksum(f[j, a, s] for a in reach[j]) <= c * y[j] for j in J for s in S), "capacity")
    m.addConstrs((gp.quicksum(f[j, a, s] for j in by_point[a]) + z[a, s] == d[a, s] for a in A for s in S), "demand")
    m.addConstrs((f[j, a, s] <= d[a, s] * x[j] for (j, a) in arcs for s in S), "open_tents_only")

    m.optimize()  # step 1: fewest expected unanswered visitors
    if m.SolCount and m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        tol = lambda v: 1e-8 * max(1.0, abs(v))  # later steps may not give up a served visitor
        v_star = m.ObjVal
        m.addConstr(expected_unanswered <= v_star + tol(v_star), "keep_step1")
        need = {j: sum(p[s] * d[a, s] for a in reach[j] for s in S) for j in J}  # expected visitors within reach
        top = max(need.values()) or 1.0
        tent_need = gp.quicksum(need[j] / top * x[j] for j in J)
        m.Params.MIPGap = 0.0  # tie-breaking steps: exact when the model is small,
        m.Params.TimeLimit = TIEBREAK_TIME_S  # but capped in time (they never change the number of people helped)
        m.setObjective(tent_need, GRB.MAXIMIZE)
        m.optimize()  # step 2: among equally good plans, tents where the most people are within reach
        u_star = m.ObjVal
        m.addConstr(tent_need >= u_star - tol(u_star), "keep_step2")
        r = m.addVar(lb=0.0, name="min_capacity_to_need")
        m.addConstrs((c * y[j] >= need[j] * r - c * W * (1 - x[j]) for j in J if need[j] > 0), "proportional")
        m.setObjective(r, GRB.MAXIMIZE)
        m.optimize()  # step 3a: workers in proportion to need
        m.addConstr(r >= r.X * (1 - 1e-9), "keep_step3")
        m.setObjective(y.sum(), GRB.MAXIMIZE)
        m.optimize()  # step 3b: then place every remaining worker

    status = STATUS.get(m.Status, str(m.Status))
    if m.SolCount == 0:
        raise RuntimeError(f"no solution found (status {status})")
    return _report(req, m, x, y, f, z, A, S, J, reach, d, p, calls, group, status, t0)


def _report(req, m, x, y, f, z, A, S, J, reach, d, p, calls, group, status, t0) -> DailyPlan:
    names = req.names or {}
    c = req.patients_per_worker_day
    exp = lambda vals: sum(p[s] * v for s, v in vals.items())  # expectation over scenarios
    fv = {k: v.X for k, v in f.items()}
    zv = {(a, s): z[a, s].X for a in A for s in S}
    site = req.sites.assign(site=req.sites.site.astype(str)).set_index("site")
    rows = []
    for j in J:
        if x[j].X < 0.5:
            continue
        w = int(round(y[j].X))
        answered = {s: sum(fv[j, a, s] for a in reach[j]) for s in S}
        reachable = exp({s: sum(d[a, s] for a in reach[j]) for s in S})
        row = dict(site=j, name=site.loc[j, "name"], neighbourhood=group[j],
                   neighbourhood_name=names.get(group[j], group[j]), workers=w, capacity=c * w,
                   reachable_visitors=reachable, answered=exp(answered), answered_busiest=answered[S[-1]],
                   utilisation=exp(answered) / (c * w))
        if "lat" in site.columns:
            row.update(lat=float(site.loc[j, "lat"]), lon=float(site.loc[j, "lon"]))
        rows.append(row)
    cols = ["site", "name", "neighbourhood", "neighbourhood_name", "workers", "capacity", "reachable_visitors",
            "answered", "answered_busiest", "utilisation"] + (["lat", "lon"] if "lat" in site.columns else [])
    T = pd.DataFrame(rows, columns=cols).sort_values("answered", ascending=False).reset_index(drop=True)

    pn = pd.Series(req.point_neighbourhood)
    P = pd.DataFrame({"point": A, "neighbourhood": pn.reindex(A).to_numpy(),
                      "forecast_calls": [exp({s: float(calls.loc[a, s]) for s in S}) for a in A],
                      "expected_visitors": [exp({s: d[a, s] for s in S}) for a in A],
                      "unanswered": [exp({s: zv[a, s] for s in S}) for a in A]})
    nb = P.groupby("neighbourhood", as_index=False)[["forecast_calls", "expected_visitors", "unanswered"]].sum()
    nb["answered"] = nb.expected_visitors - nb.unanswered
    nb["answered_pct"] = 100 * nb.answered / nb.expected_visitors.where(nb.expected_visitors > 0)
    nb.insert(1, "name", nb.neighbourhood.map(lambda u: names.get(u, u)))
    nb = nb[["neighbourhood", "name", "forecast_calls", "expected_visitors", "answered", "unanswered", "answered_pct"]]
    nb = nb.sort_values("unanswered", ascending=False).reset_index(drop=True)

    sc = pd.DataFrame({"scenario": S, "probability": [p[s] for s in S],
                       "visitors": [sum(d[a, s] for a in A) for s in S],
                       "unanswered": [sum(zv[a, s] for a in A) for s in S]})
    sc["answered"] = sc.visitors - sc.unanswered
    sc["answered_pct"] = 100 * sc.answered / sc.visitors
    sc = sc[["scenario", "probability", "visitors", "answered", "unanswered", "answered_pct"]]
    vis = float((sc.visitors * sc.probability).sum())
    unans = float((sc.unanswered * sc.probability).sum())
    open_ = [j for j in J if x[j].X > 0.5]
    reachable_pts = {a for j in open_ for a in reach[j]}
    kpis = dict(visitors_per_call=req.visitors_per_call, expected_visitors=vis, expected_answered=vis - unans,
                expected_unanswered=unans, answered_pct=100 * (vis - unans) / vis if vis else 0.0,
                within_reach_pct=100 * float(P[P.point.isin(reachable_pts)].expected_visitors.sum()) / vis if vis else 0.0,
                tents_open=len(open_), workers_placed=int(sum(round(y[j].X) for j in J)),
                tents_available=req.n_tents, workers_available=req.n_workers,
                capacity_per_day=c * sum(round(y[j].X) for j in J),
                **{k: v for k, v in req.info.items() if k != "point_latlon"})
    stats = dict(sites=len(J), demand_points=len(A), arcs=sum(len(v) for v in reach.values()), binaries=len(J), integers=len(J),
                 continuous=len(f) + len(z), constraints=m.NumConstrs)
    return DailyPlan(T, nb, sc, kpis, status, m.MIPGap if m.IsMIP else 0.0, time.time() - t0, stats)


# --- coverage version: unlimited capacity ---------------------------------------------------------------------------
def solve_coverage(req: DailyRequest, env: gp.Env | None = None, verbose: bool = False) -> DailyPlan:
    r"""Maximum-coverage version (unlimited_capacity=True): a tent helps everyone within reach; workers are ignored.

        max  sum_a E[D_a] u_a          E[D_a] = sum_s p_s D_as   (expected people at point a)
        s.t. u_a <= sum_{j: a in R(j)} x_j      for all a     a point is covered only if an open tent reaches it
             sum_j x_j <= N;   sum_{j: g(j)=u} x_j <= Q;   x_j in {0,1};   0 <= u_a <= 1
    Without capacity the scenarios enter only through their expectation, so the per-scenario flows are not needed.
    Tie-break (keeps the optimum): tents where the most people are within reach. For the report each covered point is
    attributed to its nearest open tent, so per-tent "answered" never counts a person twice.
    """
    t0 = time.time()
    calls = req.demand.pivot_table(index="point", columns="scenario", values="expected_cases", aggfunc="sum").fillna(0.0)
    prob = req.demand.groupby("scenario").probability.first().reindex(calls.columns).to_numpy()
    if not np.isclose(prob.sum(), 1.0):
        raise ValueError(f"scenario probabilities sum to {prob.sum()}, expected 1")
    Dm = calls * req.visitors_per_call
    A, S = Dm.index.astype(str).tolist(), Dm.columns.tolist()
    E = dict(zip(A, Dm.to_numpy() @ prob))
    J = req.sites.site.astype(str).tolist()
    group = dict(zip(J, req.sites.neighbourhood.astype(str)))
    Aset = set(A)
    reach = {j: [a for a in req.reach.get(j, []) if a in Aset] for j in J}
    by_point = {a: [] for a in A}
    for j in J:
        for a in reach[j]:
            by_point[a].append(j)

    m = gp.Model("daily_tents_coverage", env=env) if env is not None else gp.Model("daily_tents_coverage")
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit, m.Params.MIPGap = req.time_limit_s, req.mip_gap
    x = m.addVars(J, vtype=GRB.BINARY, name="x")
    u = m.addVars(A, lb=0.0, ub=1.0, name="u")
    covered = gp.quicksum(E[a] * u[a] for a in A)
    m.setObjective(covered, GRB.MAXIMIZE)
    m.addConstr(x.sum() <= req.n_tents, "tents")
    for g in set(group.values()):
        m.addConstr(gp.quicksum(x[j] for j in J if group[j] == g) <= req.max_per_neighbourhood, f"max_per_nb[{g}]")
    m.addConstrs((u[a] <= gp.quicksum(x[j] for j in by_point[a]) for a in A), "covered_only_if_reached")
    m.optimize()  # step 1: most expected people within reach of an open tent
    if m.SolCount and m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        v = m.ObjVal
        m.addConstr(covered >= v - 1e-8 * max(1.0, v), "keep_step1")
        need = {j: sum(E[a] for a in reach[j]) for j in J}
        top = max(need.values()) or 1.0
        m.setObjective(gp.quicksum(need[j] / top * x[j] for j in J), GRB.MAXIMIZE)
        m.Params.MIPGap, m.Params.TimeLimit = 0.0, TIEBREAK_TIME_S
        m.optimize()  # step 2: on ties, tents where the most people are within reach
    status = STATUS.get(m.Status, str(m.Status))
    if m.SolCount == 0:
        raise RuntimeError(f"no solution found (status {status})")

    names = req.names or {}
    open_ = [j for j in J if x[j].X > 0.5]
    site = req.sites.assign(site=req.sites.site.astype(str)).set_index("site")
    cov = {a for j in open_ for a in reach[j]}
    owner = {}  # nearest open tent of each covered point (report only)
    if open_ and "lat" in site.columns:
        pts = pd.DataFrame({"point": list(cov)})
        latlon = req.info.get("point_latlon")
        if latlon is not None:
            P = latlon.set_index("point").reindex(pts.point)
            Dk = _km(P.lat, P.lon, site.loc[open_, "lat"], site.loc[open_, "lon"])
            for i, a in enumerate(pts.point):
                cand = [k for k, j in enumerate(open_) if a in set(reach[j])]
                owner[a] = open_[min(cand, key=lambda k: Dk[i, k])]
    for a in cov:
        owner.setdefault(a, next(j for j in open_ if a in set(reach[j])))
    rows = []
    for j in open_:
        mine = [a for a in cov if owner[a] == j]
        row = dict(site=j, name=site.loc[j, "name"], neighbourhood=group[j],
                   neighbourhood_name=names.get(group[j], group[j]), workers=0, capacity=np.nan,
                   reachable_visitors=sum(E[a] for a in reach[j]), answered=sum(E[a] for a in mine),
                   answered_busiest=float(Dm.loc[mine, S[-1]].sum()) if mine else 0.0, utilisation=np.nan)
        if "lat" in site.columns:
            row.update(lat=float(site.loc[j, "lat"]), lon=float(site.loc[j, "lon"]))
        rows.append(row)
    cols = ["site", "name", "neighbourhood", "neighbourhood_name", "workers", "capacity", "reachable_visitors",
            "answered", "answered_busiest", "utilisation"] + (["lat", "lon"] if "lat" in site.columns else [])
    T = pd.DataFrame(rows, columns=cols).sort_values("answered", ascending=False).reset_index(drop=True)
    pn = pd.Series(req.point_neighbourhood)
    P = pd.DataFrame({"point": A, "neighbourhood": pn.reindex(A).to_numpy(),
                      "forecast_calls": calls.to_numpy() @ prob, "expected_visitors": [E[a] for a in A],
                      "unanswered": [0.0 if a in cov else E[a] for a in A]})
    nb = P.groupby("neighbourhood", as_index=False)[["forecast_calls", "expected_visitors", "unanswered"]].sum()
    nb["answered"] = nb.expected_visitors - nb.unanswered
    nb["answered_pct"] = 100 * nb.answered / nb.expected_visitors.where(nb.expected_visitors > 0)
    nb.insert(1, "name", nb.neighbourhood.map(lambda g: names.get(g, g)))
    nb = nb[["neighbourhood", "name", "forecast_calls", "expected_visitors", "answered", "unanswered", "answered_pct"]]
    nb = nb.sort_values("unanswered", ascending=False).reset_index(drop=True)
    covered_mask = Dm.index.isin(list(cov))
    sc = pd.DataFrame({"scenario": S, "probability": prob, "visitors": Dm.sum().to_numpy(),
                       "answered": Dm[covered_mask].sum().to_numpy()})
    sc["unanswered"] = sc.visitors - sc.answered
    sc["answered_pct"] = 100 * sc.answered / sc.visitors.where(sc.visitors > 0)
    sc = sc[["scenario", "probability", "visitors", "answered", "unanswered", "answered_pct"]]
    vis, ans = float(sum(E.values())), float(sum(E[a] for a in cov))
    kpis = dict(visitors_per_call=req.visitors_per_call, expected_visitors=vis, expected_answered=ans,
                expected_unanswered=vis - ans, answered_pct=100 * ans / vis if vis else 0.0,
                within_reach_pct=100 * ans / vis if vis else 0.0, tents_open=len(open_), workers_placed=0,
                tents_available=req.n_tents, workers_available=0, capacity_per_day=np.nan, unlimited_capacity=True,
                **{k: v for k, v in req.info.items() if k != "point_latlon"})
    stats = dict(sites=len(J), demand_points=len(A), arcs=sum(len(v) for v in reach.values()), binaries=len(J),
                 continuous=len(A), constraints=m.NumConstrs)
    return DailyPlan(T, nb, sc, kpis, status, m.MIPGap if m.IsMIP else 0.0, time.time() - t0, stats)
