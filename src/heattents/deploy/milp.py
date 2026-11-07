r"""Daily cooling-tent and doctor deployment: a two-stage stochastic MILP solved with Gurobi (SAA).

Sets
    J            candidate tent sites                     K   doctors
    S            shifts, shift s covers hours [a_s, b_s)  T   time blocks (single hours, or user hour_blocks)
    A            demand areas (MODZCTA)                   W   forecast scenarios (equally weighted samples)
    R = {(a,j): dist(a,j) <= walk_radius}                     walkable area-site arcs
    S_k ⊆ S      shifts doctor k is available for;  O = {(s,s'): s != s' overlap}

Parameters
    D_{wat}      expected heat-illness cases in area a, block t, scenario w
    rho_{aj}     attractiveness of site j for area a: exp(-dist/decay) (1 if no decay)
    c_{st}       hours of block t covered by shift s
    mu           patients a doctor can treat per hour;   M_j  max doctors at site j
    m_min        min doctors in every shift a tent operates;   N  number of tents;   L_k  max shifts of doctor k

First-stage variables (decided before the day)
    x_j   in {0,1}   tent j is set up
    o_js  in {0,1}   tent j operates in shift s
    y_js  in Z+      doctors at tent j in shift s
    v_ks  in {0,1}   doctor k works shift s          (v_ks = 0 if s not in S_k)
Second-stage variables (per scenario)
    f_wajt >= 0      cases from area a treated at tent j in block t, scenario w    ((a,j) in R)

    max  (1/|W|) sum_{w,(a,j),t} rho_aj f_wajt  -  eps * (sum_j x_j + sum_{k,s} v_ks)

    sum_j x_j <= N                                                   tents
    o_js <= x_j,   sum_s o_js >= x_j                                 an open tent operates in >= 1 shift
    m_min o_js <= y_js <= M_j o_js                                   staffing of an operating shift
    sum_j y_js = sum_k v_ks                                          for all s   doctors on shift = doctors placed
    sum_s v_ks <= L_k,    v_ks + v_ks' <= 1  for (s,s') in O         for all k   no double booking
    sum_j f_wajt <= D_wat                                            for all w,a,t   demand
    sum_a f_wajt <= mu * sum_s c_st y_js                             for all w,j,t   treatment capacity
    f_wajt <= D_wat * sum_{s: c_st > 0} o_js                         for all w,(a,j),t   strong linking

The last family is implied by the others for integer solutions but tightens the LP relaxation considerably
(the "strong" facility-location formulation): without it the relaxation spreads fractional staff thinly over
many sites and the bound stays far from the incumbent.

Doctors differ only in availability, so the model places *numbers* of doctors per (site, shift) and links
them to individual doctors through v_ks; any doctor working shift s can staff any site in that shift, so the
named roster is recovered exactly after solving. This keeps the binaries at |K||S| + 2|J||S| + |J| instead of
|K||J||S|. eps is a tiny tie-breaker: it never trades off a treated case, it only avoids idle tents/doctors.
The objective counts attractiveness-weighted treated cases ("expected cases reached"); the medical effect of
a visit is outside the model.
"""
from __future__ import annotations

import time
from collections import defaultdict

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .model import DeploymentPlan, DeploymentRequest, Doctor, Shift, Site

EPS = 1e-5
STATUS = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible",
          GRB.SUBOPTIMAL: "suboptimal", GRB.INTERRUPTED: "interrupted"}


def _km(lat1, lon1, lat2, lon2) -> np.ndarray:
    k = np.cos(np.radians(40.7))
    dy = (np.asarray(lat1)[:, None] - np.asarray(lat2)[None, :]) * 110.57
    dx = (np.asarray(lon1)[:, None] - np.asarray(lon2)[None, :]) * 111.32 * k
    return np.hypot(dx, dy)


def _blocks(req: DeploymentRequest) -> list[tuple[int, int]]:
    if req.hour_blocks:
        return sorted(req.hour_blocks)
    return [(h, h + 1) for h in range(24)]


def solve(req: DeploymentRequest, env: gp.Env | None = None, verbose: bool = False) -> DeploymentPlan:
    t0 = time.time()
    shifts = list(req.shifts)
    shift_idx = {s.name: i for i, s in enumerate(shifts)}
    sites, doctors = list(req.sites), list(req.doctors)
    blocks = _blocks(req)
    nJ, nS = len(sites), len(shifts)

    # ---- demand: (w, a, t) -> cases -----------------------------------------------------------------------
    d = req.demand.copy()
    scenarios = sorted(d.scenario.unique())
    nW = len(scenarios)
    w_of = {w: i for i, w in enumerate(scenarios)}
    hour_to_block = {}
    for t, (a, b) in enumerate(blocks):
        for h in range(a, b):
            hour_to_block[h] = t
    d = d[d.hour.isin(hour_to_block.keys()) & (d.expected_cases > 0)]
    areas = d.drop_duplicates("area_id")[["area_id", "lat", "lon"]].reset_index(drop=True)
    a_of = {a: i for i, a in enumerate(areas.area_id)}
    d["w"], d["a"], d["t"] = d.scenario.map(w_of), d.area_id.map(a_of), d.hour.map(hour_to_block)
    D = d.groupby(["w", "a", "t"]).expected_cases.sum()
    total_cases = req.demand.expected_cases.clip(lower=0).sum() / max(nW, 1)

    # ---- walkable arcs and attractiveness --------------------------------------------------------------------
    dist = _km(areas.lat, areas.lon, [s.lat for s in sites], [s.lon for s in sites]) if len(areas) else np.zeros((0, nJ))
    ai, aj = np.nonzero(dist <= req.walk_radius_km)
    rho = np.ones(len(ai)) if req.distance_decay_km is None else np.exp(-dist[ai, aj] / req.distance_decay_km)
    arcs_of_area = defaultdict(list)
    for k, a in enumerate(ai):
        arcs_of_area[int(a)].append(k)
    cover = np.zeros((nS, len(blocks)))  # c_st
    for s, sh in enumerate(shifts):
        for t, (a, b) in enumerate(blocks):
            cover[s, t] = max(0, min(b, sh.end_hour) - max(a, sh.start_hour))

    m = gp.Model("deployment", env=env) if env is not None else gp.Model("deployment")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = req.time_limit_s
    m.Params.MIPGap = req.mip_gap

    # ---- first stage -----------------------------------------------------------------------------------------
    x = m.addVars(nJ, vtype=GRB.BINARY, name="x")
    o = m.addVars(nJ, nS, vtype=GRB.BINARY, name="o")
    y = m.addVars(nJ, nS, vtype=GRB.INTEGER, lb=0, name="y")
    avail = [(k, shift_idx[s]) for k, doc in enumerate(doctors) for s in doc.available_shifts if s in shift_idx]
    v = m.addVars(avail, vtype=GRB.BINARY, name="v")
    m.addConstr(x.sum() <= req.n_tents, name="n_tents")
    for j, site in enumerate(sites):
        m.addConstr(o.sum(j, "*") >= x[j], name=f"operates_{j}")
        for s in range(nS):
            m.addConstr(o[j, s] <= x[j])
            m.addConstr(y[j, s] >= req.min_doctors_per_open_tent * o[j, s])
            m.addConstr(y[j, s] <= site.max_doctors * o[j, s])
    for s in range(nS):
        m.addConstr(y.sum("*", s) == v.sum("*", s), name=f"doctors_on_shift_{s}")
    for k, doc in enumerate(doctors):
        ks = [s for (kk, s) in avail if kk == k]
        if ks:
            m.addConstr(gp.quicksum(v[k, s] for s in ks) <= doc.max_shifts, name=f"max_shifts_{k}")
        for i, s1 in enumerate(ks):
            for s2 in ks[i + 1:]:
                if shifts[s1].overlaps(shifts[s2]):
                    m.addConstr(v[k, s1] + v[k, s2] <= 1, name=f"overlap_{k}_{s1}_{s2}")

    # ---- second stage: one flow per (scenario, arc, block) with positive demand --------------------------------
    keys = [(w, k, t) for (w, a, t) in D.index for k in arcs_of_area.get(a, [])]
    f = m.addVars(keys, lb=0.0, name="f")
    by_wat = defaultdict(list)
    by_wjt = defaultdict(list)
    for (w, k, t) in keys:
        by_wat[(w, int(ai[k]), t)].append(f[w, k, t])
        by_wjt[(w, int(aj[k]), t)].append(f[w, k, t])
    for (w, a, t), vars_ in by_wat.items():
        m.addLConstr(gp.LinExpr([1.0] * len(vars_), vars_), GRB.LESS_EQUAL, float(D[(w, a, t)]))
    for (w, j, t), vars_ in by_wjt.items():
        cap = gp.LinExpr([req.patients_per_doctor_hour * cover[s, t] for s in range(nS) if cover[s, t] > 0],
                         [y[j, s] for s in range(nS) if cover[s, t] > 0])
        m.addLConstr(gp.LinExpr([1.0] * len(vars_), vars_) - cap, GRB.LESS_EQUAL, 0.0)
    for (w, k, t) in keys:  # strong linking: no flow to a site that does not operate in a shift covering t
        j = int(aj[k])
        ops = [o[j, s] for s in range(nS) if cover[s, t] > 0]
        m.addLConstr(f[w, k, t] - float(D[(w, int(ai[k]), t)]) * gp.quicksum(ops), GRB.LESS_EQUAL, 0.0)

    obj =gp.LinExpr([rho[k] / nW for (w, k, t) in keys], [f[key] for key in keys])
    m.setObjective(obj - EPS * (x.sum() + v.sum()), GRB.MAXIMIZE)
    build_s = time.time() - t0
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"deployment model: no solution (status {m.Status})")

    # ---- read solution ---------------------------------------------------------------------------------------
    X = np.array([x[j].X > 0.5 for j in range(nJ)])
    Y = np.array([[int(round(y[j, s].X)) for s in range(nS)] for j in range(nJ)])
    roster = defaultdict(list)
    for (k, s) in avail:
        if v[k, s].X > 0.5:
            roster[s].append(doctors[k].doctor_id)
    assignments = []
    for s in range(nS):
        pool = list(roster[s])
        for j in range(nJ):
            for _ in range(Y[j, s]):
                assignments.append(dict(doctor_id=pool.pop(0), site_id=sites[j].site_id, shift=shifts[s].name))
    fv = {key: var.X for key, var in f.items()}
    raw_w = np.zeros(nW)  # unweighted treated cases per scenario (what a clinician would count)
    for (w, k, t), val in fv.items():
        raw_w[w] += val
    per_w = np.zeros(nW)
    site_served = np.zeros(nJ)
    area_t = defaultdict(float)
    for (w, k, t), val in fv.items():
        eff = rho[k] * val
        per_w[w] += eff
        site_served[aj[k]] += eff / nW
        area_t[(int(ai[k]), t)] += eff / nW
    dem_w = D.groupby(level="w").sum().reindex(range(nW), fill_value=0).to_numpy()
    dem_at = D.groupby(level=["a", "t"]).sum() / nW
    served = pd.DataFrame([dict(area_id=areas.area_id[a], hour=blocks[t][0], hour_end=blocks[t][1],
                                expected_served=area_t.get((a, t), 0.0), expected_demand=val)
                           for (a, t), val in dem_at.items()])
    open_rows = []
    for j in np.flatnonzero(X):
        st = sites[j]
        open_rows.append(dict(site_id=st.site_id, name=st.name, lat=st.lat, lon=st.lon, kind=st.kind,
                              **{f"doctors_{sh.name}": int(Y[j, s]) for s, sh in enumerate(shifts)},
                              expected_served=float(site_served[j])))
    exp_served = float(per_w.mean()) if nW else 0.0
    kpis = dict(expected_cases=float(total_cases), expected_served=exp_served,
                served_pct=float(100 * exp_served / total_cases) if total_cases > 0 else 0.0,
                scenario_served_min=float(per_w.min()) if nW else 0.0, scenario_served_max=float(per_w.max()) if nW else 0.0,
                scenario_served_pct_min=float(np.min(100 * per_w / np.maximum(dem_w, 1e-12))) if nW else 0.0,
                scenario_served_pct_max=float(np.max(100 * per_w / np.maximum(dem_w, 1e-12))) if nW else 0.0,
                expected_treated=float(raw_w.mean()) if nW else 0.0,
                treated_pct=float(100 * raw_w.mean() / total_cases) if nW and total_cases > 0 else 0.0,
                tents_open=int(X.sum()), doctors_used=len({a["doctor_id"] for a in assignments}),
                doctor_shifts=len(assignments), n_scenarios=nW)
    return DeploymentPlan(
        open_sites=pd.DataFrame(open_rows), assignments=pd.DataFrame(assignments, columns=["doctor_id", "site_id", "shift"]),
        served=served, kpis=kpis, status=STATUS.get(m.Status, str(m.Status)),
        mip_gap=m.MIPGap if m.IsMIP else 0.0, runtime_s=time.time() - t0,
        stats=dict(variables=m.NumVars, binaries=m.NumBinVars, integers=m.NumIntVars, constraints=m.NumConstrs,
                   arcs=len(ai), flow_vars=len(keys), build_s=round(build_s, 2), solve_s=round(m.Runtime, 2)))


# ---- synthetic instance for benchmarks and demos -------------------------------------------------------------
DEFAULT_SHIFTS = [Shift("early", 8, 14), Shift("mid", 11, 17), Shift("late", 14, 20)]
HOURLY_PROFILE = np.array([.17, .11, .09, .06, .06, .04, .07, .10, .24, .46, .70, 1.07, 1.57, 1.97, 2.16, 2.09,
                           1.86, 1.47, 1.17, .74, .53, .42, .31, .23])  # NYC EMS HEAT calls by hour (relative)


def synthetic_request(n_areas: int = 177, n_sites: int = 150, n_doctors: int = 22, n_scenarios: int = 20,
                      n_tents: int = 10, total_daily_cases: float = 600.0, seed: int = 0, **kw) -> DeploymentRequest:
    """Random NYC-like instance: areas and sites in the city bounding box, lognormal area intensities, the
    observed EMS hourly profile, and a lognormal city-wide scenario multiplier (forecast uncertainty)."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(40.57, 40.89, n_areas)
    lon = rng.uniform(-74.03, -73.76, n_areas)
    intensity = rng.lognormal(0, 0.8, n_areas)
    intensity /= intensity.sum()
    prof = HOURLY_PROFILE / HOURLY_PROFILE.sum()
    mult = rng.lognormal(-0.08, 0.4, n_scenarios)
    rows = []
    for w in range(n_scenarios):
        cases = total_daily_cases * mult[w] * intensity[:, None] * prof[None, :]
        for a in range(n_areas):
            for h in range(24):
                rows.append((f"A{a:03d}", lat[a], lon[a], h, w, cases[a, h]))
    demand = pd.DataFrame(rows, columns=["area_id", "lat", "lon", "hour", "scenario", "expected_cases"])
    # sites near areas (parks are spread through neighbourhoods)
    base = rng.integers(0, n_areas, n_sites)
    sites = [Site(f"S{j:03d}", f"Site {j}", lat[b] + rng.normal(0, 0.004), lon[b] + rng.normal(0, 0.005))
             for j, b in enumerate(base)]
    names = [s.name for s in DEFAULT_SHIFTS]
    doctors = [Doctor(f"D{k:02d}", list(rng.choice(names, size=rng.integers(1, 3), replace=False)),
                      max_shifts=int(rng.integers(1, 3))) for k in range(n_doctors)]
    return DeploymentRequest(n_tents=n_tents, sites=sites, doctors=doctors, shifts=list(DEFAULT_SHIFTS),
                             demand=demand, **kw)


if __name__ == "__main__":
    for blocks in (None, [(0, 8), (8, 11), (11, 14), (14, 17), (17, 20), (20, 24)]):
        req = synthetic_request(hour_blocks=blocks)
        plan = solve(req)
        print("blocks" if blocks else "hourly", plan.status, f"{plan.runtime_s:.1f}s", plan.stats,
              {k: round(v, 2) if isinstance(v, float) else v for k, v in plan.kpis.items()})
