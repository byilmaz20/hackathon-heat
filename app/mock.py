"""MOCK deployment interface + greedy heuristic.

Mirrors the interface of ``heattents.deploy.milp`` (dataclasses and ``solve``) so the app runs before the
Gurobi model exists. The greedy solver is NOT optimal - it only exists to exercise the UI end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 111.0
COS_NYC = float(np.cos(np.radians(40.7)))


@dataclass(frozen=True)
class Shift:
    name: str
    start_hour: int  # inclusive, 0-23
    end_hour: int  # exclusive, 1-24

    @property
    def hours(self) -> list[int]:
        return list(range(self.start_hour, self.end_hour))


@dataclass(frozen=True)
class Doctor:
    doctor_id: str
    available_shifts: tuple[str, ...]
    max_shifts: int = 1


@dataclass(frozen=True)
class Site:
    site_id: str
    name: str
    lat: float
    lon: float
    max_doctors: int = 4
    kind: str = "cooling_site"


@dataclass
class DeploymentRequest:
    n_tents: int
    sites: list[Site]
    doctors: list[Doctor]
    shifts: list[Shift]
    demand: pd.DataFrame  # [area_id, lat, lon, hour, scenario, expected_cases]
    walk_radius_km: float = 1.0
    distance_decay_km: float = 0.6
    patients_per_doctor_hour: float = 3.0
    min_doctors_per_open_tent: int = 1
    time_limit_s: float = 60.0
    mip_gap: float = 0.005


@dataclass
class DeploymentPlan:
    open_sites: pd.DataFrame  # [site_id, name, lat, lon, kind, doctors_total, expected_served]
    assignments: pd.DataFrame  # [doctor_id, site_id, site_name, shift, start_hour, end_hour]
    served: pd.DataFrame  # [area_id, hour, expected_cases, expected_served]
    kpis: dict
    status: str
    mip_gap: float
    runtime_s: float
    stats: dict = field(default_factory=dict)


def distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    dy = (lat1[:, None] - lat2[None, :]) * KM_PER_DEG_LAT
    dx = (lon1[:, None] - lon2[None, :]) * KM_PER_DEG_LAT * COS_NYC
    return np.hypot(dx, dy)


def solve(req: DeploymentRequest) -> DeploymentPlan:
    """Greedy: open sites by decayed reachable demand, spread doctors by demand, then fill capacity nearest-first."""
    import time
    t0 = time.time()
    dem = req.demand.groupby(["area_id", "hour"], as_index=False).agg(
        lat=("lat", "first"), lon=("lon", "first"), expected_cases=("expected_cases", "mean"))
    areas = dem.drop_duplicates("area_id")[["area_id", "lat", "lon"]].reset_index(drop=True)
    S = pd.DataFrame([s.__dict__ for s in req.sites])
    D = distance_km(areas.lat, areas.lon, S.lat, S.lon)
    reach = np.where(D <= req.walk_radius_km, np.exp(-D / max(req.distance_decay_km, 1e-6)), 0.0)
    area_tot = dem.groupby("area_id").expected_cases.sum().reindex(areas.area_id).to_numpy()

    # 1) open sites greedily on residual decayed demand
    residual, chosen = area_tot.copy(), []
    for _ in range(min(req.n_tents, len(S))):
        score = residual @ reach
        score[chosen] = -1
        j = int(np.argmax(score))
        if score[j] <= 0:
            break
        chosen.append(j)
        residual = residual * (1 - np.clip(reach[:, j], 0, 1))

    # 2) doctors per shift, proportional to demand reachable by each open site in the shift's hours
    rows, used = [], {d.doctor_id: 0 for d in req.doctors}
    load = {(j, sh.name): 0 for j in chosen for sh in req.shifts}
    for sh in req.shifts:
        hdem = dem[dem.hour.isin(sh.hours)].groupby("area_id").expected_cases.sum().reindex(areas.area_id).fillna(0).to_numpy()
        site_need = {j: float(hdem @ reach[:, j]) for j in chosen}
        avail = [d for d in req.doctors if sh.name in d.available_shifts and used[d.doctor_id] < d.max_shifts]
        for d in avail:
            open_ = [j for j in chosen if load[(j, sh.name)] < S.max_doctors.iloc[j]]
            if not open_:
                break
            # first satisfy the minimum per tent, then the largest need per doctor
            under = [j for j in open_ if load[(j, sh.name)] < req.min_doctors_per_open_tent]
            pool = under or open_
            j = max(pool, key=lambda k: site_need[k] / (1 + load[(k, sh.name)]))
            load[(j, sh.name)] += 1
            used[d.doctor_id] += 1
            rows.append(dict(doctor_id=d.doctor_id, site_id=S.site_id.iloc[j], site_name=S.name.iloc[j],
                             shift=sh.name, start_hour=sh.start_hour, end_hour=sh.end_hour))
    assignments = pd.DataFrame(rows, columns=["doctor_id", "site_id", "site_name", "shift", "start_hour", "end_hour"])

    # 3) serve demand hour by hour: capacity = doctors on duty x rate, nearest open site first
    served_rows, site_served = [], {j: 0.0 for j in chosen}
    for h, g in dem.groupby("hour"):
        cap = {j: req.patients_per_doctor_hour * sum(load[(j, sh.name)] for sh in req.shifts if h in sh.hours) for j in chosen}
        for r in g.itertuples():
            i = int(np.flatnonzero(areas.area_id.to_numpy() == r.area_id)[0])
            left = r.expected_cases
            for j in sorted(chosen, key=lambda k: D[i, k]):
                if D[i, j] > req.walk_radius_km or left <= 0:
                    break
                take = min(left, cap[j])
                cap[j] -= take
                left -= take
                site_served[j] += take
            served_rows.append(dict(area_id=r.area_id, hour=h, expected_cases=r.expected_cases,
                                    expected_served=r.expected_cases - left))
    served = pd.DataFrame(served_rows)
    doc_tot = assignments.groupby("site_id").size()
    open_sites = S.iloc[chosen][["site_id", "name", "lat", "lon", "kind"]].copy()
    open_sites["doctors_total"] = open_sites.site_id.map(doc_tot).fillna(0).astype(int)
    open_sites["expected_served"] = [round(site_served[j], 2) for j in chosen]
    total, got = float(served.expected_cases.sum()), float(served.expected_served.sum())
    kpis = dict(expected_cases=total, expected_served=got, pct_served=100 * got / max(total, 1e-9),
                tents_open=len(chosen), doctors_assigned=int(len(assignments)), doctors_available=len(req.doctors))
    return DeploymentPlan(open_sites.reset_index(drop=True), assignments, served, kpis,
                          status="greedy_mock", mip_gap=float("nan"), runtime_s=time.time() - t0,
                          stats=dict(solver="greedy mock (app/mock.py)", sites=len(S), areas=len(areas)))
