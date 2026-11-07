"""Single data-access layer of the app.

get_forecast(scenario)   -> heattents.service.DayForecast from the trained models: citywide predictive distribution
                            (DLNM-NB / LightGBM ensemble, forecast-error Monte Carlo), area shares (beach & event share model),
                            hour profile, and coherent demand scenarios for the optimiser.
candidate_sites(...)     -> NYC Parks properties screened by reachable demand (or Cool It! NYC sites).
solver_backend()         -> heattents.deploy.milp (Gurobi) if importable, else app/mock.py (greedy).
"""
from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heattents.ingest import geography  # noqa: E402

RAW = ROOT / "data" / "raw"


@dataclass(frozen=True)
class Scenario:
    """A day to forecast: a historical date under an information set, or a user-specified weather scenario."""
    label: str = "custom scenario"
    date: str | None = None  # historical date (2024-2026) -> archived 1-day-ahead forecast issued at 18:00 the day before
    heat_index_f: float = 100.0  # custom: max apparent temperature (heat index) in F
    hot_days_before: int = 2  # custom: consecutive days with heat index >= 90 F before today
    weekend: bool = False
    holiday: bool = False
    upstream_gradient_f: float = 0.0  # custom: yesterday's heat index in Philadelphia/Baltimore/DC minus NYC (F)
    calls_yesterday_by18: float = 30.0  # custom: yesterday's citywide heat EMS calls before 18:00
    event: str | None = None  # custom: recurring permitted event on the day (see heattents.models.context_share.EVENTS)


# --- static data ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def areas() -> pd.DataFrame:
    Z = geography.load_modzcta()
    return Z.rename(columns={"modzcta": "area_id"})


@lru_cache(maxsize=1)
def area_polygons() -> pd.DataFrame:
    """One row per polygon part (pydeck PolygonLayer needs simple rings)."""
    rows = []
    for r in areas().itertuples():
        parts = list(r.geometry.geoms) if r.geometry.geom_type == "MultiPolygon" else [r.geometry]
        for p in parts:
            rows.append(dict(area_id=r.area_id, polygon=[list(c) for c in p.exterior.coords]))
    return pd.DataFrame(rows)


# --- forecasts (real models via heattents.service) --------------------------------------------------------------
@lru_cache(maxsize=1)
def service():
    from heattents.service import ForecastService
    return ForecastService()


def f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def get_forecast(sc: Scenario, seed: int = 0):
    """DayForecast: city predictive summary + samples, per-area map values, and Gurobi demand scenarios."""
    svc = service()
    if sc.date is not None:
        return svc.forecast_date(sc.date, seed=seed)
    d = pd.Timestamp("2026-07-15")
    return svc.simulate(seed=seed, atmax=f_to_c(sc.heat_index_f),
                        hot_days_before=int(sc.hot_days_before), season_hot_days_before=max(int(sc.hot_days_before), 8),
                        weekend=int(sc.weekend), holiday=int(sc.holiday), up_sw_gradient=sc.upstream_gradient_f * 5 / 9,
                        calls_by18=float(sc.calls_yesterday_by18), event=sc.event, year=2026, day_of_year=int(d.day_of_year), date=d)


def area_of(area_id: pd.Series) -> pd.Series:
    return area_id.astype(str).str.split("#").str[0]


def _poisson_sf(k: int, lam: np.ndarray) -> np.ndarray:
    """P(N >= k) for N ~ Poisson(lam)."""
    if k <= 0:
        return np.ones_like(lam)
    term, cdf = np.exp(-lam), np.exp(-lam)
    for j in range(1, k):
        term = term * lam / j
        cdf = cdf + term
    return 1 - cdf


def summarise(demand: pd.DataFrame, hours: list[int], k: int = 1, seed: int = 1) -> pd.DataFrame:
    """Per-area expected cases in an hour band, 90% band and P(cases >= k), mixing over forecast scenarios."""
    rng = np.random.default_rng(seed)
    d = demand[demand.hour.isin(hours)].assign(area_id=lambda x: area_of(x.area_id))
    band = d.groupby(["area_id", "scenario"]).expected_cases.sum().unstack("scenario").fillna(0)
    lam = band.to_numpy()
    counts = rng.poisson(lam[:, :, None], size=lam.shape + (50,)).reshape(lam.shape[0], -1)
    return pd.DataFrame({"area_id": band.index, "expected": lam.mean(1), "q05": np.quantile(counts, 0.05, axis=1),
                         "q95": np.quantile(counts, 0.95, axis=1), f"p_ge_{k}": _poisson_sf(k, lam).mean(1)})


# --- candidate sites ------------------------------------------------------------------------------------------
def candidate_sites(source: str, demand: pd.DataFrame, radius_km: float = 1.0, top_n: int = 250) -> pd.DataFrame:
    if source == "Cool It! NYC cooling sites":
        c = pd.read_csv(RAW / "nyc_cool_it_sites_2020.csv")
        c = c[c.status.str.lower().eq("activated")]
        g = c.groupby("propertyname").agg(lat=("lat", "mean"), lon=("lon", "mean"), kind=("featuretype", "first"),
                                          borough=("borough", "first")).reset_index()
        g["site_id"] = ["CS%03d" % i for i in range(len(g))]
        return g.rename(columns={"propertyname": "name"})[["site_id", "name", "lat", "lon", "kind", "borough"]]
    from heattents.deploy.candidates import park_points, screen
    return screen(park_points(), demand, radius_km, top_n)


# --- solver ---------------------------------------------------------------------------------------------------
def solver_backend():
    """(module, label): the Gurobi model when available, else the greedy mock with the same interface."""
    try:
        from heattents.deploy import milp as backend  # written by the optimisation work stream
        return backend, "Gurobi MILP (heattents.deploy.milp)"
    except Exception:
        import mock as backend
        return backend, "MOCK greedy heuristic (app/mock.py) - not optimal"


def _build(cls, **kw):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in names})


def run_solver(n_tents, sites: pd.DataFrame, doctors: pd.DataFrame, shifts: pd.DataFrame, demand: pd.DataFrame,
               **params):
    backend, label = solver_backend()
    shift_objs = [_build(backend.Shift, name=r.name, start_hour=int(r.start_hour), end_hour=int(r.end_hour))
                  for r in shifts.itertuples()]
    doc_objs = [_build(backend.Doctor, doctor_id=str(r.doctor_id),
                       available_shifts=[s.strip() for s in str(r.available_shifts).split(",") if s.strip()],
                       max_shifts=int(r.max_shifts)) for r in doctors.itertuples()]
    site_objs = [_build(backend.Site, site_id=r.site_id, name=r.name, lat=float(r.lat), lon=float(r.lon),
                        max_doctors=int(params.get("max_doctors_per_tent", 4)), kind=r.kind) for r in sites.itertuples()]
    req = _build(backend.DeploymentRequest, n_tents=int(n_tents), sites=site_objs, doctors=doc_objs,
                 shifts=shift_objs, demand=demand, **params)
    return normalise_plan(backend.solve(req), sites, shifts), label


def normalise_plan(plan, sites: pd.DataFrame, shifts: pd.DataFrame):
    """Give the app one plan shape whichever backend produced it (Gurobi or mock)."""
    O = plan.open_sites.copy()
    if "doctors_total" not in O:
        doc_cols = [c for c in O.columns if c.startswith("doctors_")]
        O["doctors_total"] = O[doc_cols].sum(axis=1) if doc_cols else 0
    A = plan.assignments.copy()
    if "site_name" not in A and len(A):
        A = A.merge(sites[["site_id", "name"]].rename(columns={"name": "site_name"}), on="site_id", how="left")
    if "start_hour" not in A and len(A):
        A = A.merge(shifts.rename(columns={"name": "shift"}), on="shift", how="left")
    S = plan.served.rename(columns={"expected_demand": "expected_cases"})
    if "area_id" in S:
        S = S.assign(area_id=area_of(S.area_id))
    k = dict(plan.kpis)
    k.setdefault("pct_served", k.get("served_pct", np.nan))
    k.setdefault("expected_treated", k.get("expected_served", np.nan))
    k.setdefault("treated_pct", k.get("pct_served", np.nan))
    k.setdefault("doctors_assigned", k.get("doctors_used", len(A)))
    plan.open_sites, plan.assignments, plan.served, plan.kpis = O, A, S, k
    return plan
