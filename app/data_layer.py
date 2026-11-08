"""Single data-access layer of the app (real data and trained models only). All counts are PEOPLE needing help
(EMS heat calls x 20, converted once by heattents.service right after the forecast).

get_forecast(scenario)   -> heattents.service.DayForecast for a historical date: citywide predictive distribution
                            (DLNM-NB / LightGBM ensemble, forecast-error Monte Carlo), shares from the beach & event
                            share model, and the 42-neighbourhood x 5-scenario daily demand.
plan_hospitals(fc, n)    -> heattents.deploy.daily plan: where to place n field hospitals (cooling tents) and how
                            to staff them, from that day's forecast (15-minute walk, sites per RULES: parks or anywhere, Gurobi).
city history             -> out-of-sample citywide forecasts vs observed around a date.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heattents.ingest import geography  # noqa: E402

WORKERS_PER_HOSPITAL = 4
from heattents.service import PEOPLE_PER_CALL  # noqa: E402  (forecast output is in people = calls x 20)

# Planning rules. Switch ACTIVE_RULES to "previous" to go back to the earlier behaviour.
RULES = {
    "previous": dict(activation_calls=0.0, min_calls_per_hospital=0.0, max_per_neighbourhood=1, site_mode="parks",
                     radius_km=None, unlimited_capacity=False),
    "threshold": dict(activation_calls=0.0,  # (was 20: deploy only if the city forecast expects >= 20 heat calls)
                      min_calls_per_hospital=0.0,  # (was 3: each open hospital serves >= 3 calls' worth of people)
                      max_per_neighbourhood=3,  # up to 3 hospitals in the same neighbourhood
                      site_mode="anywhere",  # hospitals anywhere on land (400 m grid), not only parks and plazas
                      radius_km=2.0,  # reach: everyone within 2 km straight-line (instead of a 15-minute walk)
                      unlimited_capacity=True),  # a hospital helps everyone in reach; health workers are ignored
}
ACTIVE_RULES = "threshold"  # "previous" = exact number of hospitals, one per neighbourhood, no threshold


@dataclass(frozen=True)
class Scenario:
    """A historical day to forecast, as the forecast was issued at 18:00 the evening before."""
    label: str
    date: str


# --- the 42 UHF neighbourhoods ----------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def neighbourhoods() -> pd.DataFrame:
    """42 UHF neighbourhoods: area_id (= uhf code as str), uhf, name, borough, lat, lon, geometry."""
    N = geography.load_neighbourhoods()[["uhf", "name", "borough", "geometry"]]
    N = N.assign(uhf=N.uhf.astype(str), area_id=N.uhf.astype(str))
    pts = [g.representative_point() for g in N.geometry]
    return N.assign(lat=[p.y for p in pts], lon=[p.x for p in pts])[
        ["area_id", "uhf", "name", "borough", "lat", "lon", "geometry"]]


@lru_cache(maxsize=1)
def neighbourhood_polygons() -> pd.DataFrame:
    """One row per polygon part (pydeck PolygonLayer needs simple rings)."""
    rows = []
    for r in neighbourhoods().itertuples():
        parts = list(r.geometry.geoms) if r.geometry.geom_type == "MultiPolygon" else [r.geometry]
        rows += [dict(area_id=r.area_id, polygon=[list(c) for c in p.exterior.coords]) for p in parts]
    return pd.DataFrame(rows)


# --- forecasts --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def service():
    from heattents.service import ForecastService
    return ForecastService()


def get_forecast(sc: Scenario, seed: int = 0):
    """DayForecast for a historical date (forecast as issued at 18:00 the evening before)."""
    return service().forecast_date(sc.date, seed=seed)


def neighbourhood_forecast(fc) -> pd.DataFrame:
    """Per neighbourhood: area_id, expected calls, 90% range and observed calls (past dates)."""
    nb = fc.neighbourhoods.assign(area_id=fc.neighbourhoods.uhf.astype(str))
    cols = ["area_id", "expected", "q05", "q95"] + (["observed"] if "observed" in nb else [])
    return nb[cols]


# --- field hospitals (daily MILP) -------------------------------------------------------------------------------
def deployment_needed(fc, rules: str = ACTIVE_RULES) -> bool:
    """Activation threshold: deploy only if the city forecast expects at least RULES[rules]['activation_calls']."""
    return float(fc.city["mean"]) / fc.people_per_call >= RULES[rules]["activation_calls"]  # threshold is in calls


def plan_hospitals(fc, n_hospitals: int, workers_per_hospital: int = WORKERS_PER_HOSPITAL, rules: str = ACTIVE_RULES):
    """deploy.daily plan for the day: up to n field hospitals, workers_per_hospital x n health workers in total,
    under the planning rules RULES[rules]. None when the activation threshold is not reached."""
    from heattents.deploy.daily import solve, walking_request
    r = RULES[rules]
    if not deployment_needed(fc, rules):
        return None
    return solve(walking_request(fc, int(n_hospitals), int(n_hospitals) * int(workers_per_hospital),
                                 max_per_neighbourhood=r["max_per_neighbourhood"],
                                 min_calls_per_hospital=r["min_calls_per_hospital"], site_mode=r["site_mode"],
                                 radius_km=r["radius_km"], unlimited_capacity=r["unlimited_capacity"]))


# --- history around a date ----------------------------------------------------------------------------------------
MODEL_LABELS = {"glm": "GLM", "gbm": "GBM", "ens": "Ensemble"}


@lru_cache(maxsize=1)
def city_predictions() -> pd.DataFrame:
    """Out-of-sample city forecasts 2024-26: season, model (glm/gbm/ens), date, y (observed), mean, q05, q50, q95."""
    return pd.read_csv(ROOT / "outputs" / "models" / "city" / "oos_predictions.csv", parse_dates=["date"])


def _window(date: str, days_before: int, days_after: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(date)
    return d - pd.Timedelta(days=days_before), d + pd.Timedelta(days=days_after)


def city_history(date: str, days_before: int = 30, days_after: int = 7) -> pd.DataFrame:
    """Ensemble forecast vs observed citywide calls around a date: date, observed, forecast, q05, q95."""
    lo, hi = _window(date, days_before, days_after)
    P = city_predictions()
    P = P[(P.model == "ens") & P.date.between(lo, hi)]
    P = P.rename(columns={"y": "observed", "mean": "forecast"})[["date", "observed", "forecast", "q05", "q95"]]
    return P.assign(**{c: P[c] * PEOPLE_PER_CALL for c in ["observed", "forecast", "q05", "q95"]})  # in people


def city_models_history(date: str, days_before: int = 30, days_after: int = 7) -> pd.DataFrame:
    """Citywide calls around a date, long format: date, series (GLM/GBM/Ensemble/Actual), calls."""
    lo, hi = _window(date, days_before, days_after)
    P = city_predictions()
    P = P[P.date.between(lo, hi)]
    models = P.assign(series=P.model.map(MODEL_LABELS), calls=P["mean"])[["date", "series", "calls"]]
    actual = P[P.model == "ens"].assign(series="Actual", calls=lambda d: d.y)[["date", "series", "calls"]]
    out = pd.concat([models, actual], ignore_index=True)
    return out.assign(calls=out.calls * PEOPLE_PER_CALL)  # in people (column name kept for the chart)


# --- slide tabs (precomputed by app/precompute.py) ------------------------------------------------------------------
APP_OUT = ROOT / "outputs" / "app"
COVERAGE_OUT = ROOT / "outputs" / "models" / "deploy_compare" / "coverage"


@lru_cache(maxsize=1)
def city_accuracy() -> dict:
    """Out-of-sample accuracy of the citywide ensemble, 2024-26, in people."""
    E = city_predictions()
    E = E[E.model == "ens"]
    err = (E.y - E["mean"]).abs() * PEOPLE_PER_CALL
    top = E.nlargest(8, "y")
    return dict(days=len(E), median_error=float(err.median()), mean_error=float(err.mean()),
                correlation=float(np.corrcoef(E.y, E["mean"])[0, 1]),
                in_range_pct=float(100 * ((E.y >= E.q05) & (E.y <= E.q95)).mean()),
                busiest_error={MODEL_LABELS[m]: float(PEOPLE_PER_CALL * (g.set_index("date").loc[top.date, "mean"]
                                                                         - top.set_index("date").y).abs().mean())
                               for m, g in city_predictions().groupby("model")})


@lru_cache(maxsize=1)
def input_correlations() -> pd.DataFrame:
    """label, correlation of each main input with daily citywide calls (May-Sep 2005-2026)."""
    return pd.read_csv(APP_OUT / "input_correlations.csv")


def oos_2026(uhf: str | None = None) -> pd.DataFrame:
    """2026 forecast (models trained on 2005-2025) vs observed, in people: date, observed, forecast[, q05, q95]."""
    if uhf is None:
        P = city_predictions()
        P = P[(P.model == "ens") & (P.season == 2026)].rename(columns={"y": "observed", "mean": "forecast"})
        P = P[["date", "observed", "forecast", "q05", "q95"]]
        return P.assign(**{c: P[c] * PEOPLE_PER_CALL for c in ["observed", "forecast", "q05", "q95"]})
    N = _oos_2026_neighbourhoods()
    return N[N.uhf == str(uhf)].rename(columns={"expected": "forecast"})[["date", "observed", "forecast"]]


@lru_cache(maxsize=1)
def _oos_2026_neighbourhoods() -> pd.DataFrame:
    return pd.read_csv(APP_OUT / "oos_2026_neighbourhoods.csv", parse_dates=["date"], dtype={"uhf": str})


@lru_cache(maxsize=1)
def coverage_results() -> pd.DataFrame:
    """date, plan, needed, helped, helped_pct (people) under the active rules (deploy.compare.run_coverage)."""
    return pd.read_csv(COVERAGE_OUT / "daily_results.csv")


@lru_cache(maxsize=1)
def coverage_sites() -> pd.DataFrame:
    """date, plan, name, lat, lon of every compared plan."""
    return pd.read_csv(COVERAGE_OUT / "sites.csv")


@lru_cache(maxsize=8)
def observed_neighbourhoods(date: str) -> pd.DataFrame:
    """area_id, observed people needing help per neighbourhood on a date."""
    P = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet", columns=["modzcta", "date", "y"],
                        filters=[("date", "==", pd.Timestamp(date))])
    nb = P.modzcta.map(geography.modzcta_to_neighbourhood()).astype(str)
    return (P.y * PEOPLE_PER_CALL).groupby(nb).sum().rename("observed").rename_axis("area_id").reset_index()
