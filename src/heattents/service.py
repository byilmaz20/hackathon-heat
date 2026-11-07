"""One entry point from "a day" (historical date or user scenario) to forecasts, maps and Gurobi demand scenarios.

    day  ->  city model (predictive distribution of the citywide total, chosen information set)
         ->  area model (shares pi_it)  ->  hour profile  ->  coherent demand scenarios D^s_{a,h}

Used by the Streamlit app and by the decision-focused evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .config import PATHS, Paths
from .models.area import DAILY, AreaModel, prepare
from .models.city import Context, decision_features, load_city_model
from .models.scenarios import build_demand, hour_profile, mapped_fraction, stratified_totals

REFERENCE_DATE = "2025-07-15"  # template row set for user scenarios (static columns only are kept)


@dataclass
class DayForecast:
    label: str
    city: pd.Series  # mean, q05..q95, p_ge50, p_ge100
    samples: np.ndarray  # predictive samples of the citywide total
    areas: pd.DataFrame  # modzcta, share, expected (mean total x share x mapped fraction), p_ge1, borough
    demand: pd.DataFrame  # Gurobi input: area_id, lat, lon, hour, scenario, expected_cases
    observed_total: float | None = None
    observed_areas: pd.Series | None = None


class ForecastService:
    def __init__(self, paths: Paths = PATHS, n_draws: int = 30):
        self.paths, self.n_draws = paths, n_draws

    @cached_property
    def ctx(self) -> Context:
        return Context.load(self.paths)

    @cached_property
    def area_model(self) -> AreaModel:
        return AreaModel.load(self.paths.out("models/area") / "area_model.txt")

    @cached_property
    def profile(self) -> pd.DataFrame:
        return hour_profile(self.paths)

    @cached_property
    def mapped(self) -> float:
        return mapped_fraction(self.paths)

    @cached_property
    def panel_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.ctx.city.date)

    @cached_property
    def city_model(self):
        return load_city_model(self.paths)

    def area_rows(self, date) -> pd.DataFrame:
        P = pd.read_parquet(self.paths.processed / "panel.parquet", filters=[("date", "==", pd.Timestamp(date))])
        return prepare(P)

    # ------------------------------------------------------------------------------------------------------------
    def _finish(self, label, cm, f, rows, n_scenarios, n_samples, rng, observed=None) -> DayForecast:
        pred = cm.predict(f).iloc[0]
        samples = cm.sample(f, n_samples, rng)[0]
        shares = self.area_model.shares(rows)
        exp_total = float(pred["mean"])
        areas = pd.DataFrame({"modzcta": rows.modzcta.to_numpy(), "borough": rows.borough.to_numpy(), "share": shares,
                              "expected": exp_total * shares * self.mapped})
        # P(at least one call in the area) under the predictive mixture of totals (Poisson thinning)
        areas["p_ge1"] = 1 - np.exp(-np.outer(samples, shares * self.mapped)).mean(0)
        totals = stratified_totals(samples, n_scenarios)
        demand = build_demand(rows, totals, self.area_model, self.profile, self.mapped)
        obs_areas = rows.set_index("modzcta").y if observed is not None and "y" in rows else None
        return DayForecast(label, pred, samples, areas, demand, observed, obs_areas)

    def forecast_date(self, date, n_scenarios: int = 5, n_samples: int = 4000, seed: int = 0) -> DayForecast:
        """Forecast a historical day as issued at 18:00 the evening before (archived 1-day-ahead weather forecast)."""
        rng = np.random.default_rng(seed)
        date = pd.Timestamp(date)
        row = self.ctx.city[self.ctx.city.date == date]
        if row.empty:
            raise ValueError(f"{date.date()} is not in the modelled season (May-Sep 2005-2026)")
        f, _ = decision_features(self.ctx, row, rng, self.n_draws)
        return self._finish(f"{date.date()} (forecast issued 18:00 the day before)", self.city_model, f,
                            self.area_rows(date), n_scenarios, n_samples, rng, observed=float(row.city_y.iloc[0]))

    def simulate(self, n_scenarios: int = 5, n_samples: int = 4000, seed: int = 0, lead_error_sd: float = 1.5,
                 **scenario) -> DayForecast:
        """User scenario, e.g. simulate(atmax=41, hot_days_before=3, weekend=1, up_sw_gradient=2, calls_by18=90)."""
        rng = np.random.default_rng(seed)
        cm = self.city_model
        f = cm.scenario(rng=rng, n_draws=self.n_draws, lead_error_sd=lead_error_sd, **scenario)
        rows = self.area_rows(REFERENCE_DATE)
        first = f.iloc[0]
        override = {"atmax": first.atmax, "tmin": first.tmin, "rh": first.rh, "ehf": first.ehf, "streak": first.streak,
                    "season_hot_days": first.season_hot_days, "weekend": first.weekend, "holiday": first.holiday,
                    "july4_window": first.july4_window, "school_break": first.school_break, "covid": first.covid,
                    "year": first.year, "day_of_year": first.day_of_year}
        for c, v in override.items():
            if c in DAILY:
                rows[c] = v
        # yesterday's borough calls: scale yesterday's citywide count by the borough's historical share
        y_l1 = (scenario.get("calls_by18") or 0.0) / 0.81  # full-day estimate: 81% of calls arrive before 18:00
        bshare = rows.groupby("borough").daytime_presence.transform("sum") / rows.daytime_presence.sum()
        rows["borough_y_l1"] = y_l1 * bshare
        return self._finish("scenario", cm, f, rows, n_scenarios, n_samples, rng)
