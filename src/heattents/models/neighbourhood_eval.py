r"""How many heat EMS calls will each of NYC's 42 neighbourhoods get tomorrow, and by how much are we off?

All forecasts are made at 18:00 the evening before; every average, share and coefficient for summer s uses only
summers before s. The citywide number is the city model's out-of-sample forecast (ensemble), never the realised total.

STEP 1 - simple methods
    zero                 0 calls every day
    past average         the neighbourhood's mean daily calls in earlier summers
    historical share     E[N_d] * m * pi_i       E[N_d] city forecast, m = share of calls with a neighbourhood
                                                 (earlier summers), pi_i = neighbourhood share of earlier calls
STEP 2 - neighbourhood weather (difference from the 42-neighbourhood mean on the same day, deg C)
    night_anom   local night minimum minus the city mean of night minimums
    day_anom     local afternoon maximum minus the city mean of afternoon maximums
    training     Daymet 1 km daily tmin / tmax (2005-2025) at the neighbourhood centres
    testing      HRRR 3 km forecast issued the day before (daily min / max at the centres), rescaled to Daymet's scale
                 by a regression on the 2024-25 overlap (weather only, no call data)
STEP 3 - shared response to local heat (no neighbourhood-specific coefficients)
    log pi_{i,d} = log pi_i + beta_n * night_anom_{i,d} [+ beta_d * day_anom_{i,d}],  renormalised to sum to 1 per day
    beta fitted by Poisson regression with offset log(calls with a neighbourhood that day) + log pi_i on earlier summers.
DECISION METRICS  MAE over all neighbourhood-days (zero days included) and MAE per neighbourhood on days with >= 30
    calls citywide, for 2024, 2025 and 2026 separately.
Run: python -m heattents.models.neighbourhood_eval
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .glm import fit_glm
from .neighbourhood import historical_share

TEST_SEASONS = (2024, 2025, 2026)
BUSY = 30


def load(paths: Paths = PATHS) -> pd.DataFrame:
    """Neighbourhood x day calls, city forecast (out of sample), local night/day anomalies (Daymet and HRRR)."""
    from ..ingest import daymet, weather_local
    P = pd.read_parquet(paths.processed / "neighbourhood_panel.parquet")[["uhf", "date", "year", "y", "city_y", "mapped_total"]]
    P["uhf"] = P.uhf.astype(str)
    oos = pd.read_csv(paths.out("models/city") / "oos_predictions.csv", parse_dates=["date"])
    P = P.merge(oos[oos.model == "ens"][["date", "mean"]].rename(columns={"mean": "city_forecast"}), on="date", how="left")
    for src, d in (("daymet", daymet.fetch(paths)), ("hrrr", weather_local.forecasts(paths))):
        d = d[["uhf", "date", "tmin", "tmax"]].copy()
        d["uhf"] = d.uhf.astype(str)
        for v, name in (("tmin", "night"), ("tmax", "day")):
            d[f"{name}_anom_{src}"] = d[v] - d.groupby("date")[v].transform("mean")
        P = P.merge(d.drop(columns=["tmin", "tmax"]), on=["uhf", "date"], how="left")
    return P


def hrrr_scale(P: pd.DataFrame) -> dict:
    """daymet_anom ~ b * hrrr_anom on the 2024-25 overlap, per variable (weather only)."""
    out = {}
    for name in ("night", "day"):
        m = P[[f"{name}_anom_daymet", f"{name}_anom_hrrr"]].dropna()
        out[name] = float(np.polyfit(m[f"{name}_anom_hrrr"], m[f"{name}_anom_daymet"], 1)[0])
        out[f"{name}_corr"] = float(m.corr().iloc[0, 1])
    return out


def fit_beta(train: pd.DataFrame, share: pd.Series, cols: list[str]) -> np.ndarray:
    T = train[(train.mapped_total > 0)].dropna(subset=cols)
    off = np.log(T.mapped_total.to_numpy()) + np.log(T.uhf.map(share).to_numpy())
    return fit_glm(T[cols].to_numpy(float), T.y.to_numpy(float), 0.0, np.full(len(cols), 0.1), offset=off,
                   b0=np.zeros(len(cols)))


def local_shares(test: pd.DataFrame, share: pd.Series, beta: np.ndarray, cols: list[str]) -> np.ndarray:
    raw = test.uhf.map(share).to_numpy() * np.exp(test[cols].fillna(0).to_numpy(float) @ beta)
    return raw / pd.Series(raw, index=test.index).groupby(test.date.to_numpy()).transform("sum").to_numpy()


def evaluate(paths: Paths = PATHS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P = load(paths)
    sc = hrrr_scale(P)
    for name in ("night", "day"):
        P[f"{name}_anom_fc"] = sc[name] * P[f"{name}_anom_hrrr"]
    forecasts, betas = [], []
    for s in TEST_SEASONS:
        tr, te = P[P.year < s], P[P.year == s].copy()
        share = historical_share(tr)
        m = tr.y.sum() / tr.drop_duplicates("date").city_y.sum()
        te["zero"] = 0.0
        te["past average"] = te.uhf.map(tr.groupby("uhf").y.mean())
        te["historical share"] = te.city_forecast * m * te.uhf.map(share)
        for label, cols in (("+ night difference", ["night_anom"]), ("+ night and day difference", ["night_anom", "day_anom"])):
            beta = fit_beta(tr.rename(columns={f"{c}_daymet": c for c in cols}), share, cols)
            te[label] = te.city_forecast * m * local_shares(te.rename(columns={f"{c}_fc": c for c in cols}), share, beta, cols)
            betas.append(dict(season=s, model=label, **{f"share_change_per_C_{c}": float(np.exp(b) - 1) for c, b in zip(cols, beta)}))
        te["mapped_fraction_used"] = m
        forecasts.append(te)
    F = pd.concat(forecasts, ignore_index=True)
    methods = ["zero", "past average", "historical share", "+ night difference", "+ night and day difference"]
    rows = []
    for s, g in F.groupby("year"):
        busy = g[g.city_y >= BUSY]
        for meth in methods:
            rows.append(dict(season=s, method=meth, mae_all=(g[meth] - g.y).abs().mean(),
                             mae_busy=(busy[meth] - busy.y).abs().mean(), busy_days=busy.date.nunique(),
                             forecast_total=g[meth].sum(), actual_total=g.y.sum()))
    S = pd.DataFrame(rows)
    return S, pd.DataFrame(betas), F.assign(**{"scale_" + k: v for k, v in sc.items()})


def data_check(paths: Paths = PATHS, days=("2025-06-24", "2026-07-02", "2026-07-16")) -> pd.DataFrame:
    """Night (daily min) and afternoon (daily max) HRRR forecasts for the 42 neighbourhoods on a few hot days."""
    from ..ingest import weather_local
    from .combined import neighbourhood_names
    fc = weather_local.forecasts(paths)
    fc["uhf"] = fc.uhf.astype(str)
    names = neighbourhood_names(paths)
    rows = []
    for d in days:
        x = fc[fc.date == pd.Timestamp(d)]
        for v in ("tmin", "tmax"):
            rows.append(dict(day=d, variable={"tmin": "night min", "tmax": "afternoon max"}[v], distinct_values=x[v].round(2).nunique(),
                             min_C=x[v].min(), max_C=x[v].max(), range_C=x[v].max() - x[v].min(),
                             coolest=names[x.loc[x[v].idxmin(), "uhf"]], hottest=names[x.loc[x[v].idxmax(), "uhf"]]))
    return pd.DataFrame(rows)


def run(paths: Paths = PATHS):
    out = paths.out("models/neighbourhood/evaluation")
    C = data_check(paths)
    C.round(2).to_csv(out / "hrrr_data_check.csv", index=False)
    print(C.round(1).to_string(index=False))
    S, B, F = evaluate(paths)
    S.round(4).to_csv(out / "mae_by_season.csv", index=False)
    B.round(4).to_csv(out / "local_heat_coefficients.csv", index=False)
    F.to_csv(out / "forecasts.csv", index=False)
    print({k: round(v, 3) for k, v in F.filter(like="scale_").iloc[0].items()})
    print(S.pivot_table(index="method", columns="season", values="mae_all", sort=False).round(4).to_string())
    print(S.pivot_table(index="method", columns="season", values="mae_busy", sort=False).round(4).to_string())
    print(B.round(4).to_string(index=False))
    return S


if __name__ == "__main__":
    from heattents.models.neighbourhood_eval import run as _run
    _run()
