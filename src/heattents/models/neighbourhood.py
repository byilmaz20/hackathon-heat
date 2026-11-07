r"""Neighbourhood model: where in New York tomorrow's heat EMS calls will happen.

UNIT   the 42 United Hospital Fund neighbourhoods (UHF42) used by the NYC Health Department; each is a group of whole
       ZIP codes. At ZIP level 77% of areas have zero calls even on busy days (>= 30 citywide calls); at UHF42, 40%.
TASK   given the citywide total N_d (from the city model), the share pi_{i,d} of calls in neighbourhood i:
           y_{i,d} | N_d ~ Multinomial(N_d, pi_d),     expected calls_i = N_d * pi_{i,d}
INFORMATION (at 18:00 on d-1, same rules as the city model)
       weather of d       evening-before forecast (training: observed weather, "perfect prog")
       calls on d-1       before 18:00 only, per neighbourhood
       population         ACS 5-year estimate of year d-2 (the latest published before the summer of d)
INPUTS of the LightGBM share model
       neighbourhood      population, population density, share aged 65+ (ACS, year d-2); surface temperature,
       (static)           vegetative cover, tree canopy, park share, household AC, poverty, older adults living
                          alone, Black share, Heat Vulnerability Index (one snapshot each; they change slowly)
       day                apparent max, relative humidity, weekend, holiday, 4 July window, school break, day of year
       recent calls       log(1 + calls in the neighbourhood on d-1 before 18:00)
       identity           neighbourhood id (categorical)
METHODS compared on 2024, 2025, 2026 (each trained on earlier summers)
       historical share   each neighbourhood's share of all earlier calls (2005 onwards)
       recent share       exponentially year-weighted share (half-life 4 years), shrunk towards the population share
       LightGBM           Poisson with offset log N_d (shares; trained on 2013 onwards because of the ACS lag)
       oracle             the test season's own realised shares - a ceiling, not a forecast
METRICS  log-likelihood per call (multinomial, given the true total); share of calls in the predicted top 4
         neighbourhoods (~10%); both on all days and on busy days (>= 30 citywide calls).
HEAT SHIFT TEST  do hot or weekend days move calls to different neighbourhoods? Likelihood-ratio statistic of
         "same shares on both kinds of day", with a day-level permutation p-value.
Run: python -m heattents.models.neighbourhood
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from ..ingest.demographics import acs
from ..ingest.geography import load_modzcta
from .city import DECISION_HOUR, Context

TEST_SEASONS = (2024, 2025, 2026)
GBM_FIRST_YEAR = 2013  # first summer with an ACS estimate two years older (2011)
ACS_LAG = 2
HALF_LIFE_YEARS = 4.0
SHRINK_CALLS = 30.0  # pseudo-calls pulling small neighbourhoods towards their population share
BUSY = 30
STATIC = ["surface_temp_f", "vegetative_cover_pct", "tree_canopy_pct", "park_share", "households_with_ac_pct",
          "poverty_pct", "older_living_alone_pct", "black_pct", "hvi"]
YEARLY = ["pop", "pop_density", "pct_65plus"]
DAILY = ["atmax", "rh", "weekend", "holiday", "july4_window", "school_break", "day_of_year", "lag_calls"]
PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, verbose=-1, seed=7, num_threads=4)
N_ROUNDS = 400


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------
def neighbourhood_of(paths: Paths = PATHS) -> pd.Series:
    """MODZCTA -> UHF42 code (NYC Health Department crosswalk; every MODZCTA is itself a ZCTA code)."""
    x = pd.read_csv(paths.raw / "ehdp_zcta_to_uhf.csv", dtype=str)
    x = x[x.uhfcode != "9999"]  # placeholder for "no neighbourhood" (ZIP 99999)
    return pd.Series(x.uhfcode.to_numpy(), index=x.zcta.to_numpy())


def static_table(paths: Paths = PATHS) -> pd.DataFrame:
    """Neighbourhood snapshot features: area-weighted environment, population-weighted vulnerability."""
    S = pd.read_csv(paths.processed / "modzcta_static.csv", dtype={"modzcta": str})
    S["uhf"] = S.modzcta.map(neighbourhood_of(paths))
    env = ["surface_temp_f", "vegetative_cover_pct", "tree_canopy_pct", "households_with_ac_pct"]
    soc = ["poverty_pct", "older_living_alone_pct", "black_pct", "hvi"]
    g = S.groupby("uhf")
    out = pd.DataFrame({"area_km2": g.area_km2.sum(), "park_share": g.apply(lambda d: (d.park_area_km2.sum() / d.area_km2.sum()), include_groups=False)})
    for c in env:
        out[c] = g.apply(lambda d, c=c: np.average(d[c], weights=d.area_km2), include_groups=False)
    for c in soc:
        out[c] = g.apply(lambda d, c=c: np.average(d[c], weights=d.pop_est), include_groups=False)
    return out.reset_index()


def yearly_population(paths: Paths = PATHS) -> pd.DataFrame:
    """Population and 65+ share per neighbourhood and summer, from the ACS estimate of year - ACS_LAG."""
    A = acs(paths)
    if A is None:
        raise RuntimeError("ACS data missing: run heattents ingest with CENSUS_API_KEY set")
    nb = neighbourhood_of(paths)
    A = A[A.zcta.isin(nb.index)].assign(uhf=lambda d: d.zcta.map(nb))
    g = A.groupby(["uhf", "year"])[["pop_total", "pop_65plus"]].sum().reset_index()
    g["summer"] = g.year + ACS_LAG
    return g.rename(columns={"pop_total": "pop"}).assign(pct_65plus=lambda d: 100 * d.pop_65plus / d["pop"])[
        ["uhf", "summer", "pop", "pct_65plus"]]


def build_panel(paths: Paths = PATHS) -> pd.DataFrame:
    """Neighbourhood x day table (May-Sep 2005-2026) with calls, calls on d-1 before 18:00, observed and
    forecast weather of day d, calendar, static and yearly features."""
    from ..features.panel import load_ems
    nb = neighbourhood_of(paths)
    e = load_ems(paths).dropna(subset=["modzcta"])
    e["uhf"] = e.modzcta.map(nb)
    ctx = Context.load(paths)
    C = ctx.city[["date", "year", "weekend", "holiday", "july4_window", "school_break", "day_of_year", "city_y"]]
    uhfs = sorted(nb.unique())
    P = pd.DataFrame({"uhf": np.repeat(uhfs, len(C)), "date": np.tile(C.date.to_numpy(), len(uhfs))})
    P = P.merge(e.groupby(["uhf", "date"]).size().rename("y"), on=["uhf", "date"], how="left").fillna({"y": 0})
    by18 = e[e.incident_datetime.dt.hour < DECISION_HOUR].groupby(["uhf", "date"]).size().rename("lag_n").reset_index()
    by18["date"] = by18.date + pd.Timedelta(days=1)
    P = P.merge(by18, on=["uhf", "date"], how="left").fillna({"lag_n": 0})
    P["lag_calls"] = np.log1p(P.pop("lag_n"))
    P = P.merge(C, on="date", how="left")
    obs = ctx.obs[["atmax", "rh"]].rename(columns=lambda c: c + "_obs")
    fc = ctx.fc[["atmax", "rh"]].rename(columns=lambda c: c + "_fc")
    P = P.merge(obs, left_on="date", right_index=True, how="left").merge(fc, left_on="date", right_index=True, how="left")
    P = P.merge(static_table(paths), on="uhf", how="left")
    P = P.merge(yearly_population(paths).rename(columns={"summer": "year"}), on=["uhf", "year"], how="left")
    P["pop_density"] = P["pop"] / P.area_km2
    P["mapped_total"] = P.groupby("date").y.transform("sum")
    P["uhf_cat"] = P.uhf.astype("category")
    return P


# ---------------------------------------------------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------------------------------------------------
def historical_share(train: pd.DataFrame) -> pd.Series:
    s = train.groupby("uhf").y.sum() + 0.5
    return s / s.sum()


def recent_share(train: pd.DataFrame, test_year: int) -> pd.Series:
    """Year-weighted share (weight 2^-(age/half-life)), shrunk towards the population share."""
    w = 0.5 ** ((test_year - train.year) / HALF_LIFE_YEARS)
    calls = (train.y * w).groupby(train.uhf).sum()
    pop = train[train.year == train.year.max()].groupby("uhf")["pop"].first()
    pop = (pop / pop.sum()).reindex(calls.index).fillna(1 / len(calls))
    s = calls + SHRINK_CALLS * pop
    return s / s.sum()


def features(P: pd.DataFrame, weather: str) -> pd.DataFrame:
    X = P[STATIC + YEARLY + [c for c in DAILY if c not in ("atmax", "rh")] + ["uhf_cat"]].copy()
    X["atmax"], X["rh"] = P[f"atmax_{weather}"].to_numpy(), P[f"rh_{weather}"].to_numpy()
    return X


def fit_gbm(train: pd.DataFrame) -> lgb.Booster:
    t = train[(train.year >= GBM_FIRST_YEAR) & (train.mapped_total > 0)]
    ds = lgb.Dataset(features(t, "obs"), t.y, init_score=np.log(t.mapped_total), categorical_feature=["uhf_cat"])
    return lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)


def gbm_shares(model: lgb.Booster, test: pd.DataFrame) -> np.ndarray:
    raw = np.exp(model.predict(features(test, "fc"), raw_score=True))
    return raw / pd.Series(raw, index=test.index).groupby(test.date.to_numpy()).transform("sum").to_numpy()


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------------------
def score(test: pd.DataFrame, share: np.ndarray, top_k: int = 4) -> dict:
    t = test.assign(share=share)
    t = t[t.mapped_total > 0]
    out = {}
    for name, sub in (("all", t), ("busy", t[t.mapped_total >= BUSY])):
        ll = (sub.y * np.log(np.maximum(sub.share, 1e-12))).sum() / sub.y.sum()
        hit = sum(g.nlargest(top_k, "share").y.sum() for _, g in sub.groupby("date")) / sub.y.sum()
        out[f"loglik_per_call_{name}"], out[f"top{top_k}_capture_{name}"] = ll, hit
    out["calls"], out["busy_days"] = int(t.y.sum()), int(t[t.mapped_total >= BUSY].date.nunique())
    return out


def evaluate(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in TEST_SEASONS:
        tr, te = P[P.year < s], P[(P.year == s)].copy()
        per_day = lambda shares: te.uhf.map(shares).to_numpy()
        methods = {
            "historical share": per_day(historical_share(tr)),
            "recent share": per_day(recent_share(tr, s)),
            "LightGBM": gbm_shares(fit_gbm(tr), te),
            "oracle (not a forecast)": per_day(historical_share(te)),
        }
        for name, sh in methods.items():
            rows.append(dict(season=s, method=name, **score(te, sh)))
        print(f"[neighbourhood] season {s} done", flush=True)
    return pd.DataFrame(rows)


def heat_shift_test(P: pd.DataFrame, n_perm: int = 500, seed: int = 7) -> pd.DataFrame:
    """Do neighbourhood shares differ between two kinds of day? G-statistic (multinomial likelihood ratio) of
    'one common set of shares' vs 'separate shares', p-value from permuting the day labels."""
    rng = np.random.default_rng(seed)
    T = P[P.mapped_total > 0].pivot_table(index="date", columns="uhf", values="y", aggfunc="sum")
    day = P.drop_duplicates("date").set_index("date").loc[T.index]
    splits = {"hot day (forecast-free: observed apparent max >= 32 C)": (day.atmax_obs >= 32).to_numpy(),
              "busy day (>= 30 citywide calls)": (day.mapped_total >= BUSY).to_numpy(),
              "weekend or holiday": ((day.weekend == 1) | (day.holiday == 1)).to_numpy()}
    Y = T.to_numpy()

    def G(mask):
        a, b = Y[mask].sum(0), Y[~mask].sum(0)
        tot = a + b
        ll = lambda c: (c[c > 0] * np.log(c[c > 0] / c.sum())).sum()
        return 2 * (ll(a) + ll(b) - ll(tot))

    rows = []
    for name, mask in splits.items():
        g0 = G(mask)
        perm = np.array([G(rng.permutation(mask)) for _ in range(n_perm)])
        a, b = Y[mask].sum(0), Y[~mask].sum(0)
        diff = pd.Series(100 * (a / a.sum() - b / b.sum()), index=T.columns)
        top = diff.abs().sort_values(ascending=False).index[:3]
        rows.append(dict(split=name, days_in_group=int(mask.sum()), calls_in_group=int(a.sum()), G=round(g0, 1),
                         p_value=round((np.sum(perm >= g0) + 1) / (n_perm + 1), 4),
                         largest_share_shifts_pct_points="; ".join(f"UHF {u}: {diff[u]:+.2f}" for u in top)))
    return pd.DataFrame(rows)


def run(paths: Paths = PATHS) -> pd.DataFrame:
    out = paths.out("models/neighbourhood")
    P = build_panel(paths)
    P.drop(columns=["uhf_cat"]).to_parquet(paths.processed / "neighbourhood_panel.parquet", index=False)
    shift = heat_shift_test(P)
    shift.to_csv(out / "heat_shift_test.csv", index=False)
    print(shift.to_string(index=False))
    E = evaluate(P)
    E.round(4).to_csv(out / "scores_by_season.csv", index=False)
    summary = E.groupby("method", sort=False).mean(numeric_only=True).drop(columns="season")
    summary.round(4).to_csv(out / "scores_summary.csv")
    print(summary.round(4).to_string())
    return E


if __name__ == "__main__":
    from heattents.models.neighbourhood import run as _run
    _run()
