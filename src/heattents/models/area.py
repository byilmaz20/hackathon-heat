r"""Area model: where in the city today's heat calls happen (top-down reconciliation with forecast proportions).

Given the citywide total N_t, calls are allocated to the 177 areas as  y_{it} | N_t ~ Multinomial(N_t, pi_t),
    pi_{it} = exp(g(x_i, w_t)) / sum_k exp(g(x_k, w_t))
which is fitted as a Poisson model with offset log N_t (the Poisson-multinomial equivalence):
    E[y_{it}] = N_t * exp(g(x_i, w_t)),    g = LightGBM (Poisson loss)
x_i  static area features (daytime presence, jobs, subway, density, surface temperature, vegetation, parks,
     vulnerability) + the area id as a categorical (persistent hotspots, rank stability 0.94 in the EDA)
w_t  the day's heat and calendar (apparent max, EHF, streak, weekend, holiday, school break, season, year)
so the *shape* of the map can change with the weather (e.g. beaches on hot weekends).
At forecast time the shares pi_t are multiplied by the city model's predictive total: coherent by construction
(areas always sum to the city forecast) - Athanasopoulos, Ahmed & Hyndman (2009), "top-down with forecast
proportions".

Evaluation (rolling origin by season): multinomial log-likelihood per call given the true total, and the share
of calls landing in the predicted top-10% / top-20 areas, against historical-share, population-share and
daytime-presence-share baselines.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import PATHS, Paths

STATIC = ["log_daytime_presence", "pop_density", "jobs_density", "subway_entries", "n_subway_stations", "surface_temp_f",
          "vegetative_cover_pct", "tree_canopy_pct", "park_share", "n_parks", "poverty_pct", "older_living_alone_pct",
          "black_pct", "households_with_ac_pct", "hvi", "area_km2", "lat", "lon"]
DAILY = ["atmax", "tmin", "rh", "ehf", "streak", "season_hot_days", "weekend", "holiday", "july4_window",
         "school_break", "day_of_year", "year", "covid", "borough_y_l1"]
OPTIONAL = ["local_atmax_anomaly"]
PARAMS = dict(objective="poisson", learning_rate=0.04, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=50, verbose=-1, seed=7, num_threads=0)
N_ROUNDS = 500
TEST_SEASONS = [2018, 2019, 2022, 2023, 2024, 2025, 2026]


def feature_list(P: pd.DataFrame) -> list[str]:
    return STATIC + DAILY + [c for c in OPTIONAL if c in P] + ["area_cat"]


def prepare(P: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    P["area_cat"] = P.modzcta.astype("category")
    P["mapped_total"] = P.groupby("date").y.transform("sum")
    return P


class AreaModel:
    def __init__(self, booster: lgb.Booster, features: list[str], categories):
        self.booster, self.features, self.categories = booster, features, categories

    def shares(self, X: pd.DataFrame) -> np.ndarray:
        """pi_{it} for rows of X (must contain every area for each date); normalised within date."""
        X = X.copy()
        X["area_cat"] = pd.Categorical(X.modzcta, categories=self.categories)
        raw = np.exp(self.booster.predict(X[self.features], raw_score=True))
        return raw / pd.Series(raw, index=X.index).groupby(X.date.to_numpy()).transform("sum").to_numpy()

    def save(self, path):
        self.booster.save_model(str(path))
        json.dump({"features": self.features, "categories": list(self.categories)}, open(str(path) + ".json", "w"))

    @classmethod
    def load(cls, path):
        meta = json.load(open(str(path) + ".json"))
        return cls(lgb.Booster(model_file=str(path)), meta["features"], pd.Index(meta["categories"]))


def fit(P: pd.DataFrame, rounds: int = N_ROUNDS) -> AreaModel:
    P = P[P.mapped_total > 0]
    feats = feature_list(P)
    ds = lgb.Dataset(P[feats], P.y, init_score=np.log(P.mapped_total), categorical_feature=["area_cat"], free_raw_data=False)
    booster = lgb.train(PARAMS, ds, num_boost_round=rounds)
    return AreaModel(booster, feats, P.area_cat.cat.categories)


def _scores(test: pd.DataFrame, share: np.ndarray, name: str) -> dict:
    t = test.assign(share=share)
    t = t[t.mapped_total > 0]
    ll = (t.y * np.log(np.maximum(t.share, 1e-12))).sum() / t.y.sum()
    hit10, hit20 = [], []
    for _, g in t.groupby("date"):
        order = g.share.to_numpy().argsort()[::-1]
        y = g.y.to_numpy()[order]
        hit10.append((y[: int(np.ceil(0.1 * len(y)))].sum(), y.sum()))
        hit20.append((y[:20].sum(), y.sum()))
    h10, h20 = np.array(hit10).sum(0), np.array(hit20).sum(0)
    return dict(model=name, loglik_per_call=round(ll, 4), calls_in_pred_top10pct=round(h10[0] / h10[1], 3),
                calls_in_pred_top20_areas=round(h20[0] / h20[1], 3), n_calls=int(t.y.sum()))


def evaluate(P: pd.DataFrame, seasons=TEST_SEASONS) -> pd.DataFrame:
    rows = []
    for season in seasons:
        tr, te = P[P.year < season], P[P.year == season]
        if te.y.sum() == 0:
            continue
        m = fit(tr)
        hist = tr.groupby("modzcta").y.sum()
        base = {
            "historical share": te.modzcta.map(hist / hist.sum()).fillna(0).to_numpy() + 1e-6,
            "resident population share": te.pop_est.to_numpy(),
            "daytime presence share": te.daytime_presence.to_numpy(),
            # NOT a forecast: the test season's own realised shares (ceiling for any static map; Poisson noise remains)
            "oracle: realised season shares": te.modzcta.map(te.groupby("modzcta").y.sum() + 0.5).to_numpy(),
        }
        for name, w in base.items():
            sh = w / pd.Series(w, index=te.index).groupby(te.date.to_numpy()).transform("sum").to_numpy()
            rows.append({"season": season, **_scores(te, sh, name)})
        rows.append({"season": season, **_scores(te, m.shares(te), "LightGBM share model")})
        print(f"[area] season {season} done", flush=True)
    return pd.DataFrame(rows)


def run(paths: Paths = PATHS, evaluate_seasons: bool = True) -> AreaModel:
    out = paths.out("models/area")
    P = prepare(pd.read_parquet(paths.processed / "panel.parquet"))
    if evaluate_seasons:
        E = evaluate(P)
        E.to_csv(out / "area_scores_by_season.csv", index=False)
        summary = E.groupby("model")[["loglik_per_call", "calls_in_pred_top10pct", "calls_in_pred_top20_areas"]].mean()
        summary.round(4).to_csv(out / "area_scores_summary.csv")
        print(summary.round(4).to_string())
    m = fit(P)
    m.save(out / "area_model.txt")
    imp = pd.Series(m.booster.feature_importance("gain"), index=m.features).sort_values(ascending=False)
    (imp / imp.sum()).round(4).rename("gain_share").to_csv(out / "feature_importance.csv")
    return m
