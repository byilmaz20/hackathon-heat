r"""Dynamic neighbourhood shares: can tomorrow's *distribution* of calls across neighbourhoods be forecast?

The fixed-share model gives neighbourhood i the same share pi_i of the city's calls every day. This model lets the
share move with the day:

    y_{i,d} ~ Poisson(N_d * pi_{i,d}),      pi_{i,d} = exp(eta_{i,d}) / sum_k exp(eta_{k,d})

    eta_{i,d} = a_i                                   usual share of neighbourhood i
              + b_i * heat_d                          its own heat sensitivity          (ridge-shrunk to 0)
              + sum_j c_j * heat_d * Z_{ij}           heat sensitivity explained by traits (AC, 65+, surface temperature,
                                                      poverty, park share; standardised)
              + w_i * weekend_d                       its own weekend / holiday effect  (ridge-shrunk to 0)
              + g_1 * local_lag_{i,d}                 log(1 + calls in i on d-1 before 18:00)
              + g_2 * city_lag_d                      log(1 + citywide calls on d-1 before 18:00) (absorbs city level)

    heat_d = max(0, apparent max - 26 C) / 5  - "per 5 C above 79 F". Training: observed weather; test: the
    evening-before forecast. N_d enters as an offset (Poisson-multinomial equivalence), so only shares are learnt.
FITTING   penalised IRLS (Poisson); the ridge strength for b_i and w_i is chosen on the last training summer from
          {5, 20, 80}, then the model is refitted on all training summers (2013 onwards: the 65+ share uses ACS d-2).
EVALUATION on 2024, 2025, 2026 against the fixed historical share:
          (1) share accuracy given the true city total: log-likelihood per call, calls in the predicted top-4
          (2) next-day neighbourhood forecasts (city model x shares): MAE and CRPS, all days and busy days
Run: python -m heattents.models.dynamic_share
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .glm import fit_glm
from .neighbourhood import GBM_FIRST_YEAR, TEST_SEASONS, historical_share, score

HEAT_BASE_C = 26.0
TRAITS = ["households_with_ac_pct", "pct_65plus", "surface_temp_f", "poverty_pct", "park_share"]
RIDGE_GRID = (5.0, 20.0, 80.0)


class DynamicShare:
    def __init__(self, ridge: float):
        self.ridge = ridge

    # --- design ------------------------------------------------------------------------------------------------------
    def _design(self, P: pd.DataFrame, weather: str) -> np.ndarray:
        u = P.uhf.map(self.index).to_numpy()
        n, K = len(P), len(self.uhfs)
        heat = np.maximum(P[f"atmax_{weather}"].to_numpy() - HEAT_BASE_C, 0) / 5
        wk = ((P.weekend == 1) | (P.holiday == 1)).to_numpy().astype(float)
        onehot = np.zeros((n, K))
        onehot[np.arange(n), u] = 1.0
        Z = (P[TRAITS].to_numpy(float) - self.mu) / self.sd
        local = P.lag_calls.to_numpy()
        city = np.log1p(P.city_lag_n.to_numpy())
        return np.column_stack([onehot, onehot * heat[:, None], heat[:, None] * Z, onehot * wk[:, None], local, city])

    def _penalty(self) -> np.ndarray:
        K = len(self.uhfs)
        return np.r_[np.zeros(K), np.full(K, self.ridge), np.full(len(TRAITS), 1.0), np.full(K, self.ridge), 1.0, 1.0]

    # --- fit / predict -----------------------------------------------------------------------------------------------
    def fit(self, P: pd.DataFrame) -> "DynamicShare":
        T = P[(P.year >= GBM_FIRST_YEAR) & (P.mapped_total > 0)]
        self.uhfs = np.sort(T.uhf.unique())
        self.index = {u: i for i, u in enumerate(self.uhfs)}
        self.mu, self.sd = T[TRAITS].mean().to_numpy(), T[TRAITS].std().to_numpy() + 1e-9
        X = self._design(T, "obs")
        b0 = np.zeros(X.shape[1])
        b0[:len(self.uhfs)] = np.log(historical_share(T).reindex(self.uhfs).to_numpy())
        self.b = fit_glm(X, T.y.to_numpy(float), 0.0, self._penalty(), offset=np.log(T.mapped_total.to_numpy()), b0=b0)
        return self

    def shares(self, P: pd.DataFrame, weather: str = "fc") -> np.ndarray:
        raw = np.exp(np.clip(self._design(P, weather) @ self.b, -30, 30))
        return raw / pd.Series(raw, index=P.index).groupby(P.date.to_numpy()).transform("sum").to_numpy()

    def effects(self) -> pd.DataFrame:
        K = len(self.uhfs)
        return pd.DataFrame({"uhf": self.uhfs, "heat_sensitivity_per_5C": np.exp(self.b[K:2 * K]) - 1,
                             "weekend_effect": np.exp(self.b[2 * K + len(TRAITS):3 * K + len(TRAITS)]) - 1})


def add_city_lag(P: pd.DataFrame) -> pd.DataFrame:
    city = P.groupby("date").lag_calls.apply(lambda s: np.expm1(s).sum()).rename("city_lag_n")
    return P.drop(columns=["city_lag_n"], errors="ignore").merge(city, left_on="date", right_index=True)


def fit_tuned(train: pd.DataFrame) -> DynamicShare:
    """Choose the ridge strength on the last training summer, then refit on all training summers."""
    last = train.year.max()
    inner_tr, inner_va = train[train.year < last], train[train.year == last]
    best = max(RIDGE_GRID, key=lambda r: score(inner_va, DynamicShare(r).fit(inner_tr).shares(inner_va, "obs"))["loglik_per_call_all"])
    return DynamicShare(best).fit(train)


def dynamic_share_method(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """For combined.forecast: (date x neighbourhood) shares from the dynamic model fitted on ``train``."""
    m = fit_tuned(train)
    te = test.copy()
    te["share"] = m.shares(te, "fc")
    return te.pivot_table(index="date", columns="uhf", values="share")


def load_panel(paths: Paths = PATHS) -> pd.DataFrame:
    P = pd.read_parquet(paths.processed / "neighbourhood_panel.parquet")
    P["uhf"] = P.uhf.astype(str)
    return add_city_lag(P)


def run(paths: Paths = PATHS) -> pd.DataFrame:
    from .combined import accuracy, fixed_share_method, forecast
    out = paths.out("models/neighbourhood/dynamic")
    P = load_panel(paths)
    rows, effects = [], []
    for s in TEST_SEASONS:
        tr, te = P[P.year < s], P[P.year == s]
        m = fit_tuned(tr)
        fixed = te.uhf.map(historical_share(tr)).to_numpy()
        dyn = m.shares(te, "fc")
        rows += [dict(season=s, method="fixed share", **score(te, fixed)),
                 dict(season=s, method="dynamic share", ridge=m.ridge, **score(te, dyn))]
        # how much do the dynamic shares actually move from day to day?
        sh = te.assign(share=dyn).pivot_table(index="date", columns="uhf", values="share")
        rows[-1]["mean_daily_share_change_pct"] = float((sh.std() / sh.mean()).mean() * 100)
        effects.append(m.effects().assign(season=s))
        print(f"[dynamic] season {s}: ridge {m.ridge}", flush=True)
    S = pd.DataFrame(rows)
    S.round(4).to_csv(out / "share_scores.csv", index=False)
    pd.concat(effects).round(4).to_csv(out / "neighbourhood_effects.csv", index=False)
    print(S.groupby("method", sort=False).mean(numeric_only=True).drop(columns="season").round(4).to_string())

    # next-day neighbourhood forecasts: identical city samples, only the shares differ
    acc = []
    for name, method in (("fixed share", fixed_share_method), ("dynamic share", dynamic_share_method)):
        D = forecast(paths, share_method=lambda tr, te, method=method: method(add_city_lag(tr), add_city_lag(te)))
        D.round(4).to_csv(out / f"forecast_daily_{name.replace(' ', '_')}.csv", index=False)
        acc.append(accuracy(D).assign(share_model=name))
    A = pd.concat(acc)
    A = A[A.method == "forecast"].drop(columns=[c for c in A.columns if "skill" in c] + ["method"])
    A.round(4).to_csv(out / "forecast_accuracy.csv", index=False)
    print(A.round(3).to_string(index=False))
    return A


if __name__ == "__main__":
    from heattents.models.dynamic_share import run as _run
    _run()


# =====================================================================================================================
# Local-heat share model (HRRR neighbourhood weather)
# =====================================================================================================================
LOCAL_FIRST_YEAR = 2005  # Daymet neighbourhood weather from 2005


class LocalHeatShare:
    r"""eta_{i,d} = a_i + b1 * anom_{i,d} + b2 * anom_{i,d} * cityheat_d + w_i * weekend_d

    anom_{i,d}   neighbourhood i's max heat index minus the 42-neighbourhood mean that day (deg C, HRRR)
    cityheat_d   max(0, mean heat index - 26 C) / 5
    a_i free; b1, b2 shared (light ridge); w_i ridge-shrunk to 0 (strength chosen on the last training summer).
    """

    def __init__(self, ridge: float):
        self.ridge = ridge

    def _design(self, P: pd.DataFrame, src: str) -> np.ndarray:
        u = P.uhf.map(self.index).to_numpy()
        n, K = len(P), len(self.uhfs)
        onehot = np.zeros((n, K))
        onehot[np.arange(n), u] = 1.0
        anom = P[f"local_anom_{src}"].to_numpy()
        city = np.maximum(P[f"local_mean_{src}"].to_numpy() - HEAT_BASE_C, 0) / 5
        wk = ((P.weekend == 1) | (P.holiday == 1)).to_numpy().astype(float)
        return np.column_stack([onehot, anom, anom * city, onehot * wk[:, None]])

    def _penalty(self) -> np.ndarray:
        K = len(self.uhfs)
        return np.r_[np.zeros(K), 0.1, 0.1, np.full(K, self.ridge)]

    def fit(self, P: pd.DataFrame) -> "LocalHeatShare":
        T = P[(P.year >= LOCAL_FIRST_YEAR) & (P.mapped_total > 0) & P.local_anom_est.notna()]
        self.uhfs = np.sort(T.uhf.unique())
        self.index = {u: i for i, u in enumerate(self.uhfs)}
        X = self._design(T, "est")
        b0 = np.zeros(X.shape[1])
        b0[:len(self.uhfs)] = np.log(historical_share(T).reindex(self.uhfs).to_numpy())
        self.b = fit_glm(X, T.y.to_numpy(float), 0.0, self._penalty(), offset=np.log(T.mapped_total.to_numpy()), b0=b0)
        return self

    def shares(self, P: pd.DataFrame, src: str = "fc") -> np.ndarray:
        raw = np.exp(np.clip(self._design(P, src) @ self.b, -30, 30))
        return raw / pd.Series(raw, index=P.index).groupby(P.date.to_numpy()).transform("sum").to_numpy()

    def coefficients(self) -> dict:
        K = len(self.uhfs)
        return {"per_C_hotter_than_city": float(np.exp(self.b[K]) - 1),
                "per_C_hotter_x_per_5C_city_heat": float(np.exp(self.b[K + 1]) - 1)}


HRRR_TO_DAYMET = 0.191  # daymet_anomaly ~ 0.191 * hrrr_anomaly (2024-25 overlap, weather only, corr 0.42)


def add_local_heat(P: pd.DataFrame, paths: Paths = PATHS) -> pd.DataFrame:
    """Neighbourhood heat-index anomaly (vs the 42-neighbourhood mean that day):
    _est  Daymet 1 km (training; also a 'perfect local weather' reference on 2024-25 test days)
    _fc   HRRR 3 km evening-before forecast, rescaled to Daymet's scale (testing)."""
    from ..ingest import daymet, weather_local
    est = daymet.fetch(paths)[["uhf", "date", "atmax"]].copy()
    fc = weather_local.forecasts(paths)[["uhf", "date", "atmax"]].copy()
    for src, d, scale in (("est", est, 1.0), ("fc", fc, HRRR_TO_DAYMET)):
        d["uhf"] = d.uhf.astype(str)
        d[f"local_mean_{src}"] = d.groupby("date").atmax.transform("mean")
        d[f"local_anom_{src}"] = scale * (d.atmax - d[f"local_mean_{src}"])
        P = P.drop(columns=[f"local_mean_{src}", f"local_anom_{src}"], errors="ignore").merge(
            d[["uhf", "date", f"local_mean_{src}", f"local_anom_{src}"]], on=["uhf", "date"], how="left")
    return P


def fit_local_tuned(train: pd.DataFrame) -> LocalHeatShare:
    last = train.year.max()
    inner_tr, inner_va = train[train.year < last], train[train.year == last]
    best = max(RIDGE_GRID, key=lambda r: score(inner_va, LocalHeatShare(r).fit(inner_tr).shares(inner_va, "est"))["loglik_per_call_all"])
    return LocalHeatShare(best).fit(train)


def local_share_method(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    m = fit_local_tuned(train)
    te = test.copy()
    te["share"] = m.shares(te, "fc")
    return te.pivot_table(index="date", columns="uhf", values="share")


def run_local(paths: Paths = PATHS) -> pd.DataFrame:
    """Fixed share (2005+), fixed share (2018+) and the local-heat model on 2024-26."""
    from .combined import accuracy, forecast
    out = paths.out("models/neighbourhood/local_heat")
    P = add_local_heat(load_panel(paths), paths)

    def fixed_since(year):
        def method(tr, te):
            share = historical_share(tr[tr.year >= year])
            dates = np.sort(te.date.unique())
            return pd.DataFrame(np.tile(share.to_numpy(), (len(dates), 1)), index=pd.DatetimeIndex(dates), columns=share.index)
        return method

    methods = {"fixed share (2005+)": fixed_since(2005), "fixed share (2018+)": fixed_since(2018),
               "local-heat share": local_share_method}
    rows, coefs = [], []
    for s in TEST_SEASONS:
        tr, te = P[P.year < s], P[P.year == s]
        for name, method in methods.items():
            sh = method(tr, te).stack().rename("share").reset_index()
            sh["uhf"] = sh.uhf.astype(str)
            t = te.merge(sh, on=["date", "uhf"], how="left")
            rows.append(dict(season=s, method=name, **score(t, t.share.to_numpy())))
        coefs.append(dict(season=s, **fit_local_tuned(tr).coefficients()))
    S = pd.DataFrame(rows)
    S.round(4).to_csv(out / "share_scores.csv", index=False)
    pd.DataFrame(coefs).round(4).to_csv(out / "coefficients.csv", index=False)
    print(S.groupby("method", sort=False).mean(numeric_only=True).drop(columns="season").round(4).to_string())
    print(pd.DataFrame(coefs).round(4).to_string(index=False))

    acc, daily = [], {}
    for name, method in methods.items():
        D = forecast(paths, share_method=lambda tr, te, method=method: method(add_local_heat(add_city_lag(tr), paths),
                                                                                add_local_heat(add_city_lag(te), paths)))
        daily[name] = D
        acc.append(accuracy(D).query("method == 'forecast'").assign(share_model=name))
    A = pd.concat(acc)[["share_model", "days", "mae", "rmse", "crps", "coverage_90"]]
    A.round(4).to_csv(out / "forecast_accuracy.csv", index=False)
    print(A.round(3).to_string(index=False))

    # bias over many days, by group (daily noise averages out)
    anom = P[["uhf", "date", "local_anom_fc", "weekend", "holiday"]].copy()
    M = None
    for name, D in daily.items():
        d = D[["date", "uhf", "actual", "forecast"]].rename(columns={"forecast": name})
        M = d if M is None else M.merge(d.drop(columns="actual"), on=["date", "uhf"])
    M = M.merge(anom, on=["uhf", "date"], how="left")
    groups = {"locally hot (forecast >= 2 C above city mean)": M.local_anom_fc >= 2,
              "locally cool (<= -2 C)": M.local_anom_fc <= -2,
              "weekend/holiday": (M.weekend == 1) | (M.holiday == 1), "all days": M.local_anom_fc.notna() | True}
    brow = []
    for g, mask in groups.items():
        t = M[mask].groupby("uhf")[["actual"] + list(methods)].sum()
        t = t[t.actual >= 15]
        brow.append(dict(group=g, neighbourhoods=len(t), calls=int(t.actual.sum()),
                         **{f"bias_{k}": float(np.abs(np.log(t[k] / t.actual)).mean()) for k in methods},
                         **{f"total_ratio_{k}": float(t[k].sum() / t.actual.sum()) for k in methods}))
    B = pd.DataFrame(brow)
    B.round(4).to_csv(out / "group_bias.csv", index=False)
    print(B.round(3).to_string(index=False))
    return A


def run_local_cv(paths: Paths = PATHS, ridge: float = 80.0) -> pd.DataFrame:
    """Quick check with the archived HRRR day-ahead forecasts only (2024-26), leave-one-season-out.

    Usual shares a_i are fixed from 2005-2023 (no test-period information); only b1, b2 and the weekend effects
    w_i are learnt, from the two summers other than the one being tested.
    """
    from ..ingest import weather_local
    from .combined import accuracy, forecast
    out = paths.out("models/neighbourhood/local_heat")
    P = load_panel(paths)
    fc = weather_local.forecasts(paths)[["uhf", "date", "atmax"]].copy()
    fc["uhf"] = fc.uhf.astype(str)
    fc["local_mean_fc"] = fc.groupby("date").atmax.transform("mean")
    fc["local_anom_fc"] = fc.atmax - fc.local_mean_fc
    P = P.merge(fc.drop(columns="atmax"), on=["uhf", "date"], how="left")
    base_all = historical_share(P[P.year < 2024])
    base_recent = historical_share(P[(P.year >= 2018) & (P.year < 2024)])
    seasons = [2024, 2025, 2026]

    def design(T, K_index):
        n = len(T)
        onehot = np.zeros((n, len(K_index)))
        onehot[np.arange(n), T.uhf.map(K_index).to_numpy()] = 1.0
        anom = T.local_anom_fc.to_numpy()
        city = np.maximum(T.local_mean_fc.to_numpy() - HEAT_BASE_C, 0) / 5
        wk = ((T.weekend == 1) | (T.holiday == 1)).to_numpy().astype(float)
        return np.column_stack([anom, anom * city, onehot * wk[:, None]])

    uhfs = np.sort(base_all.index.to_numpy())
    K_index = {u: i for i, u in enumerate(uhfs)}
    fitted, coefs = {}, []
    for s in seasons:
        T = P[P.year.isin([x for x in seasons if x != s]) & (P.mapped_total > 0) & P.local_anom_fc.notna()]
        off = np.log(T.mapped_total.to_numpy()) + np.log(T.uhf.map(base_all).to_numpy())
        X = design(T, K_index)
        b = fit_glm(X, T.y.to_numpy(float), 0.0, np.r_[0.1, 0.1, np.full(len(uhfs), ridge)], offset=off)
        fitted[s] = b
        coefs.append(dict(test_season=s, per_C_hotter_than_city=np.exp(b[0]) - 1, per_C_x_per_5C_city_heat=np.exp(b[1]) - 1))

    def local_shares(te):
        s = int(te.year.iloc[0])
        raw = np.exp(np.log(te.uhf.map(base_all).to_numpy()) + design(te, K_index) @ fitted[s])
        return raw / pd.Series(raw, index=te.index).groupby(te.date.to_numpy()).transform("sum").to_numpy()

    def as_table(te, shares):
        return te.assign(share=shares).pivot_table(index="date", columns="uhf", values="share")

    methods = {"fixed share (2005-2023)": lambda te: te.uhf.map(base_all).to_numpy(),
               "fixed share (2018-2023)": lambda te: te.uhf.map(base_recent).to_numpy(),
               "local-heat share": local_shares}
    rows = [dict(season=s, method=n, **score(P[P.year == s], f(P[P.year == s]))) for s in seasons for n, f in methods.items()]
    S = pd.DataFrame(rows)
    S.round(4).to_csv(out / "cv_share_scores.csv", index=False)
    C = pd.DataFrame(coefs)
    C.round(4).to_csv(out / "cv_coefficients.csv", index=False)
    print(S.groupby("method", sort=False).mean(numeric_only=True).drop(columns=["season", "calls", "busy_days"]).round(4).to_string())
    print(C.round(4).to_string(index=False))

    daily, acc = {}, []
    for name, f in methods.items():
        D = forecast(paths, share_method=lambda tr, te, f=f: as_table(te.merge(fc.drop(columns="atmax"), on=["uhf", "date"], how="left")
                                                                       if "local_anom_fc" not in te else te,
                                                                       f(te.merge(fc.drop(columns="atmax"), on=["uhf", "date"], how="left")
                                                                         if "local_anom_fc" not in te else te)))
        daily[name] = D
        acc.append(accuracy(D).query("method == 'forecast'").assign(share_model=name))
    A = pd.concat(acc)[["share_model", "days", "mae", "rmse", "crps"]]
    A.round(4).to_csv(out / "cv_forecast_accuracy.csv", index=False)
    print(A.round(4).to_string(index=False))

    M = None
    for name, D in daily.items():
        d = D[["date", "uhf", "actual", "forecast"]].rename(columns={"forecast": name})
        M = d if M is None else M.merge(d.drop(columns="actual"), on=["date", "uhf"])
    M = M.merge(P[["uhf", "date", "local_anom_fc", "weekend", "holiday"]], on=["uhf", "date"], how="left")
    groups = {"locally hot (>= 2 C above city mean)": M.local_anom_fc >= 2, "locally hot (>= 1 C)": M.local_anom_fc >= 1,
              "locally cool (<= -1 C)": M.local_anom_fc <= -1, "all days": M.local_anom_fc.notna()}
    brow = []
    for g, mask in groups.items():
        sub = M[mask]
        brow.append(dict(group=g, cells=int(mask.sum()), actual_calls=int(sub.actual.sum()),
                         **{f"forecast/actual {k}": round(float(sub[k].sum() / sub.actual.sum()), 3) for k in methods}))
    B = pd.DataFrame(brow)
    B.to_csv(out / "cv_group_totals.csv", index=False)
    print(B.to_string(index=False))
    return S
