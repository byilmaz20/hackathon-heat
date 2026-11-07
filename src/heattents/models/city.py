r"""Citywide daily probabilistic forecast of NYC EMS heat-illness calls.

QUESTION   At 18:00 on day d-1 a planner asks: how many heat EMS calls will New York City have on day d?

INPUTS (everything known at 18:00 on d-1)
    weather of day d       archived 1-day-ahead forecast at Central Park (Open-Meteo Previous Runs):
                           max apparent temperature, max / min / mean temperature, humidity
    weather of d-1 .. d-3  observed (ERA5): apparent max at lags 1-3, mean temperatures
    heatwave structure     EHF (3-day mean temperature vs climatology and vs the previous 30 days), consecutive
                           hot days (apparent max >= 32 C), hot days so far this season, tropical night (tmin >= 24 C)
    lead-lag               yesterday's apparent max in Philadelphia / Baltimore / DC minus NYC ("heat on its way");
                           calls on d-1 before 18:00
    calendar & trend       weekend, federal holiday, 4 July window, school break, COVID years, year, day of year

MODELS (both see the same information; the forecast used is their average, fixed in advance)
    glm   negative binomial regression  y_d ~ NB2(mu_d, alpha)
              log mu_d = b0 + DLNM cross-basis(apparent max, lags 0-3)          heat x lag response surface
                         + b * EHF + b * streak + b * season hot days + b * tropical night
                         + b * max(0, upstream gradient) + b * humid heat + b * log(1 + calls before 18:00 on d-1)
                         + calendar dummies + spline(year) + spline(day of year)
          cross-basis: natural cubic spline in apparent max (knots 26/31/35 C, linear beyond 10/44 C) x quadratic
          polynomial over lags 0..3 (Gasparrini, Armstrong & Kenward 2010); ridge-penalised IRLS, alpha profiled.
    gbm   LightGBM, Poisson loss, monotone increasing in the heat variables
    ens   (glm + gbm) / 2 - the GLM extrapolates to record heat, the GBM fits ordinary days

UNCERTAINTY
    weather:  M = 30 Monte-Carlo draws of day d's weather, truth = forecast + e, with e resampled from *analog* real
              errors: 1-day-ahead forecasts already verified at decision time (target days <= d-2), in May-September,
              whose forecast apparent max was within 3 C of today's. (Unconditional all-year errors inflated the
              2024-26 total by 21% through the convex heat response; analog errors: +5%, CRPS -4%, tail CRPS -13%.)
              Early May 2024 has no verified summer errors yet -> Gaussian N(0, 2 C) fallback (21 days).
    counts:   NB2 noise with the GLM's dispersion alpha;   p(y) = (1/M) sum_m NB(y; mu(x_m), alpha)

TRAIN / TEST (rolling origin by season)
    test seasons 2024, 2025, 2026 (May-September). For season s the models are trained on all seasons 2005..s-1 with
    *observed* weather (perfect prog), then every day of s is forecast from the 18:00 information set above.
    Why only 2024-26: the forecast archive holds humidity / apparent temperature only from 19 Jan 2024; for 2022-23
    it has a single model's (GFS) air temperature, ~2.5 C too warm in apparent terms once humidity is imputed.
    Baselines on the same days: climatology (day-of-season mean), persistence (yesterday's calls), temperature-only GLM.
Run: python -m heattents.models.city
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .glm import cross_basis, fit_glm, fit_nb, nb_pmf, ns_basis

KINDS = ("glm", "gbm", "ens")
FORECAST_KIND = "ens"  # chosen before looking at test results
WX = ["tmax", "tmin", "tmean", "atmax", "rh"]
HOT = 32.0
SEASON = (5, 6, 7, 8, 9)
KMAX = 700
DECISION_HOUR = 18
CALENDAR = ["weekend", "holiday", "july4_window", "school_break", "covid", "year", "day_of_year"]
GBM_FEATURES = ["atmax", "atmax_l1", "atmax_l2", "atmax_l3", "tmax", "tmin", "tmean", "rh", "ehf", "ehi_sig", "ehi_accl",
                "streak", "season_hot_days", "tropical_night", "up_sw_gradient", "lag_count"] + CALENDAR
MONOTONE = {"atmax": 1, "atmax_l1": 1, "tmax": 1, "ehf": 1, "streak": 1}
TEST_SEASONS = (2024, 2025, 2026)
MC_DRAWS = 30
ANALOG_C = 3.0  # analog window for forecast errors (deg C of forecast apparent max)
MIN_POOL = 20
FALLBACK_SD = 2.0
TW_THRESHOLD = 50  # threshold-weighted CRPS: days with >= 50 calls are where staffing decisions bite


# =====================================================================================================================
# Features
# =====================================================================================================================
def assemble_features(atmax0, atmax1, atmax2, atmax3, tmean0, tmean1, tmean2, tmin0, tmax0, rh0, streak_prev,
                      season_count_before, base30, up_grad, calendar: dict, t95: float, calls_by18=None) -> pd.DataFrame:
    """Model features of day d from raw values (arrays broadcast together).

    streak_prev          consecutive days with apparent max >= 32 C ending on d-1
    season_count_before  number of such days this year before d
    base30               mean daily temperature over d-32..d-3 (acclimatisation baseline of the EHF)
    calls_by18           calls on d-1 before 18:00
    """
    a0 = np.asarray(atmax0, float)
    m3 = (np.asarray(tmean0, float) + np.asarray(tmean1, float) + np.asarray(tmean2, float)) / 3
    sig, accl = m3 - t95, m3 - np.asarray(base30, float)
    h0 = (a0 >= HOT).astype(float)
    streak = np.minimum(h0 * (1 + np.asarray(streak_prev, float)), 7)
    f = pd.DataFrame({
        "atmax": a0, "atmax_l1": atmax1, "atmax_l2": atmax2, "atmax_l3": atmax3,
        "tmax": tmax0, "tmin": tmin0, "tmean": tmean0, "rh": rh0,
        "ehi_sig": sig, "ehi_accl": accl, "ehf": np.maximum(sig, 0) * np.maximum(1, np.nan_to_num(accl)),
        "streak": streak, "season_hot_days": np.log1p(np.asarray(season_count_before, float)),
        "tropical_night": (np.asarray(tmin0, float) >= 24).astype(float),
        "up_sw_gradient": up_grad,
    })
    for k, v in calendar.items():
        f[k] = v
    f["lag_count"] = np.log1p(np.asarray(0.0 if calls_by18 is None else calls_by18, float))
    return f


def scenario_features(atmax, tmax=None, tmin=None, tmean=None, rh=70.0, atmax_l1=None, atmax_l2=None, atmax_l3=None,
                      tmean_l1=None, tmean_l2=None, hot_days_before=0, season_hot_days_before=10, tmean_30d=22.0,
                      up_sw_gradient=0.0, weekend=0, holiday=0, july4_window=0, school_break=1, covid=0, year=2026,
                      day_of_year=190, calls_by18=10.0, t95=27.2, date=None) -> pd.DataFrame:
    """One-day (or vectorised) feature row from a user scenario for the simulator. Temperatures in deg C.

    hot_days_before         consecutive days with apparent max >= 32 C immediately before today
    season_hot_days_before  such days so far this year
    tmean_30d               mean temperature over the month before (EHF acclimatisation baseline)
    up_sw_gradient          yesterday's apparent max in Philadelphia/Baltimore/DC minus NYC (heat on its way if > 0)
    calls_by18              heat EMS calls yesterday before 18:00
    """
    atmax = np.atleast_1d(np.asarray(atmax, float))
    tmax = atmax - 3.0 if tmax is None else tmax
    tmin = np.asarray(tmax, float) - 8.0 if tmin is None else tmin
    tmean = (np.asarray(tmax, float) + np.asarray(tmin, float)) / 2 if tmean is None else tmean
    l1 = atmax if atmax_l1 is None else atmax_l1
    l2 = l1 if atmax_l2 is None else atmax_l2
    l3 = l2 if atmax_l3 is None else atmax_l3
    tm1 = tmean if tmean_l1 is None else tmean_l1
    tm2 = tm1 if tmean_l2 is None else tmean_l2
    cal = dict(weekend=weekend, holiday=holiday, july4_window=july4_window, school_break=school_break, covid=covid,
               year=year, day_of_year=day_of_year)
    f = assemble_features(atmax, l1, l2, l3, tmean, tm1, tm2, tmin, tmax, rh, hot_days_before,
                          season_hot_days_before, tmean_30d, up_sw_gradient, cal, t95, calls_by18)
    f.insert(0, "date", pd.Timestamp("2026-07-01") if date is None else pd.Timestamp(date))
    f.insert(1, "draw", np.arange(len(f)) if len(f) > 1 else 0)
    return f


@dataclass
class Context:
    """Citywide daily series, observed weather (full year, Central Park) and archived 1-day-ahead forecasts."""
    city: pd.DataFrame
    obs: pd.DataFrame
    fc: pd.DataFrame  # columns WX, forecast for the index date issued the day before
    errors: pd.DataFrame  # forecast - observed per target date (columns WX)
    t95: float

    @classmethod
    def load(cls, paths: Paths = PATHS) -> "Context":
        from ..features.panel import city_weather, load_ems
        city = pd.read_parquet(paths.processed / "city_daily.parquet").sort_values("date").reset_index(drop=True)
        e = load_ems(paths)
        by18 = e[e.incident_datetime.dt.hour < DECISION_HOUR].groupby("date").size()
        city["calls_l1_by18"] = city.date.map(lambda d: by18.get(d - pd.Timedelta(days=1), 0)).astype(float)
        W = city_weather(paths)
        obs = W.set_index("date")
        raw = pd.read_parquet(paths.raw / "weather_forecasts_central_park.parquet").set_index("date")
        fc = raw[[f"{c}_f1" for c in WX]].copy()
        fc.columns = WX
        errors = (fc - obs[WX].reindex(fc.index)).dropna()
        return cls(city, obs, fc, errors, float(W.attrs["t95"]))

    def error_pool(self, target: pd.Timestamp) -> np.ndarray:
        """Analog forecast errors (observed - forecast) for the forecast of ``target``.

        Only errors already verified at 18:00 on target-1 (target days <= target-2), from May-September, of days
        whose forecast apparent max was within +-ANALOG_C of today's forecast (all summer errors if fewer than
        MIN_POOL such days). Conditioning on the forecast level captures regression to the mean (very hot forecasts
        tend to verify cooler); restricting to summer drops the larger winter errors. Both matter because the
        heat response is convex: too-wide or symmetric errors inflate the expected count.
        """
        hi = target - pd.Timedelta(days=2)
        p = self.errors.loc[:hi]
        p = p[p.index.month.isin(SEASON)]
        near = (self.fc.atmax.reindex(p.index) - self.fc.atmax.get(target, np.nan)).abs() <= ANALOG_C
        if near.sum() >= MIN_POOL:
            p = p[near.to_numpy()]
        return -p.to_numpy(float)


def _features(ctx: Context, rows: pd.DataFrame, day0: np.ndarray) -> pd.DataFrame:
    """Features for ``rows`` given day-d weather ``day0`` (n x WX); days d-1..d-3 observed."""
    dates = pd.DatetimeIndex(rows.date)
    o = {j: ctx.obs.reindex(dates - pd.Timedelta(days=j)) for j in range(4)}
    o1, o2, o3 = (np.array(o[j][WX].to_numpy(), float) for j in (1, 2, 3))
    a0, a1, a2 = day0[:, 3], o1[:, 3], o2[:, 3]
    h1, h2 = (a1 >= HOT).astype(float), (a2 >= HOT).astype(float)
    streak_prev = h1 * (1 + h2 * (1 + o[3].streak.to_numpy()))
    count_before = np.expm1(o[2].season_hot_days.to_numpy()) + h2 + h1
    base30 = (o[0].ehi_sig + ctx.t95 - o[0].ehi_accl).to_numpy()
    cal = {c: rows[c].to_numpy() for c in CALENDAR}
    return assemble_features(a0, a1, a2, o3[:, 3], day0[:, 2], o1[:, 2], o2[:, 2], day0[:, 1], day0[:, 0], day0[:, 4],
                             streak_prev, count_before, base30, o[0].up_sw_gradient_l1.to_numpy(), cal, ctx.t95,
                             rows.calls_l1_by18.to_numpy())


def observed_features(ctx: Context, rows: pd.DataFrame) -> pd.DataFrame:
    """Training features: day d's *observed* weather (perfect prog)."""
    f = _features(ctx, rows, np.array(ctx.obs.reindex(pd.DatetimeIndex(rows.date))[WX].to_numpy(), float))
    f.insert(0, "date", rows.date.to_numpy())
    f.insert(1, "draw", 0)
    return f


def decision_features(ctx: Context, rows: pd.DataFrame, rng, n_draws: int = MC_DRAWS) -> tuple[pd.DataFrame, int]:
    """Forecast-time features: day d from the archived 1-day-ahead forecast, n_draws Monte-Carlo weather draws.
    Returns (features with a 'draw' column, number of days that needed the Gaussian fallback)."""
    dates = pd.DatetimeIndex(rows.date)
    fcst = ctx.fc.reindex(dates)
    if fcst.isna().any().any():
        missing = fcst.index[fcst.isna().any(axis=1)]
        raise ValueError(f"no archived forecast for {len(missing)} days, e.g. {missing[0].date()}")
    base = np.array(fcst.to_numpy(), float)
    pools = [ctx.error_pool(d) for d in dates]
    fallback = sum(len(p) < MIN_POOL for p in pools)
    frames = []
    for m in range(n_draws):
        day0 = base.copy()
        for i, p in enumerate(pools):
            if len(p) >= MIN_POOL:
                day0[i] += p[rng.integers(0, len(p))]
            else:
                day0[i, :4] += rng.normal(0, FALLBACK_SD)  # temperatures shift together, humidity unchanged
        f = _features(ctx, rows, day0)
        f.insert(0, "date", dates)
        f.insert(1, "draw", m)
        frames.append(f)
    return pd.concat(frames, ignore_index=True), fallback


# =====================================================================================================================
# Models
# =====================================================================================================================
@dataclass
class Design:
    cb_knots: tuple = (26.0, 31.0, 35.0)
    cb_bound: tuple = (10.0, 44.0)
    yr_knots: tuple = (2010.0, 2015.0, 2020.0)
    yr_bound: tuple = (2005.0, 2026.0)
    doy_knots: tuple = (170.0, 200.0, 230.0)
    doy_bound: tuple = (121.0, 273.0)
    scales: dict = field(default_factory=dict)
    names: list = field(default_factory=list)

    def matrix(self, f: pd.DataFrame) -> np.ndarray:
        fit = not self.scales
        lagged = f[["atmax", "atmax_l1", "atmax_l2", "atmax_l3"]].to_numpy(float)
        cb, s = cross_basis(lagged, self.cb_knots, self.cb_bound, 2, self.scales.get("cb"))
        yr, s_yr = ns_basis(f.year.to_numpy(float), self.yr_knots, self.yr_bound, self.scales.get("yr"))
        doy, s_doy = ns_basis(f.day_of_year.to_numpy(float), self.doy_knots, self.doy_bound, self.scales.get("doy"))
        lin = {
            "ehf": f.ehf / 10, "streak": f.streak / 7, "season_hot_days": f.season_hot_days / 3,
            "tropical_night": f.tropical_night, "up_gradient_pos": np.maximum(f.up_sw_gradient, 0) / 5,
            "humid_heat": (f.rh - 70) / 10 * (f.atmax >= 30), "weekend": f.weekend, "holiday": f.holiday,
            "july4_window": f.july4_window, "school_break": f.school_break, "covid": f.covid,
            "lag_count": f.lag_count / 3,
        }
        L = np.column_stack([np.asarray(v, float) for v in lin.values()])
        X = np.column_stack([np.ones(len(f)), cb, L, yr, doy])
        if fit:
            self.scales = {"cb": s, "yr": s_yr, "doy": s_doy}
            self.names = (["intercept"] + [f"heat_x_lag_{i}" for i in range(cb.shape[1])] + list(lin)
                          + [f"year_spline_{i}" for i in range(yr.shape[1])] + [f"doy_spline_{i}" for i in range(doy.shape[1])])
        return np.nan_to_num(X)


RIDGE = 0.3


@dataclass
class CityModel:
    kind: str
    design: Design
    b: np.ndarray
    alpha: float  # NB dispersion (GLM, profiled likelihood); used for every kind
    gbm: object | None
    t95: float = 27.2

    def means(self, f: pd.DataFrame, kind: str | None = None) -> np.ndarray:
        kind = kind or self.kind
        mu_glm = np.exp(np.clip(self.design.matrix(f) @ self.b, -20, 12))
        if kind == "glm" or self.gbm is None:
            return mu_glm
        mu_gbm = np.maximum(self.gbm.predict(f[GBM_FEATURES].astype(float)), 1e-6)
        return mu_gbm if kind == "gbm" else 0.5 * (mu_glm + mu_gbm)

    def pmf(self, f: pd.DataFrame, kind: str | None = None, kmax: int = KMAX):
        """Mixture pmf over the 'draw' column; returns (dates, mean, pmf)."""
        mu = self.means(f, kind)
        P = nb_pmf(mu, self.alpha, kmax)
        g = pd.Series(np.arange(len(f))).groupby(pd.Index(f.date.to_numpy())).apply(list)
        Pm = np.stack([P[ix].mean(0) for ix in g])
        return g.index, np.array([mu[ix].mean() for ix in g]), Pm / Pm.sum(1, keepdims=True)

    def predict(self, f: pd.DataFrame) -> pd.DataFrame:
        dates, mean, Pm = self.pmf(f)
        F = np.cumsum(Pm, 1)
        q = lambda t: (F >= t).argmax(1)
        return pd.DataFrame({"date": dates, "mean": mean, "q05": q(.05), "q25": q(.25), "q50": q(.5), "q75": q(.75),
                             "q95": q(.95), "p_ge50": 1 - F[:, 49], "p_ge100": 1 - F[:, 99]})

    def sample(self, f: pd.DataFrame, n: int, rng=None) -> np.ndarray:
        """(n_days, n) predictive samples of the daily citywide total."""
        rng = np.random.default_rng(rng)
        _, _, Pm = self.pmf(f)
        F = np.cumsum(Pm, 1)
        u = rng.random((len(F), n))
        return np.stack([np.searchsorted(F[i], u[i]) for i in range(len(F))]).clip(0, F.shape[1] - 1)

    def scenario(self, rng=None, n_draws: int = 1, lead_error_sd: float | None = None, **kw) -> pd.DataFrame:
        """Feature frame for a user scenario; with ``lead_error_sd`` the scenario temperatures are treated as a
        forecast and ``n_draws`` Gaussian perturbations (sd in deg C) are mixed."""
        rng = np.random.default_rng(rng)
        kw = dict(kw)
        if n_draws > 1 and lead_error_sd:
            e = rng.normal(0, lead_error_sd, n_draws)
            for c in ("atmax", "tmax", "tmin", "tmean"):
                if kw.get(c) is not None:
                    kw[c] = np.asarray(kw[c], float) + e
        f = scenario_features(t95=self.t95, **kw)
        f["draw"] = np.arange(len(f))
        f["date"] = f.date.iloc[0]
        return f

    def coefficients(self) -> pd.DataFrame:
        """GLM coefficients with the multiplicative effect exp(b) on expected calls (features are scaled as in
        Design.matrix, e.g. 'streak' per 7 days, 'ehf' per 10 units, 'lag_count' per 3 log-units)."""
        return pd.DataFrame({"term": self.design.names, "coef": self.b, "rate_ratio": np.exp(self.b)})

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "CityModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def fit_city_model(f: pd.DataFrame, y: np.ndarray, kind: str = FORECAST_KIND, t95: float = 27.2,
                   with_gbm: bool = True, seed: int = 7) -> CityModel:
    """Fit on training features ``f`` (one row per day, observed weather) and counts ``y``."""
    d = Design()
    X = d.matrix(f)
    alpha, b = fit_nb(X, np.asarray(y, float), np.r_[0.0, np.full(X.shape[1] - 1, RIDGE)])
    gbm = None
    if with_gbm:
        Xg = f[GBM_FEATURES].astype(float)
        params = dict(objective="poisson", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, bagging_fraction=0.8,
                      bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, seed=seed, verbose=-1, num_threads=4,
                      monotone_constraints=[MONOTONE.get(c, 0) for c in Xg.columns])
        gbm = lgb.train(params, lgb.Dataset(Xg, label=np.asarray(y, float)), num_boost_round=500)
    return CityModel(kind, d, b, alpha, gbm, t95)


# =====================================================================================================================
# Scoring and evaluation
# =====================================================================================================================
def randomized_pit(cdf: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    y = np.minimum(np.asarray(y, int), cdf.shape[1] - 1)
    i = np.arange(len(y))
    lo = np.where(y > 0, cdf[i, np.maximum(y - 1, 0)], 0.0)
    return lo + rng.random(len(y)) * (cdf[i, y] - lo)


def scores(cdf: np.ndarray, y: np.ndarray, mean: np.ndarray, rng, ed=None) -> dict:
    """CRPS, threshold-weighted CRPS (days >= 50), log score, MAE, interval coverage, Brier, PIT histogram."""
    y = np.asarray(y, int)
    i = np.arange(len(y))
    yc = np.minimum(y, cdf.shape[1] - 1)
    pmf = np.diff(np.c_[np.zeros(len(cdf)), cdf], axis=1)
    ks = np.arange(cdf.shape[1])
    sq = (cdf - (ks[None, :] >= y[:, None])) ** 2
    q = lambda t: (cdf >= t).argmax(1)
    pit = randomized_pit(cdf, y, rng)
    out = dict(n=len(y), crps=float(sq.sum(1).mean()), twcrps50=float(sq[:, TW_THRESHOLD:].sum(1).mean()),
               log_score=float(np.log(np.maximum(pmf[i, yc], 1e-300)).mean()), mae=float(np.abs(mean - y).mean()),
               cov50=float(((y >= q(.25)) & (y <= q(.75))).mean()), cov90=float(((y >= q(.05)) & (y <= q(.95))).mean()),
               brier50=float((((1 - cdf[:, 49]) - (y >= 50)) ** 2).mean()),
               pred_sum=float(mean.sum()), obs_sum=int(y.sum()),
               pit_hist="|".join(str(v) for v in np.histogram(pit, bins=10, range=(0, 1))[0]))
    if ed is not None:
        ok = ~np.isnan(ed)
        out["corr_ed"] = float(np.corrcoef(mean[ok], ed[ok])[0, 1]) if ok.sum() > 10 else np.nan
    return out


def baselines(ctx: Context, train: pd.DataFrame, test: pd.DataFrame, f_test: pd.DataFrame) -> dict:
    """{name: (cdf, mean)} for climatology, persistence and a temperature-only GLM, on the same information."""
    ytr = train.city_y.to_numpy(float)
    doy = train.day_of_year.to_numpy()
    clim = lambda days: np.array([ytr[np.abs(doy - d) <= 7].mean() for d in days])
    mu_tr = clim(doy)
    nb_alpha = lambda mu: max(np.sum((ytr - mu) ** 2 - mu) / np.sum(mu ** 2), 1e-3)
    out = {"climatology": clim(test.day_of_year.to_numpy())}
    a = {"climatology": nb_alpha(mu_tr)}
    # persistence: yesterday's calls before 18:00 scaled by the training share of calls made before 18:00
    share = train.calls_l1_by18.sum() / train.city_y_l1.sum()
    mu_p_tr = np.maximum(train.calls_l1_by18.to_numpy() / share, 0.5)
    out["persistence"] = np.maximum(test.calls_l1_by18.to_numpy() / share, 0.5)
    a["persistence"] = nb_alpha(mu_p_tr)
    res = {k: (np.cumsum(nb_pmf(np.maximum(v, 1e-3), a[k], KMAX), 1), v) for k, v in out.items()}
    # temperature-only GLM on apparent max (trained on observed, applied to the same forecast draws)
    B_tr, sc = ns_basis(train.atmax.to_numpy(float), (26.0, 31.0, 35.0), (10.0, 44.0))
    b = fit_glm(np.c_[np.ones(len(B_tr)), B_tr], ytr, 0.0, np.r_[0, np.full(B_tr.shape[1], 1e-3)])
    B_te = ns_basis(f_test.atmax.to_numpy(float), (26.0, 31.0, 35.0), (10.0, 44.0), sc)[0]
    mu = np.exp(np.clip(np.c_[np.ones(len(B_te)), B_te] @ b, -20, 12))
    g = f_test.groupby("date").indices
    dates = sorted(g)
    P = nb_pmf(mu, 0.0, KMAX)
    Pm = np.stack([P[g[d]].mean(0) for d in dates])
    res["temperature_glm"] = (np.cumsum(Pm / Pm.sum(1, keepdims=True), 1), np.array([mu[g[d]].mean() for d in dates]))
    return res


def evaluate(paths: Paths = PATHS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = paths.out("models/city")
    ctx = Context.load(paths)
    C = ctx.city
    F_train = observed_features(ctx, C)
    y_all = C.city_y.to_numpy()
    rows, preds = [], []
    for season in TEST_SEASONS:
        tr, te = (C.year < season).to_numpy(), (C.year == season).to_numpy()
        model = fit_city_model(F_train[tr], y_all[tr], t95=ctx.t95)
        ft, n_fallback = decision_features(ctx, C[te], rng)
        y = y_all[te].astype(int)
        ed = C.ed_visits.to_numpy()[te]
        for kind in KINDS:
            dates, mean, Pm = model.pmf(ft, kind)
            F = np.cumsum(Pm, 1)
            rows.append(dict(season=season, model=kind, fallback_days=n_fallback, **scores(F, y, mean, rng, ed)))
            q = lambda t: (F >= t).argmax(1)
            preds.append(pd.DataFrame({"season": season, "model": kind, "date": dates, "y": y, "mean": mean,
                                       "q05": q(.05), "q50": q(.5), "q95": q(.95), "p_ge50": 1 - F[:, 49],
                                       "p_ge100": 1 - F[:, 99], "ed_visits": ed}))
        for name, (F, mean) in baselines(ctx, C[tr], C[te], ft).items():
            rows.append(dict(season=season, model=name, fallback_days=0, **scores(F, y, mean, rng, ed)))
        print(f"[city] season {season}: {n_fallback} days on fallback errors", flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(out / "metrics_by_season.csv", index=False)
    pd.concat(preds).to_csv(out / "oos_predictions.csv", index=False)
    summary = M.groupby("model").agg(crps=("crps", "mean"), twcrps50=("twcrps50", "mean"), log_score=("log_score", "mean"),
                                     mae=("mae", "mean"), cov50=("cov50", "mean"), cov90=("cov90", "mean"),
                                     brier50=("brier50", "mean"), corr_ed=("corr_ed", "mean"),
                                     pred_sum=("pred_sum", "sum"), obs_sum=("obs_sum", "sum")).sort_values("crps")
    summary.round(4).to_csv(out / "metrics_summary.csv")
    final = fit_city_model(F_train, y_all, t95=ctx.t95)
    final.save(out / "city_model.pkl")
    final.coefficients().round(4).to_csv(out / "glm_coefficients.csv", index=False)
    print(summary.round(3).to_string())
    return M


def load_city_model(paths: Paths = PATHS) -> CityModel:
    return CityModel.load(paths.out("models/city") / "city_model.pkl")


if __name__ == "__main__":
    from heattents.models.city import evaluate as _evaluate  # run the package module so pickled classes resolve
    _evaluate()
