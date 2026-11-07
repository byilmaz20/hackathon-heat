r"""Beach & event share model: how the city's heat calls split across the 177 ZIP areas (MODZCTA) on a given day.

The city model forecasts HOW MANY heat calls New York gets tomorrow; this model forecasts WHERE they land. Expected
calls in area i on day d = city forecast_d x m x pi_{i,d}, with m the share of calls that have a known location.

MODEL (Poisson-multinomial on shares)
    y_{i,d} | N_d ~ Multinomial(N_d, pi_d),   fitted as y_{i,d} ~ Poisson(N_d pi_{i,d})  (Poisson-multinomial equivalence)
    log pi_{i,d} = a_i                                      usual share of area i                       (177)
                 + c(d)                                     day-type constants, absorb normalisation     (13)
                 + sum_k beta_k Z_ik * leisure_d            weekend/holiday x beach, visitor ratio, park (3)
                 + beach_i * open_d * (g1 + g2 leisure_d + g3 heat_d + g4 leisure_d heat_d)   beach season (4)
                 + sum_e gamma_e * 1[event e on d] * route_ie   six recurring events on their routes    (6)
                 + u_i * leisure_d                          own weekend/holiday deviation, ridge-shrunk  (177)
    then normalised so that sum_i pi_{i,d} = 1 every day.
    leisure_d  weekend or federal holiday            open_d  beach season (Saturday before Memorial Day - Labor Day)
    heat_d     citywide forecast heat band (max apparent temperature <30, 30-34, 34-38, >=38 C) / 3
    Z_i        beach flag; visitor ratio log((jobs + subway entries) / residents); park share (standardised)
    events     Pride (last Sun of June), Puerto Rican Day (2nd Sun of June), Dominican Day (2nd Sun of Aug),
               West Indian Day (Labor Day), Mermaid Parade (Sat on/after 18 June), US Open (2 weeks from the last
               Mon of Aug); route = ZIP areas on the route / around the venue; cancelled in 2020-21.
FITTING   penalised IRLS (models.glm), summers weighted 0.5^(years before the latest / 8); ridge on u_i chosen on the
          last training summer from {3, 10, 30, 100}. Every day input is categorical, so the 594k area-days collapse
          exactly (Poisson sufficiency) to a few thousand (area x day-type) cells: fitting takes seconds.
WHY THIS DESIGN  Within a summer, day-to-day scatter between areas is no larger than chance (Pearson dispersion
          0.8-0.93), recent local calls do not predict tomorrow's (autocorrelation ~0.01), and heat changes how many
          calls, not where. What does move the map, and is known the evening before: weekends, beach season, events.
          Alternatives tested and not kept (heat x vulnerability, per-area heat slopes, neighbourhood temperature,
          a direct 42-neighbourhood panel model) did not beat this model out of sample.
EVALUATION  rolling origin over 2018-26: log-likelihood per call given the true total, by day type; calibration of
          the predicted moves; daily ratio error 1/2 sum|forecast share - observed share| with its noise floor.
Run: python -m heattents.models.context_share
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from ..features.calendar import _last_weekday, _nth_weekday
from .glm import fit_glm

TEST_SEASONS = [2018, 2019, 2022, 2023, 2024, 2025, 2026]
HEAT_BANDS_C = (30.0, 34.0, 38.0)  # citywide max apparent temperature cut points
BEACH = ["11224", "11235", "11693", "11694", "11697", "10464", "10305", "10306"]
# recurring permitted events: date rule and the MODZCTAs along the route / around the venue
EVENTS = {
    "pride": (lambda y: [_last_weekday(y, 6, 6)], ["10011", "10014", "10003", "10010", "10001"]),
    "puerto_rican_day": (lambda y: [_nth_weekday(y, 6, 6, 2)], ["10022", "10065", "10021", "10075", "10028", "10017", "10029"]),
    "dominican_day": (lambda y: [_nth_weekday(y, 8, 6, 2)], ["10036", "10019", "10018", "10001"]),
    "west_indian_day": (lambda y: [_nth_weekday(y, 9, 0, 1)], ["11216", "11225", "11213", "11238"]),
    "mermaid_parade": (lambda y: [pd.Timestamp(y, 6, 18) + pd.Timedelta(days=(5 - pd.Timestamp(y, 6, 18).day_of_week) % 7)], ["11224"]),
    "us_open": (lambda y: list(pd.date_range(_last_weekday(y, 8, 0), periods=14)), ["11368"]),
}
EVENTS_CANCELLED = {2020, 2021}
TRAITS = ["beach", "visitor_ratio", "park_share"]
LEISURE_TRAITS = TRAITS
RIDGE_GRID = [3.0, 10.0, 30.0, 100.0]  # ridge on the area-specific weekend/holiday deviations u_i
CELL = ["modzcta", "leisure", "band", "beach_open", "event"]
HALF_LIFE_YEARS: float | None = 8.0  # recency weighting of training summers (None = all summers equal)


# --- day context ------------------------------------------------------------------------------------------------------
def event_calendar(years) -> pd.Series:
    """date -> event name (one event per date; US Open loses to a parade on the same day)."""
    out = {}
    for y in years:
        if y in EVENTS_CANCELLED:
            continue
        for name, (rule, _) in EVENTS.items():
            for d in rule(y):
                out.setdefault(pd.Timestamp(d), name) if name == "us_open" else out.__setitem__(pd.Timestamp(d), name)
    return pd.Series(out, name="event")


def day_context(P: pd.DataFrame) -> pd.DataFrame:
    """Add leisure, heat band, beach season and event columns to panel rows (needs date, weekend, holiday, atmax)."""
    P = P.copy()
    d = pd.to_datetime(P.date)
    P["leisure"] = ((P.weekend == 1) | (P.holiday == 1)).astype(int)
    P["band"] = np.searchsorted(HEAT_BANDS_C, P.atmax.to_numpy(), side="right")
    years = range(d.dt.year.min(), d.dt.year.max() + 1)
    open_ = np.zeros(len(P), bool)
    for y in years:  # NYC beaches: Memorial Day weekend (Saturday) to Labor Day
        open_ |= ((d >= _last_weekday(y, 5, 0) - pd.Timedelta(days=2)) & (d <= _nth_weekday(y, 9, 0, 1))).to_numpy()
    P["beach_open"] = open_.astype(int)
    ev = event_calendar(years)
    P["event"] = d.map(ev).fillna("none").to_numpy()
    return P


def area_traits(P: pd.DataFrame) -> pd.DataFrame:
    S = P.drop_duplicates("modzcta").set_index("modzcta")
    T = pd.DataFrame(index=S.index)
    T["beach"] = S.index.isin(BEACH).astype(float)
    T["visitor_ratio"] = np.log((S.jobs + S.subway_entries + 100) / (S.pop_est + 100))
    for c in TRAITS[2:]:
        T[c] = S[c].fillna(S[c].median())
    return T


# --- model ------------------------------------------------------------------------------------------------------------
class ContextShare:
    def __init__(self, ridge_leisure: float = 30.0, half_life: float | None = HALF_LIFE_YEARS):
        self.ridge_leisure, self.half_life = ridge_leisure, half_life

    def _design(self, C: pd.DataFrame):
        """Design matrix for cells (or rows) C with columns CELL; returns (X, penalty)."""
        n, K = len(C), len(self.areas)
        idx = C.modzcta.map(self.index).to_numpy()
        if np.isnan(idx.astype(float)).any():
            raise ValueError("unknown area in input")
        A = np.zeros((n, K))
        A[np.arange(n), idx.astype(int)] = 1.0
        Z = self.Z[idx.astype(int)]  # (n, traits), standardised
        lei = C.leisure.to_numpy(float)
        heat = C.band.to_numpy(float) / 3
        bo = C.beach_open.to_numpy(float)
        beach = Z[:, TRAITS.index("beach")]
        cols, pen = [], []

        def add(block, p):
            block = np.atleast_2d(block.T).T if block.ndim == 1 else block
            cols.append(block)
            pen.extend([p] * block.shape[1])

        add(A, 1e-4)  # a_i (tiny ridge for identifiability)
        add(np.column_stack([lei, *(C.band.to_numpy() == b for b in (1, 2, 3)), bo, lei * bo, lei * heat]).astype(float), 1e-4)
        add(np.column_stack([(C.event == e).to_numpy(float) for e in EVENTS]), 1e-4)  # event-day main effects (tiny ridge: an event may be absent)
        add(lei[:, None] * Z[:, [TRAITS.index(t) for t in LEISURE_TRAITS]], 1.0)
        add(np.column_stack([bo * beach, bo * lei * beach, bo * heat * beach, bo * lei * heat * beach]), 1.0)
        add(np.column_stack([(C.event == e).to_numpy(float) * C.modzcta.isin(r).to_numpy(float)
                             for e, (_, r) in EVENTS.items()]), 2.0)
        add(A * lei[:, None], self.ridge_leisure)  # u_i
        return np.column_stack(cols), np.array(pen)

    def cells(self, P: pd.DataFrame) -> pd.DataFrame:
        """Collapse area x day rows to area x day-type cells: sum of calls, exposure = sum of day totals.

        With a half-life, each summer is weighted 0.5 ** (years before the latest training summer / half_life):
        a weighted Poisson likelihood, i.e. weighted calls and weighted exposure per cell."""
        w = self.year_weights(P)
        return (P.assign(yw=P.y * w, ew=P.mapped_total * w).groupby(CELL, as_index=False)
                .agg(y=("yw", "sum"), exposure=("ew", "sum")))

    def year_weights(self, P: pd.DataFrame) -> np.ndarray:
        if not self.half_life:
            return np.ones(len(P))
        yr = pd.to_datetime(P.date).dt.year.to_numpy()
        return 0.5 ** ((yr.max() - yr) / self.half_life)

    def fit(self, P: pd.DataFrame) -> "ContextShare":
        P = P[P.mapped_total > 0]
        T = area_traits(P)
        self.areas = T.index.to_numpy()
        self.index = {a: i for i, a in enumerate(self.areas)}
        self.mu, self.sd = T.mean().to_numpy(), T.std().to_numpy() + 1e-9
        self.Z = (T.to_numpy() - self.mu) / self.sd
        self.Z[:, TRAITS.index("beach")] = T.beach.to_numpy()  # keep the beach flag 0/1
        C = self.cells(P)
        X, pen = self._design(C)
        hist = P.assign(yw=P.y * self.year_weights(P)).groupby("modzcta").yw.sum().reindex(self.areas).fillna(0) + 0.5
        b0 = np.zeros(X.shape[1])
        b0[:len(self.areas)] = np.log(hist / hist.sum())
        self.b = fit_glm(X, C.y.to_numpy(float), 0.0, pen, offset=np.log(C.exposure.to_numpy()), b0=b0)
        self.usual = (hist / hist.sum()).to_numpy()
        return self

    def shares(self, P: pd.DataFrame) -> np.ndarray:
        """pi_{i,d} for panel rows P (every area for each date; day context added if missing)."""
        if "event" not in P:
            P = day_context(P)
        X, _ = self._design(P)
        eta = np.clip(X @ self.b, -30, 30)
        raw = np.exp(eta - eta.max())
        return raw / pd.Series(raw, index=P.index).groupby(pd.to_datetime(P.date).to_numpy()).transform("sum").to_numpy()

    def coefficients(self) -> pd.DataFrame:
        """Shared trait x context and event effects as multiplicative changes of an area's share."""
        K = len(self.areas)
        names = ([f"leisure x {t}" for t in LEISURE_TRAITS]
                 + ["beach season x beach", "... x leisure", "... x heat", "... x leisure x heat"]
                 + [f"event {e} on route" for e in EVENTS])
        start = K + 7 + len(EVENTS)
        return pd.DataFrame({"term": names, "share_multiplier": np.exp(self.b[start:start + len(names)])})

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path) -> "ContextShare":
        return pickle.load(open(path, "rb"))


# --- tuning / evaluation ----------------------------------------------------------------------------------------------
def loglik_per_call(P: pd.DataFrame, share: np.ndarray) -> float:
    m = (P.mapped_total > 0).to_numpy()
    y = P.y.to_numpy()[m]
    return float((y * np.log(np.maximum(share[m], 1e-12))).sum() / max(y.sum(), 1))


def fit_tuned(train: pd.DataFrame, half_life: float | None = HALF_LIFE_YEARS) -> ContextShare:
    """Choose the ridge strengths on the last training summer, then refit on all training summers."""
    last = train.year.max()
    inner_tr, inner_va = train[train.year < last], train[train.year == last]
    best = max(RIDGE_GRID, key=lambda r: loglik_per_call(inner_va, ContextShare(r, half_life).fit(inner_tr).shares(inner_va)))
    return ContextShare(best, half_life).fit(train)


def segments(P: pd.DataFrame) -> dict[str, np.ndarray]:
    lei = P.leisure.to_numpy() == 1
    return {"all days": np.ones(len(P), bool), "leisure days": lei,
            "hot beach days (open, leisure, >=34C)": lei & (P.beach_open.to_numpy() == 1) & (P.band.to_numpy() >= 2),
            "extreme heat (>=38C)": P.band.to_numpy() == 3, "event days": (P.event != "none").to_numpy(),
            "ordinary weekdays": ~lei & (P.event == "none").to_numpy()}


def movement(P: pd.DataFrame, share: np.ndarray, usual: np.ndarray, top: int = 15) -> dict:
    """How much the forecast map moves: mean |log(pi_d / pi_usual)| (call-weighted) and top-k turnover."""
    t = P.assign(share=share, usual=usual)
    lr = np.abs(np.log(t.share / t.usual))
    usual_top = set(t.drop_duplicates("modzcta").nlargest(top, "usual").modzcta)
    turnover, distinct = [], set()
    for _, g in t.groupby("date"):
        tp = set(g.nlargest(top, "share").modzcta)
        distinct |= tp
        turnover.append(len(tp - usual_top) / top)
    return {"mean_abs_log_share_change": float(np.average(lr, weights=t.usual)), f"top{top}_turnover": float(np.mean(turnover)),
            f"distinct_areas_in_top{top}": len(distinct)}


def ratio_error(te: pd.DataFrame, shares: dict[str, np.ndarray], level: str = "modzcta", min_calls: int = 10,
                n_sim: int = 20, seed: int = 0) -> dict:
    """Mean daily error on call *ratios*: 1/2 sum_i |forecast share_i - observed share_i| (0 = same map, 1 = disjoint),
    over days with >= min_calls mapped calls, with areas aggregated to ``level`` (modzcta or borough).

    Also returns the noise floor: the error a perfect model would still get, i.e. when the observed calls are a
    multinomial draw from the first model's shares. With few calls per day, observed ratios are mostly chance.
    """
    rng = np.random.default_rng(seed)
    t = te[te.mapped_total >= min_calls]
    Y = t.pivot(index="date", columns="modzcta", values="y")
    U = t.drop_duplicates("modzcta").assign(area=lambda d: d.modzcta).set_index("modzcta")
    group = U["area" if level == "modzcta" else level].reindex(Y.columns).to_numpy()
    agg = lambda M: pd.DataFrame(M, columns=Y.columns).T.groupby(group).sum().T.to_numpy()
    obs = agg(Y.to_numpy())
    obs = obs / obs.sum(1, keepdims=True)
    out, first = {}, None
    for name, sh in shares.items():
        S = t.assign(s=sh[(te.mapped_total >= min_calls).to_numpy()]).pivot(index="date", columns="modzcta", values="s").to_numpy()
        S = S / S.sum(1, keepdims=True)
        first = S if first is None else first
        out[name] = float((0.5 * np.abs(agg(S) - obs).sum(1)).mean())
    n = Y.sum(1).astype(int).to_numpy()
    sims = []
    for _ in range(n_sim):
        D = agg(np.vstack([rng.multinomial(k, p) for k, p in zip(n, first)]))
        sims.append((0.5 * np.abs(agg(first) - D / D.sum(1, keepdims=True)).sum(1)).mean())
    out["noise floor (perfect model)"] = float(np.mean(sims))
    out["days"] = len(n)
    return out


def evaluate(P: pd.DataFrame, seasons=TEST_SEASONS):
    rows, calib, ratio = [], [], []
    for s in seasons:
        tr, te = P[P.year < s], P[P.year == s]
        m = fit_tuned(tr)
        hist = tr.groupby("modzcta").y.sum() + 0.5
        fixed = te.modzcta.map(hist / hist.sum()).to_numpy()
        fixed = fixed / pd.Series(fixed, index=te.index).groupby(te.date.to_numpy()).transform("sum").to_numpy()
        ctx = m.shares(te)
        for seg, mask in segments(te).items():
            sub = te[mask]
            calls = int(sub.y.sum())
            if calls == 0:
                continue
            for name, sh in (("historical share", fixed[mask]), ("context share", ctx[mask])):
                rows.append(dict(season=s, segment=seg, method=name, calls=calls, loglik_per_call=loglik_per_call(sub, sh)))
        rows.append(dict(season=s, segment="movement", method="context share", ridge=m.ridge_leisure,
                         **movement(te, ctx, fixed)))
        calib.append(te.assign(season=s, pred_ratio=ctx / fixed, expected_fixed=te.mapped_total * fixed)[
            ["season", "date", "modzcta", "y", "pred_ratio", "expected_fixed", "mapped_total"]])
        for level in ("modzcta", "borough"):
            ratio.append(dict(season=s, level=level, **ratio_error(te, {"context share": ctx, "historical share": fixed}, level)))
        print(f"[context-share] season {s} done (ridge {m.ridge_leisure})", flush=True)
    E = pd.DataFrame(rows)
    C = pd.concat(calib)
    C = C[C.mapped_total > 0]
    C["bin"] = pd.cut(np.log(C.pred_ratio), [-9, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 9],
                      labels=["< -40%", "-40..-22%", "-22..-10%", "about usual", "+10..+28%", "+28..+65%", "> +65%"])
    cal = C.groupby("bin", observed=True).apply(lambda g: pd.Series({
        "area_days": len(g), "calls": g.y.sum(), "predicted_ratio": np.average(g.pred_ratio, weights=g.expected_fixed),
        "realised_ratio": g.y.sum() / g.expected_fixed.sum()}), include_groups=False)
    return E, cal, pd.DataFrame(ratio)


def prepare(P: pd.DataFrame) -> pd.DataFrame:
    """Add the day's mapped call total (used for fitting and scoring)."""
    P = P.copy()
    P["mapped_total"] = P.groupby("date").y.transform("sum")
    return P


def load(paths: Paths = PATHS) -> pd.DataFrame:
    P = pd.read_parquet(paths.processed / "panel.parquet")
    P["mapped_total"] = P.groupby("date").y.transform("sum")
    return day_context(P)


def run(paths: Paths = PATHS, evaluate_seasons: bool = True) -> ContextShare:
    out = paths.out("models/share")
    P = load(paths)
    if evaluate_seasons:
        E, cal, R = evaluate(P)
        E.round(4).to_csv(out / "scores_by_season.csv", index=False)
        S = (E[E.segment != "movement"].groupby(["segment", "method"], sort=False)
             .apply(lambda g: pd.Series({"calls": g.calls.sum(), "loglik_per_call": np.average(g.loglik_per_call, weights=g.calls)}),
                    include_groups=False).unstack("method"))
        S.columns = [f"{a} | {b}" for a, b in S.columns]
        S["gain_nats_per_call"] = S["loglik_per_call | context share"] - S["loglik_per_call | historical share"]
        S = S.drop(columns=["calls | context share"]).rename(columns={"calls | historical share": "calls"})
        S.round(4).to_csv(out / "scores_summary.csv")
        cal.round(3).to_csv(out / "move_calibration.csv")
        R.round(4).to_csv(out / "ratio_error_by_season.csv", index=False)
        Rs = R.groupby("level").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.days) for c in
                                                         ["context share", "historical share", "noise floor (perfect model)"]}),
                                      include_groups=False)
        Rs.round(4).to_csv(out / "ratio_error_summary.csv")
        print("daily ratio error 1/2 sum|forecast share - observed share| (days with >= 10 calls):", Rs.round(3).to_string(), sep="\n")
        mv = E[E.segment == "movement"].drop(columns=["segment", "method", "calls", "loglik_per_call"], errors="ignore")
        mv.round(3).to_csv(out / "movement.csv", index=False)
        print(S.round(4).to_string(), cal.round(3).to_string(), mv.round(3).to_string(index=False), sep="\n\n")
    m = fit_tuned(P)
    m.save(out / "share_model.pkl")
    m.coefficients().round(3).to_csv(out / "coefficients.csv", index=False)
    print(m.coefficients().round(3).to_string(index=False))
    return m


if __name__ == "__main__":
    from heattents.models.context_share import run as _run
    _run()
