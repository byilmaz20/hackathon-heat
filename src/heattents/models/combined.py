r"""Next-day forecast of heat EMS calls per neighbourhood, and its accuracy.

FORECAST for neighbourhood i on day d (issued 18:00 on d-1, out of sample: every model trained on earlier summers)
    N_d          citywide total: predictive distribution of the city model (models/city.py)
    pi_i         neighbourhood i's historical share of all earlier calls (models/neighbourhood.py)
    m            share of calls with a known location (earlier summers)
    y_{i,d}      ~ Binomial(N_d, m * pi_i)  (sampled jointly with N_d);   expected value  E[N_d] * m * pi_i
BASELINE "the neighbourhood's past average": mean daily calls of i over all earlier summers, every day the same
         (Poisson around it for the probabilistic score).
METRICS over all neighbourhood-days (42 neighbourhoods x 450 days), and over busy days (>= 30 calls citywide)
    MAE    mean absolute error of the expected value, calls per neighbourhood per day
    WAPE   sum |forecast - actual| / sum actual: share of the city's calls the forecast gets wrong
    CRPS   continuous ranked probability score of the full forecast distribution (calls; sample estimate)
    skill  1 - score(model) / score(baseline)
Outputs: outputs/models/neighbourhood/forecast_daily.csv, forecast_accuracy.csv, figures
Run: python -m heattents.models.combined
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .city import Context, decision_features, fit_city_model, observed_features
from .neighbourhood import historical_share

TEST_SEASONS = (2024, 2025, 2026)
BUSY = 30
N_SAMPLES = 1000


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS = E|X - y| - 0.5 E|X - X'| from samples (n_cells, S); uses the sorted-sample identity for E|X - X'|."""
    S = samples.shape[1]
    term1 = np.abs(samples - y[:, None]).mean(1)
    xs = np.sort(samples, axis=1)
    k = np.arange(1, S + 1)
    term2 = (2 * ((2 * k - S - 1) * xs).sum(1)) / (S * S)
    return term1 - 0.5 * term2


def forecast(paths: Paths = PATHS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctx = Context.load(paths)
    C = ctx.city
    F = observed_features(ctx, C)
    y_city = C.city_y.to_numpy()
    P = pd.read_parquet(paths.processed / "neighbourhood_panel.parquet")
    P["uhf"] = P.uhf.astype(str)
    out = []
    for s in TEST_SEASONS:
        tr = (C.year < s).to_numpy()
        model = fit_city_model(F[tr], y_city[tr], t95=ctx.t95)
        rows = C[(C.year == s).to_numpy()]
        ft, _ = decision_features(ctx, rows, rng)
        dates, mean_city, Pm = model.pmf(ft, "ens")
        Ptr = P[P.year < s]
        share = historical_share(Ptr)
        m = Ptr.y.sum() / C[tr].city_y.sum()  # calls with a known neighbourhood / all calls
        past_avg = Ptr.groupby("uhf").y.mean()
        uhfs = share.index.to_numpy()
        p = (m * share).to_numpy()
        cdf = np.cumsum(Pm, 1)
        actual = P[P.year == s].pivot_table(index="date", columns="uhf", values="y").reindex(index=dates, columns=uhfs)
        for di, d in enumerate(dates):
            N = np.searchsorted(cdf[di], rng.random(N_SAMPLES)).clip(0, cdf.shape[1] - 1)
            Y = rng.binomial(N[None, :], p[:, None])  # (neighbourhoods, samples)
            yb = rng.poisson(past_avg.reindex(uhfs).to_numpy()[:, None], (len(uhfs), N_SAMPLES))
            ya = actual.loc[d].to_numpy()
            out.append(pd.DataFrame({
                "season": s, "date": d, "uhf": uhfs, "actual": ya,
                "forecast": mean_city[di] * p, "q05": np.quantile(Y, .05, axis=1), "q95": np.quantile(Y, .95, axis=1),
                "crps": crps_samples(Y, ya), "baseline": past_avg.reindex(uhfs).to_numpy(),
                "crps_baseline": crps_samples(yb, ya), "city_actual": y_city[(C.date == d).to_numpy()][0]}))
        print(f"[combined] season {s} done", flush=True)
    return pd.concat(out, ignore_index=True)


def accuracy(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in (("all days", D), ("busy days (>= 30 calls citywide)", D[D.city_actual >= BUSY])):
        for method, col, crps in (("forecast", "forecast", "crps"), ("past average", "baseline", "crps_baseline")):
            err = sub[col] - sub.actual
            rows.append(dict(days=name, method=method, n_days=sub.date.nunique(), mae=err.abs().mean(),
                             rmse=np.sqrt((err ** 2).mean()), wape=err.abs().sum() / sub.actual.sum(),
                             crps=sub[crps].mean(), coverage_90=((sub.actual >= sub.q05) & (sub.actual <= sub.q95)).mean()
                             if method == "forecast" else np.nan))
    A = pd.DataFrame(rows)
    for metric in ("mae", "wape", "crps"):
        base = A[A.method == "past average"].set_index("days")[metric]
        A[f"{metric}_skill_vs_past_average"] = [1 - v / base[d] if m == "forecast" else np.nan
                                                for d, m, v in zip(A.days, A.method, A[metric])]
    return A


def run(paths: Paths = PATHS) -> pd.DataFrame:
    out = paths.out("models/neighbourhood")
    D = forecast(paths)
    D.round(4).to_csv(out / "forecast_daily.csv", index=False)
    A = accuracy(D)
    A.round(4).to_csv(out / "forecast_accuracy.csv", index=False)
    print(A.round(3).to_string(index=False))
    return D


if __name__ == "__main__":
    from heattents.models.combined import run as _run
    _run()


# ---------------------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated, CVD dE 24.7)


def neighbourhood_names(paths: Paths = PATHS) -> dict:
    import json
    t = json.loads((paths.raw / "ehdp_UHF42.topo.json").read_text())
    return {str(g["properties"]["GEOCODE"]): g["properties"]["GEONAME"] for o in t["objects"].values() for g in o["geometries"]}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_axisbelow(True)


def plot_totals(D: pd.DataFrame, path, paths: Paths = PATHS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = neighbourhood_names(paths)
    T = D.groupby("uhf")[["actual", "forecast", "baseline"]].sum().sort_values("actual")
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(T))
    ax.hlines(y, T[["actual", "forecast"]].min(1), T[["actual", "forecast"]].max(1), color=GRID, linewidth=2)
    ax.scatter(T.baseline, y, s=26, color=ORANGE, label="Past average (no forecast)", zorder=3, marker="D")
    ax.scatter(T.forecast, y, s=34, color=BLUE, label="Our next-day forecast", zorder=4)
    ax.scatter(T.actual, y, s=34, facecolor=SURFACE, edgecolor=INK, linewidth=1.4, label="Actual calls", zorder=5)
    ax.set_yticks(y, [names.get(u, u) for u in T.index], fontsize=7.5, color=INK2)
    ax.set_xlabel("Heat EMS calls, May–September 2024–2026 (sum of daily values)", color=INK2, fontsize=9)
    ax.set_title("Actual vs forecast heat calls per neighbourhood, 2024–26", loc="left", color=INK, fontsize=11, pad=22)
    ax.text(0, 1.008, "Forecast issued the evening before, out of sample. 42 NYC health neighbourhoods (UHF42).",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    _style(ax)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_busy_days(D: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = D[D.city_actual >= BUSY]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    hi = max(B.actual.max(), B.forecast.max(), B.baseline.max()) + 1
    for ax, (col, label, color) in zip(axes, (("forecast", "Our next-day forecast", BLUE),
                                               ("baseline", "Past average (no forecast)", ORANGE))):
        jitter = rng.uniform(-0.2, 0.2, len(B))
        ax.plot([0, hi], [0, hi], color=AXIS, linewidth=1, zorder=1)
        ax.scatter(B[col], B.actual + jitter, s=10, color=color, alpha=0.45, linewidth=0, zorder=2)
        mae = (B[col] - B.actual).abs().mean()
        ax.set_title(f"{label}\nmean abs. error {mae:.2f} calls per neighbourhood", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("Forecast calls", color=INK2, fontsize=9)
        ax.grid(color=GRID, linewidth=0.6)
        _style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(AXIS)
    axes[0].set_ylabel("Actual calls (jittered)", color=INK2, fontsize=9)
    fig.suptitle(f"Busy days ({B.date.nunique()} days with ≥ {BUSY} calls citywide): each dot = one neighbourhood on one day; "
                 "diagonal = perfect forecast", x=0.01, ha="left", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def figures(paths: Paths = PATHS) -> None:
    out = paths.out("models/neighbourhood")
    D = pd.read_csv(out / "forecast_daily.csv", dtype={"uhf": str})
    plot_totals(D, out / "totals_by_neighbourhood.png", paths)
    plot_busy_days(D, out / "busy_days_scatter.png")
