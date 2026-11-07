"""Where does a season's forecast error accumulate? (out-of-sample predictions from the city model evaluation)

cumulative_error(season)   running sum of (forecast - actual) per day, for the ensemble, GLM and LightGBM, plus the
                           daily actual vs ensemble forecast; figure + CSV in outputs/models/city/diagnostics/
Run: python -m heattents.models.diagnostics 2024
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..config import PATHS, Paths

SERIES = {"ens": ("Ensemble (used)", "#2a78d6"), "glm": ("GLM", "#eb6834"), "gbm": ("LightGBM", "#1baf7a")}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def cumulative_error(season: int, paths: Paths = PATHS) -> pd.DataFrame:
    o = pd.read_csv(paths.out("models/city") / "oos_predictions.csv", parse_dates=["date"])
    o = o[o.season == season]
    w = o.pivot_table(index=["date", "y"], columns="model", values="mean").reset_index().sort_values("date")
    for k in SERIES:
        w[f"cum_err_{k}"] = (w[k] - w.y).cumsum()
    return w


def plot(w: pd.DataFrame, season: int, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 6.4), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    ends = []
    for k, (label, color) in SERIES.items():
        y = w[f"cum_err_{k}"]
        ax.plot(w.date, y, color=color, linewidth=2.4 if k == "ens" else 1.6, label=label)
        ends.append((y.iloc[-1], label, color))
    # direct labels at the line ends, nudged apart so they never collide
    ends.sort()
    placed = []
    for v, label, color in ends:
        pos = v if not placed or v - placed[-1] > 18 else placed[-1] + 18
        placed.append(pos)
        ax.annotate(f"{label}: {v:+.0f}", (w.date.iloc[-1], v), xytext=(w.date.iloc[-1] + pd.Timedelta(days=3), pos),
                    fontsize=8, color=INK2, va="center")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("Cumulative forecast − actual\n(calls)", color=INK2, fontsize=8.5)
    ax.set_title(f"{season}: where the season's forecast error builds up", loc="left", color=INK, fontsize=11, pad=10)
    bx.bar(w.date, w.y, width=0.8, color="#c3c2b7", label="Actual calls")
    bx.plot(w.date, w["ens"], color=SERIES["ens"][1], linewidth=1.6, label="Ensemble forecast")
    bx.set_ylabel("Calls per day", color=INK2, fontsize=8.5)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(axis="y", color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(AXIS)
        a.tick_params(colors=MUTED, labelsize=8, length=0)
        leg = a.legend(loc="upper left", frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(INK2)
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    bx.set_xlim(w.date.iloc[0] - pd.Timedelta(days=2), w.date.iloc[-1] + pd.Timedelta(days=22))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run(season: int = 2024, paths: Paths = PATHS) -> pd.DataFrame:
    out = paths.out("models/city/diagnostics")
    w = cumulative_error(season, paths)
    w.round(2).to_csv(out / f"cumulative_error_{season}.csv", index=False)
    plot(w, season, out / f"cumulative_error_{season}.png")
    return w


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
