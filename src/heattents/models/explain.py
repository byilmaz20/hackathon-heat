"""Human-readable views of the citywide model (for the README, the app and the jury).

heat_curve     expected calls tomorrow vs tomorrow's max heat index (deg F), first hot day vs 4th consecutive hot
               day, with the 90% predictive interval (NB count noise; weather taken as known)
effects        for a typical hot July weekday, the % change in expected calls when one input is switched on/off,
               computed with the full forecast model (GLM + LightGBM ensemble), not from raw GLM coefficients
All other inputs are held at the same typical values (e.g. yesterday's calls before 18:00), so each line or row
isolates one factor. These are the model's associations, not causal effects.
Outputs: outputs/models/city/heat_curve.csv, heat_curve.png, effects.csv
Run: python -m heattents.models.explain
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS, Paths
from .city import load_city_model

# typical hot July weekday (deg C inside the model)
BASE = dict(hot_days_before=1, season_hot_days_before=8, weekend=0, holiday=0, july4_window=0, school_break=1,
            covid=0, year=2026, day_of_year=196, up_sw_gradient=0.0, rh=60.0, calls_by18=20.0, tmean_30d=24.5)
F_TO_C = lambda f: (np.asarray(f, float) - 32) * 5 / 9
COLORS = {"first": "#2a78d6", "fourth": "#eb6834"}  # categorical slots 1-2, validated (CVD dE 24.7)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _predict(model, **kw) -> pd.DataFrame:
    f = model.scenario(**{**BASE, **kw})
    return model.predict(f)


def heat_curve(model, temps_f=np.arange(75, 111, 1)) -> pd.DataFrame:
    rows = []
    for label, hot_before in (("first hot day", 0), ("4th consecutive hot day", 3)):
        for t in temps_f:
            a = float(F_TO_C(t))
            # earlier days of the spell at the same heat index; for the first hot day the previous days were mild (85 F)
            prev = a if hot_before else float(F_TO_C(85))
            p = _predict(model, atmax=a, atmax_l1=prev, atmax_l2=prev, atmax_l3=prev, hot_days_before=hot_before,
                         tmean_l1=prev - 4, tmean_l2=prev - 4).iloc[0]
            rows.append(dict(case=label, heat_index_f=int(t), expected_calls=p["mean"], q05=p.q05, q95=p.q95))
    return pd.DataFrame(rows)


def effects(model, heat_index_f: float = 95.0) -> pd.DataFrame:
    a = float(F_TO_C(heat_index_f))
    base_kw = dict(atmax=a, atmax_l1=a - 1, atmax_l2=a - 2, atmax_l3=a - 3)
    base = _predict(model, **base_kw)["mean"].iloc[0]
    cases = [
        ("Weekend instead of weekday", dict(weekend=1)),
        ("Federal holiday", dict(holiday=1)),
        ("4th instead of 2nd hot day in a row", dict(hot_days_before=3, atmax_l1=a, atmax_l2=a, atmax_l3=a)),
        ("Heat index +5 F (same spell)", dict(atmax=a + 25 / 9)),
        ("Warm night (min >= 75 F)", dict(tmin=24.5)),
        ("South-west 9 F hotter than NYC yesterday", dict(up_sw_gradient=5.0)),
        ("Twice as many calls yesterday before 18:00", dict(calls_by18=40.0)),
        ("Early in the season (1 June instead of mid-July)", dict(day_of_year=152, season_hot_days_before=1)),
        ("Humid (80% instead of 60% relative humidity)", dict(rh=80.0)),
        ("COVID years (2020-21)", dict(covid=1)),
    ]
    rows = [dict(factor="Baseline: weekday, 2nd hot day, 95 F heat index", expected_calls=round(base, 1), change_pct=0.0)]
    for name, kw in cases:
        v = _predict(model, **{**base_kw, **kw})["mean"].iloc[0]
        rows.append(dict(factor=name, expected_calls=round(v, 1), change_pct=round(100 * (v / base - 1), 1)))
    return pd.DataFrame(rows)


def plot_heat_curve(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for (label, g), color in zip(df.groupby("case", sort=False), COLORS.values()):
        ax.fill_between(g.heat_index_f, g.q05, g.q95, color=color, alpha=0.12, linewidth=0)
        ax.plot(g.heat_index_f, g.expected_calls, color=color, linewidth=2, label=label, solid_capstyle="round")
    ax.set_xlim(75, 110)
    ax.set_ylim(0, None)
    ax.set_xlabel("Tomorrow's max heat index (°F)", color=INK2, fontsize=9)
    ax.set_ylabel("Expected heat EMS calls in NYC", color=INK2, fontsize=9)
    ax.set_title("Expected heat EMS calls climb steeply above a 90°F heat index", loc="left", color=INK,
                 fontsize=11, pad=12)
    ax.text(0, 1.01, "Model forecast for a July weekday, yesterday's calls held at a typical 20; shaded: 90% prediction interval.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run(paths: Paths = PATHS):
    out = paths.out("models/city")
    m = load_city_model(paths)
    curve = heat_curve(m)
    curve.round(2).to_csv(out / "heat_curve.csv", index=False)
    plot_heat_curve(curve, out / "heat_curve.png")
    eff = effects(m)
    eff.to_csv(out / "effects.csv", index=False)
    print(curve[curve.heat_index_f.isin([85, 90, 95, 100, 105, 110])].round(1).to_string(index=False))
    print(eff.to_string(index=False))


if __name__ == "__main__":
    run()
