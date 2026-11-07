"""Exploratory analysis of the NYC label (EMS HEAT calls) and the secondary label (syndromic heat ED visits).

Run:  python -m heattents.eda      -> CSVs in outputs/eda/ (findings are summarised in outputs/eda/findings.md)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PATHS, Paths
from .ingest import context, ems, geography


def us_holidays(years) -> set:
    """Federal holidays observed in summer + Memorial/Labor Day (rule-based, no dependency)."""
    out = set()
    for y in years:
        may = pd.date_range(f"{y}-05-01", f"{y}-05-31")
        out.add(may[may.dayofweek == 0][-1])                                 # Memorial Day
        sep = pd.date_range(f"{y}-09-01", f"{y}-09-30")
        out.add(sep[sep.dayofweek == 0][0])                                  # Labor Day
        for md in ("06-19", "07-04"):                                        # Juneteenth (from 2021), Independence Day
            if md == "06-19" and y < 2021:
                continue
            d = pd.Timestamp(f"{y}-{md}")
            out.add(d - pd.Timedelta(days=1) if d.dayofweek == 5 else d + pd.Timedelta(days=1) if d.dayofweek == 6 else d)
            out.add(d)
    return out


def spearman(a, b) -> float:
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def run(paths: Paths = PATHS) -> None:
    out = paths.out("eda")
    e = ems.fetch(paths)
    e["date"] = e.incident_datetime.dt.normalize()
    e["year"], e["month"], e["hour"] = e.date.dt.year, e.date.dt.month, e.incident_datetime.dt.hour
    s = context.syndromic(paths)
    Z = geography.load_modzcta(paths)
    zmap = geography.zip_to_modzcta(Z)
    e["zip5"] = e.zipcode.astype("string").str[:5]
    e["modzcta"] = e.zip5.map(zmap)
    temp = s.set_index("date").lga_max_temp_f

    # (1) yearly / monthly, coding mix
    yr = e.groupby("year").agg(ems_heat=("incident_id", "size"),
                               initially_heat=("initial_call_type", lambda x: (x == "HEAT").mean()),
                               held=("held_indicator", lambda x: (x == "Y").mean()),
                               special_event=("special_event_indicator", lambda x: (x == "Y").mean()),
                               zip_missing=("zipcode", lambda x: x.isna().mean()))
    ssum = s[s.date.dt.month.between(5, 9)]
    yr = yr.join(ssum.groupby(ssum.date.dt.year).agg(syndromic_ed=("heat_ed_visits", "sum"),
                                                    days_ge90f=("lga_max_temp_f", lambda x: (x >= 90).sum()),
                                                    days_ge95f=("lga_max_temp_f", lambda x: (x >= 95).sum()),
                                                    degree_days_above_85f=("lga_max_temp_f", lambda x: np.maximum(x - 85, 0).sum())))
    yr["ems_per_ed"] = yr.ems_heat / yr.syndromic_ed
    yr["ems_per_degree_day"] = yr.ems_heat / yr.degree_days_above_85f
    yr["ed_per_degree_day"] = yr.syndromic_ed / yr.degree_days_above_85f
    yr.round(4).to_csv(out / "yearly.csv")
    e.pivot_table(index="year", columns="month", values="incident_id", aggfunc="size", fill_value=0).to_csv(out / "year_month_counts.csv")
    disp = e.pivot_table(index="year", columns="incident_disposition_code", values="incident_id", aggfunc="size", fill_value=0)
    (disp.div(disp.sum(1), axis=0)).round(3).to_csv(out / "disposition_share_by_year.csv")
    init = e[e.year >= 2015].pivot_table(index="year", columns="initial_call_type", values="incident_id", aggfunc="size", fill_value=0)
    init = init[init.sum().sort_values(ascending=False).index[:8]]
    init.div(init.sum(1), axis=0).round(3).to_csv(out / "initial_call_type_share_by_year.csv")

    # temperature response by year: EMS/ED counts on days with LGA tmax >= 95F and in 90-94F
    daily = e.groupby("date").agg(ems=("incident_id", "size"), special_event=("special_event_indicator", lambda x: (x == "Y").mean()))
    days = pd.DataFrame({"lga_max_temp_f": temp, "heat_ed_visits": s.set_index("date").heat_ed_visits})
    days = days.join(daily, how="left").fillna({"ems": 0})
    days = days[days.index.month.isin(range(5, 10))]
    days["band"] = pd.cut(days.lga_max_temp_f, [0, 84, 89, 94, 200], labels=["<85", "85-89", "90-94", ">=95"])
    resp = days.groupby([days.index.year, "band"], observed=True)[["ems", "heat_ed_visits"]].mean().round(2).unstack()
    resp.to_csv(out / "mean_daily_counts_by_temp_band_and_year.csv")

    # (2) top days
    hol = us_holidays(range(2005, 2027))
    top = daily.sort_values("ems", ascending=False).head(20).copy()
    top["lga_max_temp_f"] = temp.reindex(top.index)
    top["heat_ed_visits"] = s.set_index("date").heat_ed_visits.reindex(top.index)
    top["weekday"] = top.index.day_name()
    top["holiday"] = top.index.isin(hol)
    top.to_csv(out / "top20_days.csv")
    jul = days.loc["2026-06-25":"2026-07-10"]
    jul.to_csv(out / "july_2026_window.csv")

    # (3) EMS vs syndromic, 2017-2026 May-Sep
    d17 = days[days.index.year >= 2017].dropna(subset=["heat_ed_visits"])
    corr = []
    for y, g in d17.groupby(d17.index.year):
        corr.append(dict(year=y, corr_lag0=g.ems.corr(g.heat_ed_visits), corr_ems_lead_ed_lag1=g.ems.corr(g.heat_ed_visits.shift(-1)),
                         corr_log1p=np.log1p(g.ems).corr(np.log1p(g.heat_ed_visits)), ratio=g.ems.sum() / g.heat_ed_visits.sum()))
    corr.append(dict(year="all", corr_lag0=d17.ems.corr(d17.heat_ed_visits), corr_ems_lead_ed_lag1=d17.ems.corr(d17.heat_ed_visits.shift(-1)),
                     corr_log1p=np.log1p(d17.ems).corr(np.log1p(d17.heat_ed_visits)), ratio=d17.ems.sum() / d17.heat_ed_visits.sum()))
    pd.DataFrame(corr).round(3).to_csv(out / "ems_vs_syndromic.csv", index=False)

    # (4) spatial
    pop = Z.set_index("modzcta").pop_est
    sp = e.dropna(subset=["modzcta"]).groupby("modzcta").agg(calls=("incident_id", "size"),
                                                              calls_2005_2015=("year", lambda x: (x <= 2015).sum()),
                                                              calls_2016_2026=("year", lambda x: (x >= 2016).sum()))
    sp = sp.reindex(pop.index, fill_value=0)
    sp["pop_est"] = pop
    sp["calls_per_1k_pop"] = sp.calls / sp.pop_est * 1000
    sp["borough"] = e.dropna(subset=["modzcta"]).groupby("modzcta").borough.agg(lambda x: x.mode().iat[0]).reindex(sp.index)
    h = context.hvi(paths).set_index("zcta").hvi
    sp["hvi"] = h.reindex(sp.index)
    sp.sort_values("calls", ascending=False).to_csv(out / "modzcta_counts.csv")
    unm = e[e.modzcta.isna()].zip5.fillna("<missing>").value_counts()
    unm.to_csv(out / "unmapped_zips.csv")

    # (5) hourly profiles
    e["lga"] = e.date.map(temp)
    e["day_type"] = np.where(e.lga >= 95, "hot>=95F", np.where(e.lga < 90, "mild<90F", "90-94F"))
    hp = e.pivot_table(index="hour", columns="day_type", values="incident_id", aggfunc="size", fill_value=0)
    hp = hp.div(hp.sum(), axis=1).round(4)
    hp["all"] = e.groupby("hour").size() / len(e)
    hp.to_csv(out / "hourly_profile.csv")
    hb = e.pivot_table(index="hour", columns="borough", values="incident_id", aggfunc="size", fill_value=0)
    hb.div(hb.sum(), axis=0 if False else 1).round(4).to_csv(out / "hourly_profile_by_borough.csv")

    # (6) sparsity: MODZCTA x day, May-Sep, all years
    dates = pd.date_range("2005-05-01", "2026-09-30")
    dates = dates[dates.month.isin(range(5, 10))]
    cnt = e.dropna(subset=["modzcta"]).groupby(["modzcta", "date"]).size()
    n_cells = len(pop) * len(dates)
    nz = cnt[cnt.index.get_level_values("date").isin(dates)].to_numpy()
    vals = np.zeros(n_cells)
    vals[:len(nz)] = nz  # non-zero cells; the rest of the grid is zero
    dist = pd.Series(vals).value_counts().sort_index()
    m, v = vals.mean(), vals.var()
    pd.DataFrame({"calls": dist.index, "cells": dist.values, "share": dist.values / n_cells}).to_csv(out / "modzcta_day_distribution.csv", index=False)

    # (7) duplicates: same MODZCTA within 30 min
    ee = e.dropna(subset=["modzcta"]).sort_values(["modzcta", "incident_datetime"])
    gap = ee.groupby("modzcta").incident_datetime.diff().dt.total_seconds()
    dup30 = (gap <= 1800).mean()
    dup5 = (gap <= 300).mean()

    # --- print summary
    pd.set_option("display.width", 250)
    print(yr.round(3).to_string())
    print(resp.to_string())
    print(top.to_string())
    print(jul.to_string())
    print(pd.DataFrame(corr).round(3).to_string())
    conc = sp.calls.sort_values(ascending=False)
    rate = sp[sp.pop_est > 1000].calls_per_1k_pop
    print(f"MODZCTA: {len(sp)}, top10% share of calls {conc.head(int(len(conc)*.1)).sum()/conc.sum():.3f}, "
          f"rank stability 2005-15 vs 2016-26 (counts) {spearman(sp.calls_2005_2015, sp.calls_2016_2026):.3f}, "
          f"per-capita rate vs HVI spearman {spearman(rate, sp.hvi.reindex(rate.index).fillna(rate.median())):.3f}, "
          f"counts vs pop spearman {spearman(sp.calls, sp.pop_est):.3f}")
    print(sp.sort_values("calls", ascending=False).head(10).to_string())
    print(sp.groupby("hvi").calls_per_1k_pop.median())
    print("unmapped:", int(e.modzcta.isna().sum()), unm.head(10).to_dict())
    print(hp.loc[[6, 9, 12, 14, 16, 18, 21]].to_string())
    print(f"MODZCTA-day May-Sep cells {n_cells}, zero share {(vals == 0).mean():.4f}, mean {m:.4f}, var {v:.4f}, "
          f"var/mean {v/m:.2f}, max {vals.max():.0f}, P(>=2) {(vals >= 2).mean():.5f}")
    print(f"same-MODZCTA gaps <=30min {dup30:.3f}, <=5min {dup5:.3f}")
    print((disp.div(disp.sum(1), axis=0)).round(3).loc[[2015, 2019, 2022, 2024, 2025, 2026]].to_string())
    print(init.div(init.sum(1), axis=0).round(3).to_string())


if __name__ == "__main__":
    run()
