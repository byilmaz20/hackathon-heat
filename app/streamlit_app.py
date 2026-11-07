"""NYC heat-illness forecast & cooling-tent deployment - decision-support app.

Run:  .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import data_layer as dl

st.set_page_config(page_title="NYC Heat Response", layout="wide")

SHIFT_BANDS = {"Whole day (0-24)": list(range(24)), "Morning (8-14)": list(range(8, 14)),
               "Afternoon (12-18)": list(range(12, 18)), "Evening (18-24)": list(range(18, 24))}
DEFAULT_SHIFTS = pd.DataFrame({"name": ["morning", "afternoon", "evening"], "start_hour": [8, 12, 16], "end_hour": [14, 18, 22]})
EVENT_NAMES = ["pride", "puerto_rican_day", "dominican_day", "west_indian_day", "mermaid_parade", "us_open"]
VIEW = pdk.ViewState(latitude=40.71, longitude=-73.94, zoom=9.6, pitch=0)


def ramp(v: np.ndarray) -> list[list[int]]:
    """Sequential yellow -> dark red ramp for non-negative values (98th-percentile capped)."""
    stops = np.array([[255, 245, 205], [253, 174, 97], [227, 74, 51], [127, 0, 0]], float)
    hi = np.quantile(v, 0.98) if np.any(v > 0) else 1.0
    t = np.clip(v / (hi or 1.0), 0, 1) * (len(stops) - 1)
    k = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - k)[:, None]
    rgb = stops[k] * (1 - f) + stops[k + 1] * f
    return [[int(a), int(b), int(c), 190] for a, b, c in rgb]


def choropleth(values: pd.DataFrame, col: str, tooltip_cols: list[str]) -> pdk.Layer:
    P = dl.area_polygons().merge(values, on="area_id", how="left").fillna({col: 0})
    P["fill"] = ramp(P[col].to_numpy(float))
    for c in tooltip_cols:
        if c in P and P[c].dtype.kind == "f":
            P[c] = P[c].round(2)
    return pdk.Layer("PolygonLayer", P, get_polygon="polygon", get_fill_color="fill", get_line_color=[90, 90, 90, 80],
                     line_width_min_pixels=0.5, pickable=True, stroked=True)


def sidebar_scenario() -> dl.Scenario:
    st.sidebar.header("Day to forecast")
    mode = st.sidebar.radio("Input", ["Historical date", "Custom scenario"], horizontal=True)
    if mode == "Historical date":
        date = st.sidebar.date_input("Date (May-Sep 2024-2026)", dt.date(2026, 7, 2), min_value=dt.date(2024, 5, 1),
                                     max_value=dt.date(2026, 9, 21))
        st.sidebar.caption("Forecast as issued at 18:00 the evening before, from the archived weather forecast.")
        return dl.Scenario(label=f"{date:%Y-%m-%d} · forecast issued the evening before", date=str(date))
    hi = st.sidebar.slider("Max heat index today (°F)", 75, 115, 102)
    hot = st.sidebar.slider("Hot days (heat index ≥ 90°F) already in this spell", 0, 7, 2)
    wk = st.sidebar.checkbox("Weekend")
    hol = st.sidebar.checkbox("Public holiday")
    event = st.sidebar.selectbox("Permitted event today", ["none", *EVENT_NAMES],
                                 help="Recurring parades / venues with a known route; they shift where calls happen.")
    up = st.sidebar.slider("Upstream signal: yesterday's heat index in Philadelphia/Baltimore/DC minus NYC (°F)", -10, 10, 0,
                           help="Positive = hotter air mass on its way from the south-west.")
    y1 = st.sidebar.number_input("Heat EMS calls yesterday before 18:00 (citywide)", 0, 400, 30)
    return dl.Scenario(label=f"scenario: {hi}°F heat index, day {hot + 1} of spell", heat_index_f=hi,
                       hot_days_before=hot, weekend=wk, holiday=hol, upstream_gradient_f=up, calls_yesterday_by18=float(y1),
                       event=None if event == "none" else event)


@st.cache_data(show_spinner="Running the forecast models...")
def cached_forecast(sc: dl.Scenario):
    return dl.get_forecast(sc)


def forecast_page(sc: dl.Scenario, fc):
    st.subheader(f"Heat-illness forecast · {sc.label}")
    c = fc.city
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected heat EMS calls (city)", f"{c['mean']:.0f}")
    m2.metric("90% prediction interval", f"{c['q05']:.0f} – {c['q95']:.0f}")
    m3.metric("P(≥ 100 calls)", f"{100 * c['p_ge100']:.0f}%")
    m4.metric("Observed that day", "–" if fc.observed_total is None else f"{fc.observed_total:.0f}")
    c1, c2 = st.columns([1, 1])
    band = c1.selectbox("Hours", list(SHIFT_BANDS), index=0)
    k = c2.slider("Risk threshold k (cases in the band, per ZIP area)", 1, 5, 1)
    per_area = dl.summarise(fc.demand, SHIFT_BANDS[band], k)
    metric = st.radio("Map", ["Expected cases", f"P(cases ≥ {k})", "Upper 95% bound"], horizontal=True)
    col = {"Expected cases": "expected", f"P(cases ≥ {k})": f"p_ge_{k}", "Upper 95% bound": "q95"}[metric]
    layer = choropleth(per_area, col, ["expected", "q05", "q95", f"p_ge_{k}"])
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=VIEW, map_style="light",
                             tooltip={"text": "ZIP area {area_id}\nexpected {expected}\n90% band {q05}–{q95}\n"
                                              f"P(≥{k}) {{p_ge_{k}}}"}), width="stretch")
    hourly = fc.demand.groupby(["hour", "scenario"]).expected_cases.sum().groupby("hour").agg(
        expected="mean", low="min", high="max")
    st.caption("City-wide expected cases by hour (mean, and the lowest / highest of the representative forecast scenarios)")
    st.line_chart(hourly)
    if "change_vs_usual" in fc.areas:
        chg = fc.areas.set_index("modzcta").change_vs_usual
        per_area = per_area.assign(vs_usual=per_area.area_id.map(chg))
        moved = fc.areas[(fc.areas.change_vs_usual - 1).abs() >= 0.25].sort_values("change_vs_usual", ascending=False)
        st.markdown("**How today's map differs from a usual day** (share of the city's calls vs the long-run share)")
        if moved.empty:
            st.caption("An ordinary day: no area's share moves by 25% or more. The calls follow the usual pattern.")
        else:
            st.dataframe(moved.assign(change_vs_usual=lambda d: (d.change_vs_usual - 1).map("{:+.0%}".format))
                         [["modzcta", "borough", "change_vs_usual", "expected"]].round(2), width="stretch", hide_index=True)
    top = per_area.sort_values("expected", ascending=False).head(15)
    if fc.observed_areas is not None:
        top = top.assign(observed=top.area_id.map(fc.observed_areas))
    st.dataframe(top.round(3), width="stretch", hide_index=True)


def deploy_page(sc: dl.Scenario, fc):
    backend, label = dl.solver_backend()
    st.subheader(f"Daily deployment · {sc.label}")
    (st.success if label.startswith("Gurobi") else st.warning)(f"Optimiser: {label}")
    c1, c2, c3 = st.columns(3)
    n_tents = c1.number_input("Tents available today", 1, 50, 10)
    source = c2.selectbox("Candidate sites", ["NYC parks & plazas (screened by reachable demand)", "Cool It! NYC cooling sites"])
    radius = c3.slider("Walking radius (km)", 0.3, 2.0, 1.0, 0.1)
    c4, c5, c6, c7 = st.columns(4)
    rate = c4.number_input("Patients per doctor-hour", 0.5, 10.0, 4.0, 0.5)
    min_docs = c5.number_input("Min doctors per open tent", 0, 4, 1)
    max_docs = c6.number_input("Max doctors per tent", 1, 10, 4)
    decay = c7.slider("Distance decay (km)", 0.1, 2.0, 0.6, 0.1, help="Willingness to walk falls as exp(-d / decay)")

    st.markdown("**Shifts**")
    shifts = st.data_editor(DEFAULT_SHIFTS, num_rows="dynamic", width="stretch", key="shifts")
    st.markdown("**Doctors** (available shifts as a comma-separated list)")
    n_doc = st.number_input("Number of doctors", 1, 200, 22)
    names = list(shifts.name)
    default_docs = pd.DataFrame({"doctor_id": [f"D{i + 1:02d}" for i in range(int(n_doc))],
                                 "available_shifts": [", ".join(names[(i % len(names)):(i % len(names)) + 2]) for i in range(int(n_doc))],
                                 "max_shifts": 1})
    doctors = st.data_editor(default_docs, num_rows="dynamic", width="stretch", key=f"doctors_{n_doc}")

    sites = dl.candidate_sites(source, fc.demand, radius)
    st.caption(f"{len(sites)} candidate sites · {fc.demand.scenario.nunique()} representative forecast scenarios "
               f"(stratified quantiles of the citywide predictive distribution)")
    if st.button("Solve deployment", type="primary"):
        blocks = sorted({int(h) for r in shifts.itertuples() for h in (r.start_hour, r.end_hour)})
        with st.spinner("Gurobi is optimising..."):
            plan, used = dl.run_solver(n_tents, sites, doctors, shifts, fc.demand, walk_radius_km=radius, distance_decay_km=decay,
                                       patients_per_doctor_hour=rate, min_doctors_per_open_tent=int(min_docs),
                                       max_doctors_per_tent=int(max_docs), time_limit_s=60.0, mip_gap=0.01,
                                       hour_blocks=list(zip(blocks[:-1], blocks[1:])))
        st.session_state["plan"] = (plan, used, radius)
    if "plan" not in st.session_state:
        return
    plan, used, radius = st.session_state["plan"]
    k = plan.kpis
    m = st.columns(4)
    m[0].metric("Expected cases (all hours)", f"{k.get('expected_cases', np.nan):.1f}")
    m[1].metric("Expected treated at tents", f"{k.get('expected_treated', np.nan):.1f}",
                help="Cases from areas within walking distance that tent capacity can absorb (scenario average).")
    m[2].metric("% of expected cases treated", f"{k.get('treated_pct', np.nan):.1f}%")
    m[3].metric("Tents / doctors used", f"{k.get('tents_open', len(plan.open_sites))} / {k.get('doctors_assigned', len(plan.assignments))}")
    st.caption(f"{used} · status {plan.status} · gap {plan.mip_gap:.4f} · {plan.runtime_s:.1f}s · "
               f"objective (distance-weighted) {k.get('expected_served', np.nan):.1f} · {plan.stats}")

    served = plan.served.groupby("area_id")[["expected_cases", "expected_served"]].sum().reset_index()
    served["unserved"] = served.expected_cases - served.expected_served
    O = plan.open_sites.copy()
    O["r"] = 120 + 90 * O.doctors_total
    layers = [
        choropleth(served, "unserved", ["expected_cases", "expected_served", "unserved"]),
        pdk.Layer("ScatterplotLayer", O, get_position=["lon", "lat"], get_radius=radius * 1000, stroked=True, filled=True,
                  get_fill_color=[30, 90, 200, 35], get_line_color=[30, 90, 200, 200], line_width_min_pixels=1),
        pdk.Layer("ScatterplotLayer", O, get_position=["lon", "lat"], get_radius="r", get_fill_color=[20, 60, 160, 230],
                  pickable=True),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=VIEW, map_style="light",
                             tooltip={"text": "{name}\ndoctors {doctors_total}\nZIP area {area_id}\nunserved {unserved}"}),
                    width="stretch")
    st.caption("Map: expected cases not reached by a tent, by area (colour); open tents (dots sized by doctors) and walking catchments (rings).")
    a, b = st.columns(2)
    a.markdown("**Open tents**")
    a.dataframe(plan.open_sites, width="stretch", hide_index=True)
    b.markdown("**Doctor assignments**")
    b.dataframe(plan.assignments, width="stretch", hide_index=True)
    st.markdown("**Areas with the most unserved expected cases**")
    st.dataframe(served.sort_values("unserved", ascending=False).head(15).round(2), width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download assignments (CSV)", plan.assignments.to_csv(index=False), "assignments.csv", "text/csv")
    d2.download_button("Download open tents (CSV)", plan.open_sites.to_csv(index=False), "open_tents.csv", "text/csv")


sc = sidebar_scenario()
fc = cached_forecast(sc)
st.sidebar.caption("The optimiser plans for 5 equally likely versions of tomorrow: the 10th, 30th, 50th, 70th and "
                   "90th percentiles of the citywide forecast.")
st.sidebar.divider()
st.sidebar.caption("Models: citywide DLNM-NB + LightGBM ensemble with analog forecast-error Monte Carlo; "
                   "area shares by the beach & event share model (weekend/holiday, beach season, event calendar); "
                   "Gurobi two-stage stochastic MILP.")
page = st.sidebar.radio("Screen", ["Forecast / simulation", "Deploy"])
st.title("NYC heat-illness response")
if page == "Forecast / simulation":
    forecast_page(sc, fc)
else:
    deploy_page(sc, fc)
