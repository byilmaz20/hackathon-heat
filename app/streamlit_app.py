"""NYC heat-illness forecast & cooling-tent deployment - decision-support app.

Run:  .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import data_layer as dl

st.set_page_config(page_title="NYC Heat Response", layout="wide")

# chart colours (categorical, validated for colour-blind separation and contrast on the dark surface)
MODEL_COLORS = {"GLM": "#3987e5", "GBM": "#d95926", "Ensemble": "#199e70"}
OBSERVED_COLOR = "#c3c2b7"
CHART_HEIGHT = 300
MAP_LAYER = "neighbourhoods"
DEFAULT_SHIFTS = pd.DataFrame({"name": ["morning", "afternoon", "evening"], "start_hour": [8, 12, 16], "end_hour": [14, 18, 22]})
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


def choropleth(values: pd.DataFrame, col: str, tooltip_cols: list[str], layer_id: str = MAP_LAYER) -> pdk.Layer:
    """Colour the 42 neighbourhoods by `col` (values keyed by area_id = uhf code)."""
    names = dl.neighbourhoods()[["area_id", "name", "borough"]]
    P = dl.neighbourhood_polygons().merge(names, on="area_id").merge(values, on="area_id", how="left").fillna({col: 0})
    P["fill"] = ramp(P[col].to_numpy(float))
    for c in tooltip_cols:
        if c in P and P[c].dtype.kind == "f":
            P[c] = P[c].round(2)
    return pdk.Layer("PolygonLayer", P, id=layer_id, get_polygon="polygon", get_fill_color="fill",
                     get_line_color=[90, 90, 90, 80], line_width_min_pixels=0.5, pickable=True, stroked=True)


def sidebar_scenario() -> dl.Scenario:
    mode = st.sidebar.radio("Day to forecast", ["Historical date", "Custom scenario"])
    if mode == "Historical date":
        date = st.sidebar.date_input("Date (May-Sep 2024-2026)", dt.date(2026, 7, 2), min_value=dt.date(2024, 5, 1),
                                     max_value=dt.date(2026, 9, 21))
        return dl.Scenario(label=f"{date:%Y-%m-%d} · forecast issued the evening before", date=str(date))
    hi_c = st.sidebar.slider("Max heat index today (°C)", 24, 46, 39)
    wk = st.sidebar.checkbox("Weekend")
    hol = st.sidebar.checkbox("Public holiday")
    # everything else keeps the Scenario defaults (day 3 of a hot spell, no upstream signal, no event)
    return dl.Scenario(label=f"scenario: {hi_c}°C heat index", heat_index_f=hi_c * 9 / 5 + 32, weekend=wk, holiday=hol)


@st.cache_data(show_spinner="Running the forecast models...")
def cached_forecast(sc: dl.Scenario):
    return dl.get_forecast(sc)


@st.cache_data(show_spinner=False)
def cached_nb_demand(sc: dl.Scenario) -> pd.DataFrame:
    """Hourly demand per neighbourhood point and scenario (optimiser input), from the 42-neighbourhood forecast."""
    return dl.neighbourhood_demand(cached_forecast(sc))


def _x(date_format: str = "%d %b") -> alt.X:
    return alt.X("date:T", title=None, axis=alt.Axis(format=date_format, grid=False))


def _date_rule(date: str) -> alt.Chart:
    """Dashed vertical line on the chosen date."""
    return alt.Chart(pd.DataFrame({"date": [pd.Timestamp(date)]})).mark_rule(
        color="#f0efec", strokeDash=[4, 4], strokeWidth=1).encode(x=_x())


def history_chart(H: pd.DataFrame, date: str, y_title: str) -> alt.Chart:
    """Observed (dots) vs ensemble forecast (line, 90% band) day by day around the chosen date."""
    long = H.melt("date", ["observed", "forecast"], var_name="series", value_name="calls")
    long["series"] = long.series.map({"observed": "Observed", "forecast": "Forecast (ensemble)"})
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Observed", "Forecast (ensemble)"],
                                      range=[OBSERVED_COLOR, MODEL_COLORS["Ensemble"]]))
    x = _x()
    band = alt.Chart(H).mark_area(opacity=0.18, color=MODEL_COLORS["Ensemble"]).encode(
        x=x, y=alt.Y("q05:Q", title=y_title), y2="q95:Q")
    line = alt.Chart(long[long.series != "Observed"]).mark_line(strokeWidth=2).encode(x=x, y="calls:Q", color=color)
    dots = alt.Chart(long[long.series == "Observed"]).mark_circle(size=45, opacity=1).encode(x=x, y="calls:Q", color=color)
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
    hit = alt.Chart(H).mark_rule(strokeWidth=12, opacity=0).encode(
        x=x, tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), alt.Tooltip("observed:Q", title="Observed"),
                      alt.Tooltip("forecast:Q", title="Forecast", format=".1f"),
                      alt.Tooltip("q05:Q", title="90% low"), alt.Tooltip("q95:Q", title="90% high")]).add_params(hover)
    cross = alt.Chart(H).mark_rule(color="#888", strokeWidth=1).encode(x=x).transform_filter(hover)
    return (band + line + dots + _date_rule(date) + cross + hit).properties(height=CHART_HEIGHT)


def models_chart(M: pd.DataFrame, date: str) -> alt.Chart:
    """Citywide calls: GLM, GBM and ensemble forecasts against the actual count, day by day."""
    domain = ["Actual", "GLM", "GBM", "Ensemble"]
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=domain, range=[OBSERVED_COLOR] + [MODEL_COLORS[m] for m in domain[1:]]))
    x = _x()
    y = alt.Y("calls:Q", title="Heat EMS calls (city)")
    lines = alt.Chart(M[M.series != "Actual"]).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    actual = alt.Chart(M[M.series == "Actual"]).mark_line(strokeWidth=1.5, strokeDash=[2, 2],
                                                          point=alt.OverlayMarkDef(size=45, filled=True)).encode(
        x=x, y=y, color=color)
    W = M.pivot(index="date", columns="series", values="calls").reset_index()
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
    hit = alt.Chart(W).mark_rule(strokeWidth=12, opacity=0).encode(
        x=x, tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y")]
        + [alt.Tooltip(f"{s}:Q", format=".1f") for s in domain if s in W]).add_params(hover)
    cross = alt.Chart(W).mark_rule(color="#888", strokeWidth=1).encode(x=x).transform_filter(hover)
    return (lines + actual + _date_rule(date) + cross + hit).properties(height=CHART_HEIGHT)


def selected_neighbourhood(event) -> dict | None:
    """The neighbourhood clicked on the map (area_id, name, borough), if any."""
    try:
        objs = event.selection["objects"].get(MAP_LAYER, [])
    except (AttributeError, KeyError, TypeError):
        return None
    return objs[0] if objs else None


def forecast_page(sc: dl.Scenario, fc, demand: pd.DataFrame):
    st.subheader(f"Heat-illness forecast · {sc.label}")
    st.metric("Expected heat EMS calls (city)", f"{fc.city['mean']:.0f}")
    per_nb = dl.summarise(demand, list(range(24)))
    # the map spec must not depend on the selection, or Streamlit resets the selection when it redraws
    event = st.pydeck_chart(pdk.Deck(layers=[choropleth(per_nb, "expected", ["expected"])], initial_view_state=VIEW,
                                     map_style="light", tooltip={"text": "{name} ({borough})\nexpected cases {expected}"}),
                            width="stretch", on_select="rerun", selection_mode="single-object", key="nb_map")
    chosen = selected_neighbourhood(event)
    st.caption("Expected heat EMS calls by neighbourhood (whole day, ensemble forecast). "
               "Click a neighbourhood to see its history; click it again to clear.")
    if sc.date is None:
        st.info("Pick a historical date in the sidebar to compare the forecasts with what happened.")
        return
    st.markdown("**City: observed vs forecast**")
    st.altair_chart(history_chart(dl.city_history(sc.date), sc.date, "Heat EMS calls (city)"), width="stretch")
    if chosen:
        st.markdown(f"**{chosen['name']}: observed vs forecast**")
        H = dl.neighbourhood_history(str(chosen["area_id"]), sc.date)
        st.altair_chart(history_chart(H, sc.date, "Heat EMS calls"), width="stretch")
    else:
        st.info("Click a neighbourhood on the map to see its observed vs forecast calls here.")
    st.markdown("**City total: GLM, GBM, ensemble and actual**")
    st.altair_chart(models_chart(dl.city_models_history(sc.date), sc.date), width="stretch")
    st.caption("30 days before and 7 days after the chosen date (dashed line). Forecasts are issued the evening before; "
               "the band is the ensemble's 90% interval.")


def deploy_page(sc: dl.Scenario, fc, demand: pd.DataFrame):
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

    sites = dl.candidate_sites(source, demand, radius)
    st.caption(f"{len(sites)} candidate sites · demand for the 42 neighbourhoods in {demand.scenario.nunique()} "
               f"equally likely forecast scenarios (10th-90th percentiles of the citywide forecast)")
    if st.button("Solve deployment", type="primary"):
        blocks = sorted({int(h) for r in shifts.itertuples() for h in (r.start_hour, r.end_hour)})
        with st.spinner("Gurobi is optimising..."):
            plan, used = dl.run_solver(n_tents, sites, doctors, shifts, demand, walk_radius_km=radius, distance_decay_km=decay,
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
                             tooltip={"text": "{name}\ndoctors {doctors_total}\nunserved {unserved}"}),
                    width="stretch")
    st.caption("Map: expected cases not reached by a tent, by neighbourhood (colour); open tents (dots sized by doctors) and walking catchments (rings).")
    a, b = st.columns(2)
    a.markdown("**Open tents**")
    a.dataframe(plan.open_sites, width="stretch", hide_index=True)
    b.markdown("**Doctor assignments**")
    b.dataframe(plan.assignments, width="stretch", hide_index=True)
    st.markdown("**Neighbourhoods with the most unserved expected cases**")
    worst = dl.neighbourhoods()[["area_id", "name", "borough"]].merge(served, on="area_id")
    st.dataframe(worst.sort_values("unserved", ascending=False).head(15).drop(columns="area_id").round(2),
                 width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download assignments (CSV)", plan.assignments.to_csv(index=False), "assignments.csv", "text/csv")
    d2.download_button("Download open tents (CSV)", plan.open_sites.to_csv(index=False), "open_tents.csv", "text/csv")


sc = sidebar_scenario()
fc = cached_forecast(sc)
st.title("NYC heat-illness response")
if dl.MOCK:
    st.caption(f"⚠️ {dl.DATA_SOURCE}")
page = st.radio("Screen", ["Forecast / simulation", "Deploy"], horizontal=True, label_visibility="collapsed")
demand = cached_nb_demand(sc)
if page == "Forecast / simulation":
    forecast_page(sc, fc, demand)
else:
    deploy_page(sc, fc, demand)
