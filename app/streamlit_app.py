"""NYC heat-illness forecast and field-hospital placement - decision-support app (real data and trained models).

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

st.set_page_config(page_title="NYC Heat Response", layout="wide", initial_sidebar_state="expanded")
# presentation mode: keep only the sidebar open/close arrow from Streamlit's top bar (menu, Deploy and the
# running indicator are hidden) and trim the empty space above the title
st.html("""<style>
header[data-testid="stHeader"] {background: transparent; height: 2.6rem;}
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"],
[data-testid="stToolbarActions"], [data-testid="stDecoration"] {display: none !important;}
.block-container {padding-top: 2.4rem;}
</style>""")

# chart colours (categorical, validated for colour-blind separation and contrast on the dark surface)
MODEL_COLORS = {"GLM": "#3987e5", "GBM": "#d95926", "Ensemble": "#199e70"}
OBSERVED_COLOR = "#c3c2b7"
CHART_HEIGHT = 300
MAP_LAYER = "neighbourhoods"
HOSPITAL_LAYER = "hospitals"
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


def choropleth(values: pd.DataFrame, col: str, tooltip_cols: list[str], layer_id: str = MAP_LAYER,
               detail: str = "expected people needing help") -> pdk.Layer:
    """Colour the 42 neighbourhoods by `col` (values keyed by area_id = uhf code)."""
    names = dl.neighbourhoods()[["area_id", "name", "borough"]]
    P = dl.neighbourhood_polygons().merge(names, on="area_id").merge(values, on="area_id", how="left").fillna({col: 0})
    P["fill"] = ramp(P[col].to_numpy(float))
    for c in tooltip_cols:
        if c in P and P[c].dtype.kind == "f":
            P[c] = P[c].round(2)
    P["detail"] = [f"{b} · {detail} {v:,.0f}" for b, v in zip(P.borough.str.title(), P[col])]
    return pdk.Layer("PolygonLayer", P, id=layer_id, get_polygon="polygon", get_fill_color="fill",
                     get_line_color=[90, 90, 90, 80], line_width_min_pixels=0.5, pickable=True, stroked=True)


def sidebar_scenario() -> dl.Scenario:
    date = st.sidebar.date_input("Day to forecast (May-Sep 2024-2026)", dt.date(2026, 6, 23), min_value=dt.date(2024, 5, 1),
                                 max_value=dt.date(2026, 9, 21))
    st.sidebar.caption("The forecast is shown as it was issued at 18:00 the evening before, from the archived weather "
                       "forecast.")
    return dl.Scenario(label=f"{date:%a %d %b %Y}", date=str(date))


@st.cache_data(show_spinner="Running the forecast models...")
def cached_forecast(sc: dl.Scenario):
    return dl.get_forecast(sc)


@st.cache_data(show_spinner=False)
def cached_plan(sc: dl.Scenario, n_hospitals: int):
    return dl.plan_hospitals(cached_forecast(sc), n_hospitals)


def hospital_layers(plan) -> list[pdk.Layer]:
    """Reach circles (the plan's radius) and hospital dots (sized by health workers when capacity is modelled)."""
    from heattents.deploy.daily import walk_radius_km
    H = plan.tents.copy()
    unlimited = plan.kpis.get("unlimited_capacity", False)
    if unlimited:
        H["detail"] = [f"helps everyone within reach · expected {a:.0f} people" for a in H.answered]
        H["r"] = 220
    else:
        H["detail"] = [f"{w} health workers · up to {int(c)} people a day · expected {a:.0f} helped"
                       for w, c, a in zip(H.workers, H.capacity, H.answered)]
        H["r"] = 90 + 45 * H.workers
    radius_m = 1000 * plan.kpis.get("radius_km", walk_radius_km())
    return [pdk.Layer("ScatterplotLayer", H, id="walk", get_position=["lon", "lat"], get_radius=radius_m,
                      filled=True, stroked=True, get_fill_color=[30, 90, 200, 30], get_line_color=[30, 90, 200, 220],
                      line_width_min_pixels=1.5),
            pdk.Layer("ScatterplotLayer", H, id=HOSPITAL_LAYER, get_position=["lon", "lat"], get_radius="r",
                      get_fill_color=[20, 60, 160, 240], get_line_color=[255, 255, 255, 255], stroked=True,
                      line_width_min_pixels=1.5, pickable=True)]


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
    y = alt.Y("calls:Q", title="People needing help (city)")
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


def forecast_page(sc: dl.Scenario, fc):
    st.subheader(f"Heat-illness forecast · {sc.label}")
    c = fc.city
    m1, m2 = st.columns(2)  # filled once the plan is known, but shown above the map

    rules = dl.RULES[dl.ACTIVE_RULES]
    limited = rules["min_calls_per_hospital"] > 0
    q = rules["max_per_neighbourhood"]
    n_hospitals = st.number_input("Maximum number of field hospitals" if limited else "Number of field hospitals to place",
                                  min_value=1, max_value=42 * q, value=10, step=1,
                                  help=("Each hospital helps everyone within reach (no worker limit). "
                                        if rules.get("unlimited_capacity") else
                                        f"Each hospital gets {dl.WORKERS_PER_HOSPITAL} health workers. ") + f"At most {q} per "
                                       "neighbourhood." + (f" A hospital opens only if it is expected to serve at least "
                                       f"{rules['min_calls_per_hospital']:g} calls' worth of people." if limited else ""))
    plan = cached_plan(sc, int(n_hospitals)) if dl.deployment_needed(fc) else None
    has_tents = plan is not None and len(plan.tents) > 0
    need = float((fc.scenarios.located_total * fc.scenarios.probability).sum())  # same base as the plan's % helped
    m1.metric("People expected to need help", f"{need:,.0f}",
              help=f"Forecast heat EMS calls with a known location × {fc.people_per_call:g} people per call, averaged "
                   "over the 5 forecast scenarios.")
    if has_tents:
        m2.metric("People expected to be helped",
                  f"{plan.kpis['expected_answered']:,.0f} ({plan.kpis['answered_pct']:.0f}%)")
    else:
        m2.metric("People expected to be helped", "0")
    per_nb = dl.neighbourhood_forecast(fc)
    layers = [choropleth(per_nb, "expected", ["expected"])] + (hospital_layers(plan) if has_tents else [])
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=VIEW, map_style="light",
                             tooltip={"text": "{name}\n{detail}"}), width="stretch", key="nb_map")
    if plan is None:
        st.info(f"**No deployment needed.** The forecast expects {c['mean'] / fc.people_per_call:.1f} heat calls "
                f"({c['mean']:,.0f} people) citywide, below the "
                f"activation threshold of {rules['activation_calls']:g}.")
    elif not has_tents:
        st.info(f"**No field hospital is worth opening.** No site would serve at least "
                f"{rules['min_calls_per_hospital']:g} calls' worth of people within a 15-minute walk.")

    st.markdown("**City: observed vs forecast**")
    st.altair_chart(history_chart(dl.city_history(sc.date), sc.date, "People needing help (city)"), width="stretch")
    st.markdown("**City total: GLM, GBM, ensemble and actual**")
    st.altair_chart(models_chart(dl.city_models_history(sc.date), sc.date), width="stretch")
    st.caption("30 days before and 7 days after the chosen date (dashed line). Forecasts are issued the evening before; "
               "the band is the ensemble's 90% interval.")


# --- slide tabs ------------------------------------------------------------------------------------------------------
PIPELINE = r"""digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.2; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#f4f3ef", color="#9a9890", fontname="Helvetica", fontsize=11,
        margin="0.15,0.06"];
  edge [color="#9a9890", arrowsize=0.6];
  ci [label="City inputs", color="#199e70"];
  glm [label="GLM\n(heat curve with lags)", fillcolor="#dbe8fa", color="#3987e5"];
  gbm [label="GBM\n(500 trees, monotone)", fillcolor="#f8e0d4", color="#d95926"];
  ens [label="Ensemble\naverage of the two", fillcolor="#d5efe5", color="#199e70"];
  city [label="City forecast\n+ 5 scenarios"];
  si [label="Share inputs", color="#e08a3c"];
  share [label="Share model\nwhere the calls land"];
  nb [label="42 neighbourhoods\npeople needing help", fillcolor="#fde3c8", color="#e08a3c"];
  ci -> glm; ci -> gbm; glm -> ens; gbm -> ens; ens -> city; city -> share; si -> share; share -> nb;
  {rank=same; city; si}
}"""
# every model input once: (card title, colour, [(group, [inputs])]); the heat calls are shown first as the main data
CALLS = ("Heat EMS calls", "NYC 911 ambulance dispatches coded as heat illness, May–September 2005–2026, each with "
         "date, time and ZIP code. They are what we forecast, and both models learn from them:",
         [("City model", "calls the day before, up to 18:00"),
          ("Share model", "each ZIP area's usual share of the calls")])
INPUT_CARDS = [
    ("City model · how many people", MODEL_COLORS["Ensemble"], [
        ("Weather forecast for the day", ["Heat index (max)", "Max / min / mean temperature", "Humidity",
                                          "Hot night (≥ 24 °C)"]),
        ("Heat build-up", ["Heat index, past 3 days", "Hot days in a row", "Hot days so far this summer",
                           "Heat vs normal", "Heat vs past month", "Heat to the south-west (Philadelphia–DC)"]),
        ("Calendar", ["Weekends and holidays", "4 July", "School break", "COVID period", "Year", "Day of year"]),
    ]),
    ("Share model · where", "#e08a3c", [
        ("Each ZIP area", ["Beach", "Visitors (jobs + subway entries)", "Parks"]),
        ("The day", ["Weekends and holidays", "Beach season", "Heat level (beach areas)",
                     "6 event days and routes: Pride, Puerto Rican Day, Dominican Day, West Indian Day, "
                     "Mermaid Parade, US Open"]),
    ]),
]


def inputs_html() -> str:
    chip = ("<span style='display:inline-block;padding:2px 9px;margin:2px 4px 2px 0;border-radius:11px;"
            "background:rgba(140,140,140,.16);font-size:.8rem;line-height:1.5'>{}</span>")
    head = "<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;opacity:.6;margin:8px 0 2px'>{}</div>"
    title, text, uses = CALLS
    used = "".join(f"<div style='font-size:.82rem;margin-top:2px'><b>{m}:</b> {u}</div>" for m, u in uses)
    calls = (f"<div style='border-left:3px solid #c3c2b7;background:rgba(140,140,140,.08);border-radius:4px;"
             f"padding:6px 12px;margin-bottom:6px'>{head.format('Main data')}"
             f"<div style='font-weight:600;font-size:.95rem'>{title}</div>"
             f"<div style='font-size:.82rem;opacity:.85'>{text}</div>{used}</div>")
    cards = []
    for t, color, groups in INPUT_CARDS:
        body = "".join(head.format(g) + "".join(chip.format(i) for i in items) for g, items in groups)
        cards.append(f"<div style='flex:1;min-width:0;border-left:3px solid {color};padding:2px 0 4px 12px'>"
                     f"<div style='font-weight:600;font-size:.95rem'>{t}</div>{body}</div>")
    return f"<div style='font-family:inherit'>{calls}<div style='display:flex;gap:22px'>{''.join(cards)}</div></div>"


SLIDE_CHART_HEIGHT = 210


def oos_chart(H: pd.DataFrame, band: bool) -> alt.Chart:
    """2026 out-of-sample forecast (line) vs observed (dots)."""
    x = _x("%b")
    y = alt.Y("forecast:Q", title="People needing help")
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Observed", "Forecast"], range=[OBSERVED_COLOR, MODEL_COLORS["Ensemble"]]))
    layers = []
    if band:
        layers.append(alt.Chart(H).mark_area(opacity=0.18, color=MODEL_COLORS["Ensemble"]).encode(x=x, y="q05:Q", y2="q95:Q"))
    layers += [alt.Chart(H.assign(series="Forecast")).mark_line(strokeWidth=2).encode(x=x, y=y, color=color),
               alt.Chart(H.assign(series="Observed")).mark_circle(size=22, opacity=0.9).encode(
                   x=x, y="observed:Q", color=color)]
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
    hit = alt.Chart(H).mark_rule(strokeWidth=8, opacity=0).encode(
        x=x, tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), alt.Tooltip("observed:Q", title="Observed", format=",.0f"),
                      alt.Tooltip("forecast:Q", title="Forecast", format=",.1f")]).add_params(hover)
    cross = alt.Chart(H).mark_rule(color="#888", strokeWidth=1).encode(x=x).transform_filter(hover)
    return alt.layer(*layers, cross, hit).properties(height=SLIDE_CHART_HEIGHT)


def predictive_page():
    st.subheader("How the forecast works")
    st.graphviz_chart(PIPELINE, width="stretch")
    a, b = st.columns([1.7, 1])
    with a.container(border=True):
        st.markdown("**Inputs** · all known at 18:00 the evening before")
        st.html(inputs_html())
    acc = dl.city_accuracy()
    with b.container(border=True):
        st.markdown("**Accuracy** · 2024-26, out of sample")
        big = ("<div style='font-size:1.8rem;font-weight:600;line-height:1.1'>{}</div>"
               "<div style='opacity:.7;font-size:.85rem;margin-bottom:8px'>{}</div>")
        k1, k2 = st.columns(2)
        k1.markdown(big.format(f"{acc['median_error']:,.0f}", f"people: typical error on an average day "
                                                               f"(~{acc['avg_people']:,.0f} need help)"),
                    unsafe_allow_html=True)
        k2.markdown(big.format(f"{acc['in_range_pct']:.0f}%",
                               "of days the actual number fell inside the forecast's likely range"),
                    unsafe_allow_html=True)
    with b.container(border=True):
        be = acc["busiest_error"]
        st.markdown("**Why two models**")
        st.markdown(f"GLM is better on peak days (it extrapolates the heat curve), GBM on ordinary days. Their errors "
                    f"go in opposite directions, so the average balances them: error on the 8 busiest days "
                    f"GLM {be['GLM']:,.0f} · GBM {be['GBM']:,.0f} · **ensemble {be['Ensemble']:,.0f}** people.")
    N = dl.neighbourhoods().sort_values("name")
    left, right = st.columns([3, 1])
    left.markdown("**2026: forecast vs observed** · models trained on 2005-2025 only")
    pick = right.selectbox("Area", ["New York City"] + N.name.tolist(), label_visibility="collapsed")
    uhf = None if pick == "New York City" else N.set_index("name").loc[pick, "uhf"]
    st.altair_chart(oos_chart(dl.oos_2026(uhf), band=uhf is None), width="stretch")


def milp_page():
    st.subheader("Where to place the field hospitals: maximum-coverage MILP")
    a, b, c = st.columns([1, 1.25, 1.15])
    with a.container(border=True):
        st.markdown("**Objective**")
        st.latex(r"\max \; \sum_{a} \mathbb{E}[D_a]\, u_a")
        st.latex(r"\mathbb{E}[D_a] = \sum_{s=1}^{5} p_s\, D_{as}")
        st.caption("Maximise the expected number of people within reach of a field hospital, i.e. minimise the people "
                   "left without help, over the 5 forecast scenarios.")
    with b.container(border=True):
        st.markdown("**Constraints**")
        st.markdown("Hospital budget")
        st.latex(r"\sum_j x_j \le N")
        st.markdown(f"At most {dl.RULES[dl.ACTIVE_RULES]['max_per_neighbourhood']} per neighbourhood")
        st.latex(r"\sum_{j \in J_g} x_j \le Q \quad \forall g")
        st.markdown(f"Helped only if a hospital is within {dl.RULES[dl.ACTIVE_RULES]['radius_km']:g} km")
        st.latex(r"u_a \le \sum_{j:\, a \in R(j)} x_j \quad \forall a")
        st.markdown("Domains")
        st.latex(r"x_j \in \{0,1\}, \quad 0 \le u_a \le 1")
    with c.container(border=True):
        st.markdown("**Decision variables**")
        st.markdown("- $x_j$ = 1 if a field hospital opens at site $j$ (0 otherwise)\n- $u_a$ = 1 if point $a$ is "
                    "within reach of an open hospital (0 otherwise)")
        st.markdown("**Parameters**")
        st.markdown(
            "- $a$: demand points, up to 4 inside each ZIP area (708)\n"
            "- $j$: candidate sites anywhere on land (every 400 m, plus parks)\n"
            "- $s$: 5 forecast scenarios, $p_s = 0.2$ each\n"
            "- $D_{as}$: people needing help at $a$ in scenario $s$\n"
            "- $N$: number of hospitals (input box)\n"
            f"- $Q = {dl.RULES[dl.ACTIVE_RULES]['max_per_neighbourhood']}$; $J_g$: sites in neighbourhood $g$\n"
            f"- $R(j)$: points within {dl.RULES[dl.ACTIVE_RULES]['radius_km']:g} km of site $j$")
        st.caption("Solved with Gurobi in about a second.")


PLAN_LABELS = {"population": "Population rule", "vulnerability (HVI)": "Heat-vulnerability rule",
               "forecast + MILP": "Forecast + MILP", "hindsight (upper bound)": "Best possible"}
# our plan in the ensemble green; the other plans in neutral grey
PLAN_COLORS = {"population": "#a8a69e", "vulnerability (HVI)": "#a8a69e", "forecast + MILP": MODEL_COLORS["Ensemble"],
               "hindsight (upper bound)": "#a8a69e"}
DAY_LABELS = {"2024-06-21": "Fri 21 Jun 2024 · heatwave", "2025-06-29": "Sun 29 Jun 2025 · Pride"}


def results_bars(R: pd.DataFrame) -> alt.Chart:
    order = list(PLAN_LABELS.values())
    R = R[R.plan.isin(PLAN_LABELS)]
    R = R.assign(label=R.plan.map(PLAN_LABELS))
    color = alt.Color("label:N", legend=None, scale=alt.Scale(domain=order, range=[PLAN_COLORS[p] for p in PLAN_LABELS]))
    y = alt.Y("label:N", title=None, sort=order, scale=alt.Scale(paddingInner=0.18, paddingOuter=0.05),
              axis=alt.Axis(labelLimit=0, labelOverlap=False, labelFontSize=12, labelPadding=6, ticks=False))
    bars = alt.Chart(R).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("helped:Q", title="People helped", axis=alt.Axis(grid=False)), y=y, color=color,
        tooltip=[alt.Tooltip("label:N", title="Plan"), alt.Tooltip("helped:Q", title="Helped", format=",.0f"),
                 alt.Tooltip("helped_pct:Q", title="% of people needing help", format=".0f")])
    text = alt.Chart(R.assign(t=[f"{h:,.0f} ({p:.0f}%)" for h, p in zip(R.helped, R.helped_pct)])).mark_text(
        align="left", dx=4, fontSize=11, color="#888").encode(x="helped:Q", y=y, text="t:N")
    return (bars + text).properties(height=34 * len(R))


def results_map(date: str, plan: str) -> pdk.Deck:
    obs = dl.observed_neighbourhoods(date)
    S = dl.coverage_sites()
    S = S[(S.date == date) & (S.plan == plan)].assign(detail=lambda d: d.name)
    radius_m = 1000 * dl.RULES[dl.ACTIVE_RULES]["radius_km"]
    layer = choropleth(obs, "observed", ["observed"], layer_id=f"obs_{date}", detail="people who needed help")
    return pdk.Deck(layers=[layer,
                            pdk.Layer("ScatterplotLayer", S, get_position=["lon", "lat"], get_radius=radius_m,
                                      get_fill_color=[30, 90, 200, 30], get_line_color=[30, 90, 200, 220],
                                      stroked=True, line_width_min_pixels=1),
                            pdk.Layer("ScatterplotLayer", S, get_position=["lon", "lat"], get_radius=250,
                                      get_fill_color=[20, 60, 160, 240], get_line_color=[255, 255, 255, 255],
                                      stroked=True, line_width_min_pixels=1, pickable=True)],
                    initial_view_state=pdk.ViewState(latitude=40.70, longitude=-73.95, zoom=8.9, pitch=0),
                    map_style="light", tooltip={"text": "{name}\n{detail}"})


def results_page():
    R = dl.coverage_results()
    n = int(dl.coverage_sites().groupby(["date", "plan"]).size().max())
    st.subheader(f"Results: {n} field hospitals, scored on the people who actually needed help")
    plan = st.segmented_control("Hospitals on the maps", list(PLAN_LABELS), default="forecast + MILP",
                                format_func=PLAN_LABELS.get) or "forecast + MILP"
    cols = st.columns(2)
    for col, (date, label) in zip(cols, DAY_LABELS.items()):
        D = R[R.date == date]
        with col:
            st.markdown(f"**{label}** · {D.needed.iloc[0]:,.0f} people needed help")
            st.altair_chart(results_bars(D), width="stretch")
            st.pydeck_chart(results_map(date, plan), height=250, width="stretch", key=f"res_{date}")


sc = sidebar_scenario()
st.title("NYC Heat-Illness Response")
t1, t2, t3, t4 = st.tabs(["Forecast & plan", "Predictive model", "MILP model", "Results"])
with t1:
    forecast_page(sc, cached_forecast(sc))
with t2:
    predictive_page()
with t3:
    milp_page()
with t4:
    results_page()
