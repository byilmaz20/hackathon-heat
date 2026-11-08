# Decision-support app (Streamlit)

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

**Mock mode (for UI work).** With no `data/`, trained models or Gurobi licence, the app runs on `app/mock_data.py`
(synthetic NYC geography, forecasts, neighbourhoods and candidate sites in exactly the shapes the real pipeline returns)
and the greedy solver in `app/mock.py`. It switches on automatically when `outputs/models/share/share_model.pkl` is
missing; force it with `HEAT_MOCK=1`, or force the real pipeline with `HEAT_MOCK=0`. A sidebar banner shows when it is on.

**Spatial unit: the 42 UHF neighbourhoods.** Maps, tables and the optimiser input all use the prediction output
`fc.demand_neighbourhood` (42 neighbourhoods x 5 scenarios, daily). `data_layer.neighbourhood_demand` splits it into
hours with the forecast's citywide hour profile and spreads each neighbourhood's demand evenly over ~10 points inside
it (`area_id = "uhf#j"`), so walking distances to tents stay meaningful.

The app has two screens, chosen in the sidebar.

## 1. Forecast / simulation

In the sidebar, pick one of two inputs:

- **A historical date.** The heat index comes from the LaGuardia record, and the hot-spell day and weekend flag are derived from it.
- **A custom scenario.** You set the max heat index, the day in the hot spell, whether it is a weekend or holiday, and the upstream-cities signal.

For the chosen hour band, the screen shows:

- City-wide expected heat EMS calls, with a 90% prediction interval.
- The observed count, when you pick a historical date.
- A choropleth over the 177 MODZCTA areas. You can colour it by expected cases, by P(cases ≥ k), or by the upper 95% bound.
- A city-wide hourly curve with a 5–95% band across forecast scenarios.

## 2. Deploy

Inputs:

- tents available
- candidate sites: Cool It! NYC sites, or one placeholder per area until the parks ingest exists
- walking radius and distance decay
- patients per doctor-hour
- minimum and maximum doctors per tent
- shift table (editable)
- doctor table: id, available shifts, maximum shifts (editable)
- how many forecast scenarios the optimiser hedges over

**Solve** calls `heattents.deploy.milp.solve` (Gurobi). If that module cannot be imported, it falls back to `app/mock.py`, a greedy, clearly labelled, non-optimal heuristic with the same interface.

Outputs:

- KPI cards
- map of unserved expected cases with open tents (sized by doctors) and walking rings
- open-tent table and doctor-assignment table
- worst-served areas
- CSV downloads

## Real vs mock

| Piece | Status | Where to plug in |
|---|---|---|
| Forecast | **MOCK**: Poisson curve of daily EMS HEAT calls on LaGuardia max heat index (fitted on 2017+), × hot-spell, weekend and upstream multipliers, × historical area share shrunk to population, × empirical hourly profile, with gamma noise at city and area level (40 Monte-Carlo scenarios) | `data_layer.get_forecast(scenario)`. It must return `[area_id, lat, lon, hour, scenario, expected_cases]`. |
| Historical scenario | Real (LaGuardia temperature/heat index, observed EMS counts) | `data_layer.scenario_for_date` |
| Geography | Real (MODZCTA polygons, ZIP → MODZCTA mapping) | `heattents.ingest.geography` |
| Candidate sites | Real Cool It! NYC 2020 sites (active only, grouped by property); parks are a placeholder | `data_layer.candidate_sites` |
| Optimiser | Real Gurobi model when `heattents.deploy.milp` imports, otherwise mock | `data_layer.solver_backend` / `run_solver`; `normalise_plan` gives the app one plan shape for both backends |

Files: `streamlit_app.py` (UI), `data_layer.py` (only data and solver access), `mock.py` (mock dataclasses and greedy solver).
