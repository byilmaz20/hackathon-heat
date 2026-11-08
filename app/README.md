# Decision-support app (Streamlit)

```bash
.venv/bin/python app/precompute.py        # once: data behind the slide tabs (about 30 s)
.venv/bin/streamlit run app/streamlit_app.py
./app/present.sh                          # optional: presentation window (see below)
```

Run all commands from the repository root (`.streamlit/config.toml` is read from there). The app runs on the real data
and trained models only: `heattents features`, `heattents city` and `heattents area` must have been run, and a Gurobi
licence is needed to place the field hospitals. All numbers are people needing help (forecast heat EMS calls x
`heattents.service.PEOPLE_PER_CALL`, converted once right after the forecast); the slides do not mention the factor.

## Four tabs

- **Forecast & plan.** Pick a day in the sidebar (May–September 2024–2026, default 23 June 2026); the forecast is shown
  as issued at 18:00 the evening before. Two metrics (people expected to need help, people expected to be helped), an
  input box for the number of field hospitals, and a map of the 42 UHF neighbourhoods with the hospitals and their
  2 km reach rings (hover for details). Below: citywide observed vs forecast, and GLM / GBM / ensemble vs actual,
  around the chosen date.
- **Predictive model.** Pipeline diagram (city inputs → GLM and GBM → ensemble → city forecast → share model with
  its own inputs → 42 neighbourhoods); the heat EMS calls as the main data and every other input of both models,
  grouped; out-of-sample accuracy (typical error on an average day, share of days inside the forecast range); why two
  models; the 2026 forecast vs observed (models trained on 2005–2025 only) for the city or any neighbourhood.
- **MILP model.** The active maximum-coverage model: objective, constraints, decision variables and parameters.
- **Results.** Population rule, heat-vulnerability rule, forecast + MILP (green) and the best possible plan on
  Fri 21 Jun 2024 and Pride Sun 29 Jun 2025, scored on the people who actually needed help
  (`heattents.deploy.compare.run_coverage`), with a map of the selected plan's hospitals for each day.

## Presentation mode

`app/present.sh` opens the app in a Chrome window without tabs or address bar, full screen and in English (its own
Chrome profile, so your normal Chrome is untouched). Cmd+Ctrl+F leaves full screen, Cmd+Q closes it. Streamlit's own
top bar (menu, Deploy, running indicator) is hidden; the » arrow in the top-left corner reopens the sidebar.

## Planning rules

`data_layer.RULES` / `ACTIVE_RULES`. Active: `"threshold"` — maximum coverage, a hospital helps everyone within
2 km (no worker limit), sites anywhere on land (400 m grid plus parks), up to 3 per neighbourhood. `"previous"` is the
earlier two-stage stochastic model (parks only, 15-minute walk, 4 health workers per hospital, 30 people per worker
a day, one per neighbourhood). See `src/heattents/deploy/daily.py`.

## Files

| File | Role |
|---|---|
| `streamlit_app.py` | The four tabs |
| `data_layer.py` | The app's only access to the models, the solver and the precomputed data |
| `precompute.py` | Writes `outputs/app/` (2026 out-of-sample neighbourhood forecasts, input correlations) and `outputs/models/deploy_compare/coverage/` (plan comparison) |
| `present.sh` | Presentation window |

When `data_layer.py` changes, restart the app: Streamlit reloads the page script but can keep the old data layer.
