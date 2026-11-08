# Decision-support app (Streamlit)

```bash
.venv/bin/python app/precompute.py        # once: data behind the slide tabs (about 30 s)
.venv/bin/streamlit run app/streamlit_app.py
```

Runs on the real data and trained models only: `heattents features`, `heattents city` and `heattents area` must have
been run, and a Gurobi licence is needed to place the field hospitals. All numbers are people needing help (forecast
heat EMS calls x `heattents.service.PEOPLE_PER_CALL`, converted once right after the forecast).

## Four tabs

- **Forecast & plan.** Pick a day in the sidebar (May–September 2024–2026); the forecast is shown as issued at 18:00
  the evening before. Two metrics (people expected to need help, people expected to be helped), an input box for the
  number of field hospitals, and a map of the 42 UHF neighbourhoods with the hospitals and their reach rings. Below:
  citywide observed vs forecast, and GLM / GBM / ensemble vs actual, around the chosen date.
- **Predictive model.** Pipeline diagram (city model GLM + GBM ensemble, then the beach & event share model to the
  42 neighbourhoods), every input of both models, out-of-sample accuracy, why two models, and the 2026 forecast vs
  observed (models trained on 2005–2025 only) for the city or any neighbourhood.
- **MILP model.** The active maximum-coverage model: objective, constraints, decision variables and parameters.
- **Results.** Population rule, heat-vulnerability rule, forecast + MILP and the best possible plan (hindsight) on
  Fri 21 Jun 2024 and Pride Sun 29 Jun 2025, scored on the people who actually needed help
  (`heattents.deploy.compare.run_coverage`), with maps of each plan.

## Planning rules

`data_layer.RULES` / `ACTIVE_RULES`. Active: `"threshold"` — maximum coverage, a hospital helps everyone within
2 km (no worker limit), sites anywhere on land (400 m grid plus parks), up to 3 per neighbourhood. `"previous"` is the
earlier two-stage stochastic model (parks only, 15-minute walk, 4 health workers per hospital, 30 people per worker
a day, one per neighbourhood). See `src/heattents/deploy/daily.py`.

`data_layer.py` is the app's only access to the models and the solver; `precompute.py` writes `outputs/app/` and
`outputs/models/deploy_compare/coverage/`.
