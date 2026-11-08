# NYC Heat Response: probabilistic heat-illness forecasting and daily field-hospital placement

On the evening before a hot day a New York City planner has a handful of field hospitals (pop-up cooling tents).
**How many people will need help tomorrow, where, and where should the field hospitals go so that as many of them as
possible are within reach?**

This project answers that question end to end, from public data:

```
EMS heat calls (label) ─┐
weather + forecasts ────┼─► [1] citywide probabilistic forecast ──┐
upstream cities ────────┘     (DLNM-NB + LightGBM, forecast-error MC)│
                                                                      ├─► [3] daily demand: 42 neighbourhoods × 5 scenarios ─► [4] Gurobi MILP ─► app
weekend, beach, events ────► [2] beach & event share model ─────────┘     (10/30/50/70/90th percentiles)   (field hospitals)
(usual share per ZIP area)
```

The label is **where and when heat-illness emergencies actually happened**, so no disaggregation is needed: the
models learn directly from 17,688 geocoded NYC EMS heat calls (2005–2026).

---

## Headline results

| | Result |
|---|---|
| **Citywide forecast** | Forecast at 18:00 for the next day, tested out of sample on 2024–26 (450 days). Median daily error 1 call, mean 3.3; correlation with actual 0.93. CRPS is 22% lower than a temperature-only model and 58% lower than climatology. Full documentation: [docs/city_model.md](docs/city_model.md). |
| **Calibration** | 90% prediction intervals contain the actual count on 96% of days. Forecasts correlate **0.89** with an *independent* label (citywide heat ED visits). |
| **Busiest days** | 6 of the 8 test days with ≥ 50 calls were flagged the evening before. On 2 July 2026 (231 calls) the forecast was 211. |
| **Where** | The usual map is stable (area rank correlation 0.94 between 2005–15 and 2016–26) and is driven by *daytime presence*, not resident vulnerability. The beach & event share model moves it where the data supports a move: weekends/holidays, beach season and permitted events (+0.11 nats/call on event days, with calibrated moves). Heat moves *how many* calls, not *where*. |
| **Decision** | A maximum-coverage MILP places N field hospitals (2 km reach, up to 3 per neighbourhood, sites anywhere on land) in about a second. With 10 hospitals and out-of-sample forecasts it reached **54%** of the people who actually needed help on 21 Jun 2024 (population rule 29%, heat-vulnerability rule 40%) and **58%** on Pride Sunday 29 Jun 2025 (19% and 16%). |

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
brew install libomp                      # macOS only: OpenMP runtime for LightGBM
# Gurobi: a full licence is needed (models have ~50k variables); e.g. a free academic WLS licence in ~/gurobi.lic

.venv/bin/heattents ingest              # download all inputs (cached; Open-Meteo has a daily quota, re-run to resume)
.venv/bin/heattents features            # area table, citywide series, area × day panel
.venv/bin/heattents city                # citywide model: 2024-26 evaluation, final model, heat curve  (~5 s)
.venv/bin/heattents area                # share model: 2018-26 evaluation + final model               (~15 s)
.venv/bin/python app/precompute.py      # data for the app's slide tabs                                (~30 s)
.venv/bin/streamlit run app/streamlit_app.py
./app/present.sh                        # optional: open the app full screen in a Chrome window (presentation)
.venv/bin/pytest
```

Optional: set `CENSUS_API_KEY` (free) to add ACS 65+ and disability shares to the area features.

## Repository layout

```
src/heattents/
  ingest/        downloaders, one module per source (ems, geography, context, weather, activity, demographics)
  features/      static area table, heat features, calendar, area × day panel
  models/        glm.py (IRLS, splines, DLNM cross-basis), city.py, context_share.py (share model), scenarios.py,
                 explain.py, diagnostics.py
  deploy/        daily.py (daily field-hospital MILP, Gurobi), compare.py (plan vs rule-based placements),
                 candidates.py (parks); model.py / milp.py: the earlier hourly tent & doctor model (not used by the app)
  service.py     "a day" -> forecasts, maps and optimiser input (used by the app and the evaluation)
  eda.py         exploratory analysis of the label
  cli.py
app/             Streamlit app: forecast & plan, predictive model, MILP model and results tabs (see app/README.md)
data/raw/        downloaded inputs          data/processed/  features
outputs/         eda/, models/city/, models/share/, models/deploy_compare/, app/
tests/           GLM/spline/conformal, city model, deployment MILP
archive/london_v1/   the earlier London prototype (superseded)
```

---

## Data

| Input | Content | Source |
|---|---|---|
| **Label** | EMS incidents with final call type HEAT: timestamp, ZIP, borough (2005–2026) | NYC Open Data, EMS Incident Dispatch Data (`76xm-jjuj`) |
| Second label | Citywide daily heat-syndrome ED visits (100% of NYC EDs, 2017–2026) | NYC Health, Environment & Health Data Portal |
| Spatial unit | 177 MODZCTA (modified ZIP code tabulation areas) | NYC Open Data (`pri4-ifjk`) |
| Weather | Daily and hourly temperature, apparent temperature, humidity, radiation (ERA5 / ERA5-Land) | Open-Meteo archive |
| **Upstream cities** | Newark, Trenton, Philadelphia, Baltimore, Washington DC, daily | Open-Meteo archive |
| **Real forecasts** | Archived 1- and 3-day-ahead forecasts at Central Park (2022–2026) | Open-Meteo Previous Runs |
| Urban heat & green | Daytime summer surface temperature, vegetative cover, tree canopy, household AC | Environment & Health Data Portal |
| Vulnerability | Poverty, older adults living alone, race, Heat Vulnerability Index | Environment & Health Data Portal, NYC Open Data |
| Daytime presence | Workplace jobs by ZCTA; summer subway entries by station | LEHD LODES 8; MTA monthly ridership |
| Parks | NYC Parks properties (park share; tent candidate sites) | NYC Open Data (`enfh-gkve`) |
| Existing cooling | Cool It! NYC cooling sites | NYC Open Data (`h2bn-gu9k`) |

Context indicators published for neighbourhoods (NTA, UHF) are moved to MODZCTA by area-weighted polygon overlay.

## What the label tells us ([outputs/eda/findings.md](outputs/eda/findings.md))

- **Not a coding change.** The 2025–26 rise is real: the independent ED label rose in step, and the EMS/ED ratio is stable.
- **Peaks are censored.** On 2026 peak days EMS was saturated (53% of calls held), so EMS counts under-state true demand on exactly the days that matter.
- **Not event-driven.** The 2–4 July 2026 peak followed the heat (108 °F), not the US-250th celebrations.
- **Sparse.** 97.6% of area-days have zero calls. The overdispersion is temporal: the citywide total ranges from 0 to 231.
- **Stable hour profile.** Calls peak at 14:00, with 62% between 12:00 and 18:00. On very hot days (≥ 36 °C) they spread a little more into the morning and evening (60% vs 67% on mild days).

---

## Models

### 1. Citywide probabilistic forecast ([models/city.py](src/heattents/models/city.py))

**Full documentation, with every input, parameter, the pipeline, all metrics and references: [docs/city_model.md](docs/city_model.md).**

In short:
- **When:** at 18:00 on day *d−1* the model forecasts the distribution of tomorrow's citywide heat EMS calls.
- **Inputs:** tomorrow's archived weather forecast; the last three days' observed weather; heatwave indicators (EHF, hot-day streak); the south-west upstream heat gradient; calls before 18:00 yesterday; calendar.
- **Models:** the average of a negative binomial regression with a distributed-lag heat term (33 coefficients) and a monotone LightGBM.
- **Weather uncertainty:** 30 Monte-Carlo draws from real analog forecast errors.
- **Test:** trained on all earlier summers and tested season by season on 2024, 2025 and 2026.

| Model | CRPS ↓ | Mean abs. error | 90% coverage |
|---|---|---|---|
| **Ensemble (used)** | **2.32** | **3.32** | 0.96 |
| Temperature only | 2.97 | 4.37 | 0.90 |
| Persistence | 4.02 | 5.68 | 0.86 |
| Climatology | 5.48 | 6.75 | 0.94 |

![Heat curve](outputs/models/city/heat_curve.png)

### 2. Beach & event share model ([models/context_share.py](src/heattents/models/context_share.py))

**Full documentation of both forecasting models (inputs, parameters, pipeline, runtimes, why GLM + LightGBM): [docs/forecast_model.md](docs/forecast_model.md).**

The city model gives *how many* calls; this model gives each of the 177 ZIP areas' *share* of them. Every area has a
usual share (it follows daytime presence and barely changes between years). The model changes it only on days the
evening-before information says are different:

- **Weekends and holidays:** beach and park areas gain; office districts lose.
- **Beach season:** beach areas ×1.6, ×1.5 more on weekends.
- **Recurring events on their routes:** Pride ×6.1, Puerto Rican Day ×4.1, West Indian Day ×15.9, Dominican Day, Mermaid Parade, US Open.

$$
\log \pi_{i,d} = a_i + c(d) + \sum_k \beta_k Z_{ik}\,\text{leisure}_d + \text{beach}_i\,\text{open}_d\,(g_1 + g_2\,\text{leisure}_d + g_3\,\text{heat}_d + g_4\,\text{leisure}_d\text{heat}_d) + \sum_e \gamma_e\,\mathbb 1[e \text{ on } d]\,\text{route}_{ie} + u_i\,\text{leisure}_d
$$

- **Fitting:** a Poisson–multinomial GLM (380 parameters) fitted by penalised IRLS. Recent summers are weighted more (half-life 8 years).
- **Speed:** the 594k area-days collapse exactly to 7,788 cells, so one fit takes 0.3 s.
- **Heat is not used to move the map** (except at beaches): hot days bring more calls from the same places.

Out of sample, 2018–26 ([outputs/models/share/](outputs/models/share/)):

| | Result |
|---|---|
| Log-likelihood gain over the historical map | +0.013 nats per call; **+0.111 on event days**; +0.032 on weekends |
| Predicted moves come true | forecast ×3.00 → observed ×3.23 for the largest rises; ×0.53 → ×0.62 for the largest drops |
| Daily ratio error, ZIP level (0 = identical maps) | 0.819, against 0.824 for the historical map and 0.817 for a perfect model |

Most day-to-day scatter between areas is chance: within a summer it is no larger than a multinomial draw, and recent
local calls do not predict tomorrow's. So on an ordinary weekday the forecast is the usual map. Alternatives were
tested and not kept, because none beat this model out of sample:

- a LightGBM share model
- heat × vulnerability terms
- neighbourhood temperature (HRRR)
- direct and panel neighbourhood count models

### 3. Prediction output: daily demand per neighbourhood in 5 scenarios ([models/scenarios.py](src/heattents/models/scenarios.py))

$$
D^s_u = N^s \cdot m \cdot \pi_u, \qquad 42 \text{ neighbourhoods} \times 5 \text{ scenarios}, \quad P(s) = 0.2
$$

- $N^s$ are the **10th, 30th, 50th, 70th and 90th percentiles** of the city model's predictive distribution: five equally likely versions of tomorrow, from quiet to busy. Each stands for one fifth of the distribution.
- $m$ is the share of calls with a known location (0.96).
- $\pi_u$ is the neighbourhood's share: the share model's 177 ZIP-area shares summed to the 42 UHF neighbourhoods.
- The output is **daily** (no hourly split), always has exactly 5 scenarios, and is in **people needing help = calls × 20** (converted once, right after the forecast). Columns: `uhf, scenario, probability, expected_cases`.

Full specification with an example: [docs/forecast_model.md § Output](docs/forecast_model.md#output-what-the-forecast-hands-to-the-optimiser).

### 4. Daily field-hospital placement: maximum-coverage MILP ([deploy/daily.py](src/heattents/deploy/daily.py))

Active planning rules (`app/data_layer.py`, `ACTIVE_RULES = "threshold"`): a hospital helps everyone within 2 km (no
worker limit), hospitals can stand anywhere on land, up to 3 per neighbourhood.

| Symbol | Meaning |
|---|---|
| $a$ | Demand points: up to 4 interior points per ZIP area (708), each carrying part of the area's forecast |
| $j$ | Candidate sites: a 400 m grid over all land plus the best NYC parks and plazas per neighbourhood |
| $s$ | The 5 forecast scenarios, $p_s = 0.2$ |
| $D_{as}$ | People needing help at point $a$ in scenario $s$ |
| $N$, $Q$ | Number of hospitals (input); at most $Q = 3$ per neighbourhood |
| $R(j)$, $J_g$ | Points within 2 km of site $j$; sites in neighbourhood $g$ |
| $x_j \in \{0,1\}$ | A hospital opens at site $j$ |
| $u_a \in [0,1]$ | Point $a$ is within reach of an open hospital (takes 0 or 1 at the optimum) |

$$
\begin{aligned}
\max\ & \textstyle\sum_a \mathbb E[D_a]\, u_a, \qquad \mathbb E[D_a] = \sum_s p_s D_{as}\\
\text{s.t.}\ & \textstyle\sum_j x_j \le N \qquad\qquad\qquad\qquad\ \text{(hospital budget)}\\
& \textstyle\sum_{j \in J_g} x_j \le Q \quad \forall g \qquad\qquad\ \ \text{(at most 3 per neighbourhood)}\\
& u_a \le \textstyle\sum_{j:\, a \in R(j)} x_j \quad \forall a \qquad\ \ \text{(helped only if a hospital is within 2 km)}\\
& x_j \in \{0,1\},\ 0 \le u_a \le 1
\end{aligned}
$$

- $u_a \le 1$ means a person reached by two hospitals is counted once. There is no capacity: a hospital covers everyone in its circle.
- On ties, a second step prefers hospitals with the most people within reach. Solved with Gurobi in about a second.
- `ACTIVE_RULES = "previous"` switches back to the earlier **two-stage stochastic** version: parks only, a 15-minute walk, 4 health workers per hospital at 30 people per worker per day, one per neighbourhood. People are assigned to hospitals per scenario, and unhelped people are minimised.

**Does it beat simple rules?** [deploy/compare.py](src/heattents/deploy/compare.py) scores each plan on the people who
*actually* needed help that day. The forecasts are out of sample: the city and share models are refitted on earlier
summers only. All plans get the same number of hospitals and the same rules. The population and heat-vulnerability
rules place one hospital in each of the top N neighbourhoods, at the site there with the most residents within reach.

| 10 hospitals, 2 km (`run_coverage`) | Fri 21 Jun 2024 (1,100 needed help) | Pride Sun 29 Jun 2025 (740) |
|---|---|---|
| Population rule | 315 (29%) | 140 (19%) |
| Heat-vulnerability rule | 445 (40%) | 115 (16%) |
| Historical-calls rule | 480 (44%) | 400 (54%) |
| **Forecast + MILP** | **595 (54%)** | **430 (58%)** |
| Best possible (MILP on what actually happened) | 745 (68%) | 560 (76%) |

`run()` repeats the comparison under the previous (capacity) rules on all 59 days of 2024–26 with a forecast heat
index ≥ 35 °C.

The **app** ([app/](app/)) runs this pipeline for any day of May–September 2024–26 (forecast as issued the evening
before) and presents the models as slides; see [app/README.md](app/README.md).

---

## Limitations

- EMS calls are a proxy for heat illness. Peak days are censored by EMS saturation, and calls without a location (3.7%) count only towards the city total.
- "Helped" means being within 2 km of a field hospital. The medical effect of a visit, people's willingness to travel, and the ×20 people per call are assumptions. EMS calls give a ZIP code only, so people are placed at up to 4 points per ZIP area.
- The share model is trained on where calls were *made*; a deployed tent may itself shift behaviour.
- Within-city weather differences come from static surface-temperature data. Daily borough ERA5-Land series are supported but were not available within the Open-Meteo quota at build time.
- The active model ignores capacity (no worker limit); the "previous" rules model it. Placement is solved one day at a time; multi-day planning (moving hospitals, staff fatigue) is not modelled.
