# NYC Heat Response: probabilistic heat-illness forecasting and daily tent & doctor deployment

On a heatwave morning a New York City planner has a handful of pop-up cooling tents and a roster of doctors with shift
availability. **Where should the tents go today, and who should staff which tent in which shift, so that as many
heat-illness cases as possible are treated?**

This project answers that question end to end, from public data:

```
EMS heat calls (label) ─┐
weather + forecasts ────┼─► [1] citywide probabilistic forecast ──┐
upstream cities ────────┘     (DLNM-NB + LightGBM, forecast-error MC)│
                                                                      ├─► [3] coherent demand scenarios ─► [4] Gurobi MILP ─► app
area features ──────────────► [2] area share model (LightGBM) ───────┘     (city × area × hour)        (tents, doctors, shifts)
(presence, heat, green, vulnerability)
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
| **Where** | The spatial pattern is extremely stable (area rank correlation 0.94 between 2005–15 and 2016–26) and is driven by *daytime presence* (Times Square, Penn Station, Coney Island), not by resident vulnerability (correlation with HVI: 0.005). Weather moves *how many* calls happen far more than *where*. |
| **Decision** | 10 tents and 22 doctors on 2 July 2026 reach **~20%** of the day's expected cases within a 1 km walk. Capacity is not the bottleneck on that day; reach is. The Gurobi model solves to <1% gap in ~1 minute. |

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
brew install libomp                      # macOS only: OpenMP runtime for LightGBM
# Gurobi: a full licence is needed (models have ~50k variables); e.g. a free academic WLS licence in ~/gurobi.lic

.venv/bin/heattents ingest              # download all inputs (cached; Open-Meteo has a daily quota, re-run to resume)
.venv/bin/heattents features            # area table, citywide series, area × day panel
.venv/bin/heattents city                # citywide model: 2024-26 evaluation, final model, heat curve  (~3 min)
.venv/bin/heattents area                # area share model: evaluation + final model                 (~3 min)
.venv/bin/streamlit run app/streamlit_app.py
.venv/bin/pytest
```

Optional: set `CENSUS_API_KEY` (free) to add ACS 65+ and disability shares to the area features.

## Repository layout

```
src/heattents/
  ingest/        downloaders, one module per source (ems, geography, context, weather, activity, demographics)
  features/      static area table, heat features, calendar, area × day panel
  models/        glm.py (IRLS, splines, DLNM cross-basis), city.py, area.py, scenarios.py
  deploy/        model.py (request/plan dataclasses), milp.py (Gurobi), candidates.py
  service.py     "a day" -> forecasts, maps and optimiser input (used by the app and the evaluation)
  eda.py         exploratory analysis of the label
  cli.py
app/             Streamlit decision-support app (data_layer.py is its only access to models and solver)
data/raw/        downloaded inputs          data/processed/  features
outputs/         eda/, models/city/, models/area/
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
- **Stable hour profile.** It peaks at 14:00 and does not depend on temperature.

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

### 2. Area share model ([models/area.py](src/heattents/models/area.py))

Given the city total, calls are allocated multinomially, $y_{it}\mid N_t \sim \mathrm{Mult}(N_t, \pi_t)$, with
$\pi_{it} \propto \exp g(x_i, w_t)$. This is fitted as a Poisson LightGBM with offset $\log N_t$ (the Poisson–multinomial equivalence).

- $x_i$ are area features: daytime presence (residents + jobs + subway entries), density, surface temperature, vegetation, tree canopy, parks, poverty, older adults living alone, HVI, plus the area id.
- $w_t$ is the day's heat and calendar.

At forecast time the shares multiply the citywide predictive distribution. This is *top-down reconciliation with forecast proportions* (Athanasopoulos, Ahmed & Hyndman, 2009): area forecasts always sum to the city forecast.

Out-of-sample, mean over seasons 2018–2026 ([area_scores_summary.csv](outputs/models/area/area_scores_summary.csv)):

| Allocation | Log-likelihood per call | Calls in predicted top-10% areas |
|---|---|---|
| LightGBM share model | −4.901 | 27.4% |
| Historical share | −4.903 | 27.4% |
| Daytime presence share | −5.027 | 20.6% |
| Resident population share | −5.173 | 13.9% |
| *Oracle: the test season's own realised shares (not a forecast)* | *−4.793* | *30.4%* |

The oracle shows how little room there is: even knowing the season's realised map, only 30% of calls fall in the
top-10% areas, because individual calls are Poisson noise around a stable pattern. The model reaches ~90% of
that ceiling. It matches the historical map and clearly beats population-based allocation. **Daytime presence is
a much better exposure measure than residence.** The honest conclusion: the *where* is persistent, and the predictable
signal is in the *when / how many*. That is why the deployment decision depends mostly on the citywide forecast and on
reach.

### 3. Demand scenarios ([models/scenarios.py](src/heattents/models/scenarios.py))

$$
D^s_{a,h} = N^s_t \cdot m \cdot \pi_{i(a),t}\, w_a \cdot \phi_{b(i),\text{weekend},h}
$$

- $N^s_t$ are **stratified quantiles** of the predictive distribution (a low-variance SAA design; the scenarios differ in level, not pattern).
- $m$ is the share of calls with a mappable location.
- $\phi$ is the empirical hour profile by borough and weekend.
- Each area's demand is spread over up to four interior points $a$ (weights $w_a$), so that 1 km walking catchments are realistic.

### 4. Daily deployment: two-stage stochastic MILP ([deploy/milp.py](src/heattents/deploy/milp.py))

**Sets**

| Symbol | Meaning |
|---|---|
| $J$ | Candidate sites (NYC parks and plazas, screened to the 250 reaching most expected demand) |
| $K$ | Doctors |
| $S$ | Shifts |
| $T$ | Time blocks aligned to shift boundaries |
| $A$ | Demand points |
| $W$ | Scenarios |
| $R$ | Walkable pairs $(a,j)$ within the radius |

**Variables**

- $x_j\in\{0,1\}$: tent $j$ is set up.
- $o_{js}\in\{0,1\}$: tent $j$ operates in shift $s$.
- $y_{js}\in\mathbb Z_+$: doctors at tent $j$ in shift $s$.
- $v_{ks}\in\{0,1\}$: doctor $k$ works shift $s$.
- $f_{wajt}\ge 0$: cases from point $a$ treated at tent $j$ in block $t$ under scenario $w$.

**Formulation**

$$
\begin{aligned}
\max\ & \tfrac{1}{|W|}\textstyle\sum_{w,(a,j),t} \rho_{aj}\, f_{wajt} \;-\; \varepsilon\big(\sum_j x_j + \sum_{k,s} v_{ks}\big) \qquad \rho_{aj}=e^{-d_{aj}/0.6\,\mathrm{km}}\\
\text{s.t.}\ & \textstyle\sum_j x_j \le N_{\text{tents}};\quad o_{js}\le x_j;\quad \sum_s o_{js}\ge x_j\\
& m_{\min}\, o_{js} \le y_{js} \le M_j\, o_{js};\qquad \textstyle\sum_j y_{js} = \sum_k v_{ks}\ \ \forall s\\
& \textstyle\sum_s v_{ks} \le L_k;\quad v_{ks}+v_{ks'}\le 1 \text{ for overlapping } s,s';\quad v_{ks}=0 \text{ if } k \text{ unavailable}\\
& \textstyle\sum_j f_{wajt} \le D_{wat} \qquad\qquad\qquad\qquad\ \ \forall w,a,t \quad\text{(demand)}\\
& \textstyle\sum_a f_{wajt} \le \mu \sum_s c_{st}\, y_{js} \qquad\qquad\ \forall w,j,t \quad\text{(doctor capacity)}\\
& f_{wajt} \le D_{wat} \textstyle\sum_{s:\,c_{st}>0} o_{js} \qquad\quad\ \forall w,(a,j),t \quad\text{(strong linking)}
\end{aligned}
$$

- Doctors differ only by availability. The model therefore places *numbers* of doctors and recovers the named roster exactly, which keeps the binaries at $|K||S|+2|J||S|+|J|$ instead of $|K||J||S|$.
- The **strong-linking** constraints (the classic strong facility-location formulation) cut the optimality gap on a real instance from 27% to 0.4%.
- KPIs report both the distance-weighted objective and the **unweighted expected cases treated**.

The **app** ([app/](app/)) runs exactly this pipeline:
1. Pick a historical date (2024–26, forecast as issued the evening before), or type a scenario (heat index, day in the spell, weekend/holiday, upstream signal, yesterday's calls before 18:00).
2. See the citywide forecast with intervals and a per-area risk map by shift.
3. Enter tents, doctors and shifts, and let Gurobi return the plan, the roster and the unserved map.

---

## Limitations

- EMS calls are a proxy for heat illness. Peak days are censored by EMS saturation, and calls without a location (3.7%) count only towards the city total.
- "Treated at a tent" means a case occurring within walking distance that tent capacity can absorb. The medical effect of a visit, and people's willingness to walk (the decay parameter), are assumptions.
- The area model is trained on where calls were *made*; a deployed tent may itself shift behaviour.
- Within-city weather differences come from static surface-temperature data. Daily borough ERA5-Land series are supported but were not available within the Open-Meteo quota at build time.
- The deployment model is solved for one day; multi-day planning (moving tents, crew fatigue) is not modelled.
