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
| **Citywide forecast** | Out-of-sample (2022–26), the model beats climatology by **37–57%** and a temperature-only GLM by **16–31%** in CRPS, under every information set (observed weather, real 1-day and 3-day-ahead forecasts). It wins in all 7 test seasons. |
| **Calibration** | 90% intervals cover 97–98% of days; PIT histograms are close to flat. Predictions correlate **0.84–0.90** with an *independent* label (citywide syndromic heat ED visits). |
| **Record day** | 2 July 2026 had 231 calls. The 1-day-ahead forecast was 199, out of sample. |
| **Where** | The spatial pattern is extremely stable (area rank correlation 0.94 between 2005–15 and 2016–26) and is driven by *daytime presence* (Times Square, Penn Station, Coney Island), not by resident vulnerability (correlation with HVI: 0.005). Weather moves *how many* calls happen far more than *where*. |
| **Decision** | 10 tents and 22 doctors on 2 July 2026 reach **~20%** of the day's expected cases within a 1 km walk. Capacity is not the bottleneck on that day; reach is. The Gurobi model solves to <1% gap in ~1 minute. |
| **Value of forecasting** | See [Decision-focused evaluation](#decision-focused-evaluation). |

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
brew install libomp                      # macOS only: OpenMP runtime for LightGBM
# Gurobi: a full licence is needed (models have ~50k variables); e.g. a free academic WLS licence in ~/gurobi.lic

.venv/bin/heattents ingest              # download all inputs (cached; Open-Meteo has a daily quota, re-run to resume)
.venv/bin/heattents features            # area table, citywide series, area × day panel
.venv/bin/heattents city                # citywide model: rolling-origin evaluation + final models   (~70 s)
.venv/bin/heattents area                # area share model: evaluation + final model                 (~3 min)
.venv/bin/heattents decisions           # decision-focused evaluation through Gurobi
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
  evaluate_decisions.py   decision-focused evaluation
  cli.py
app/             Streamlit decision-support app (data_layer.py is its only access to models and solver)
data/raw/        downloaded inputs          data/processed/  features
outputs/         eda/, models/city/, models/area/, decision_eval/
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

The daily citywide count $N_t$ follows a negative binomial with a distributed-lag non-linear model (DLNM), the
standard in heat–health epidemiology:

$$
N_t \sim \mathrm{NB2}(\mu_t, \alpha),\qquad
\log \mu_t = \underbrace{\textstyle\sum_{l=0}^{3} \mathbf b(T^{app}_{t-l})^\top \mathbf C\, \mathbf q(l)}_{\text{cross-basis}}
+ \beta_1 \mathrm{EHF}_t + \beta_2\,\text{streak}_t + \beta_3\,\text{season heat}_t + \beta_4\,\text{tropical night}_t
+ \beta_5 (\Delta^{SW}_{t-1})^+ + \text{calendar} + s(\text{year}) + s(\text{day of year}) \,[+\ \gamma \log(1+N_{t-1})]
$$

- $\mathbf b(\cdot)$ is a natural cubic spline in the maximum apparent temperature (knots 26/31/35 °C), and $\mathbf q(l)$ is a quadratic lag basis over 0–3 days.
- **EHF** is the Excess Heat Factor (Nairn & Fawcett): heat relative to both climatology and the previous 30 days.
- $\Delta^{SW}$ is the **upstream gradient**: yesterday's apparent max in Philadelphia, Baltimore and DC minus NYC, i.e. heat on its way.
- The lagged count is used only when it is known (1-day-ahead mode).

A monotone-constrained **LightGBM** (Poisson) uses the same features. The **ensemble** averages the two means: the GLM extrapolates to record heat, while the GBM fits ordinary days better.

**Information sets.** Every forecast is built only from what was known when it would have been issued:

| Mode | Target day weather | Earlier days |
|---|---|---|
| observed | observed (hindcast ceiling) | observed |
| lead1 | archived 1-day-ahead forecast | observed; yesterday's calls known |
| lead3 | archived 3-day-ahead forecast | days d−1, d−2 also forecast; upstream signal as of d−3 |

**Uncertainty.** The predictive distribution is a mixture of NBs over 30 Monte-Carlo weather draws, built by resampling *real* archived forecast errors (test season held out). Split-conformal PIT recalibration (Kuleshov et al., 2018) was implemented. It did **not** improve CRPS out of sample, because the raw model is already calibrated and older seasons are non-stationary, so it is off by default.

**Out-of-sample results**, rolling origin with each season trained on all earlier ones, mean over 2022–26 ([metrics_summary_2022_2026.csv](outputs/models/city/metrics_summary_2022_2026.csv)):

| Mode | Model | CRPS | twCRPS (≥50) | Log score | 90% coverage | Corr. with ED |
|---|---|---|---|---|---|---|
| observed | **ensemble** | **1.87** | **0.39** | −1.93 | 0.97 | 0.90 |
| observed | temperature-only GLM | 2.69 | 0.67 | −3.51 | 0.81 | 0.83 |
| lead1 | **LightGBM** | **2.53** | **0.58** | −2.06 | 0.97 | 0.89 |
| lead1 | persistence | 3.24 | 0.60 | −3.43 | 0.87 | 0.79 |
| lead3 | **ensemble** | **2.79** | **0.72** | −2.07 | 0.98 | 0.84 |
| lead3 | persistence | 5.30 | 1.28 | −3.00 | 0.84 | 0.29 |
| any | climatology | 4.39 | 0.96 | −2.54 | 0.95 | 0.37 |

Models are selected by the threshold-weighted CRPS on days ≥ 50 calls, because those are the days that drive decisions.

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
1. Pick a historical date under an information set, or type a scenario (heat index, day in the spell, weekend/holiday, upstream signal, yesterday's calls).
2. See the citywide forecast with intervals and a per-area risk map by shift.
3. Enter tents, doctors and shifts, and let Gurobi return the plan, the roster and the unserved map.

---

## Decision-focused evaluation

_Results of `heattents decisions` (outputs/decision_eval/) are added here._

---

## Limitations

- EMS calls are a proxy for heat illness. Peak days are censored by EMS saturation, and calls without a location (3.7%) count only towards the city total.
- "Treated at a tent" means a case occurring within walking distance that tent capacity can absorb. The medical effect of a visit, and people's willingness to walk (the decay parameter), are assumptions.
- The area model is trained on where calls were *made*; a deployed tent may itself shift behaviour.
- Within-city weather differences come from static surface-temperature data. Daily borough ERA5-Land series are supported but were not available within the Open-Meteo quota at build time.
- The deployment model is solved for one day; multi-day planning (moving tents, crew fatigue) is not modelled.
