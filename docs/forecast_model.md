# Forecasting model: how many heat calls tomorrow, and where

**Question.** At 18:00 on day *d−1*, how many heat-related EMS calls will New York City have on day *d*, and how will
they be spread over the city's 177 ZIP areas?

**Answer.** Two models in sequence:

1. **City model**: how many calls citywide. A probability distribution for tomorrow's total.
2. **Beach & event share model**: where they land. Each ZIP area's share of tomorrow's calls.

Expected calls in area *i* = city forecast × *m* × share of *i*, where *m* = 0.96 is the share of calls that have a
known location. The hour-of-day profile then turns this into demand scenarios for the Gurobi tent-and-doctor
optimiser.

```
18:00 on d-1
 weather forecast for d ─┐
 last 3 days' weather ───┼─► CITY MODEL ──► P(total calls on d)  ──┐
 upstream heat, calendar ┤   GLM + LightGBM, 30 weather draws      │
 calls before 18:00 ─────┘                                         ├─► expected calls per area ─► hour profile ─► Gurobi
                                                                   │   (city total × 0.96 × share)
 weekend / holiday ──────┐                                         │
 beach season, heat band ┼─► SHARE MODEL ──► share of each area ───┘
 event calendar ─────────┘   Poisson-multinomial GLM
```

Code: [city.py](../src/heattents/models/city.py), [context_share.py](../src/heattents/models/context_share.py),
[scenarios.py](../src/heattents/models/scenarios.py), [service.py](../src/heattents/service.py).
Full technical documentation of the city model: [city_model.md](city_model.md).

---

## 1. Data and time frames

| Item | Value |
|---|---|
| Label | NYC EMS incidents with final call type HEAT (NYC Open Data `76xm-jjuj`) |
| Season | 1 May – 30 September; 97.7% of heat calls fall in these months |
| Period | 1 May 2005 – 21 Sep 2026: **3,357 days** |
| Spatial unit | 177 ZIP areas (MODZCTA): **594,189 area-days** |
| Calls | 17,278 in season; 16,642 with a known location (96%) |
| Weather for training | Observed (ERA5 via Open-Meteo, Central Park and upstream cities), 2005–2026, complete |
| Weather for testing and daily use | Archived 1-day-ahead forecast issued the evening before (Open-Meteo Previous Runs, from 2022; humidity from 2024) |

| | Training | Test (out of sample) | Final model |
|---|---|---|---|
| **City model** | All summers before the test summer | 2024, 2025, 2026: each trained on 2005 to the year before; **450 days** | 2005–2026 |
| **Share model** | All summers before the test summer | 2018, 2019, 2022–2026 (2020–21 skipped as COVID years); 6,661 calls | 2005–2026 |

Every test day is forecast exactly as it would have been at 18:00 the evening before: the weather forecast as issued,
and calls recorded before 18:00. Training uses each day's observed weather ("perfect prognosis"); the uncertainty of
the weather forecast enters only at forecast time, through 30 Monte-Carlo draws.

---

## 2. City model (how many)

### 2.1 Inputs (all known at 18:00 on *d−1*)

| Group | Inputs |
|---|---|
| Heat | Forecast max heat index for *d*; observed max heat index on *d−1*, *d−2*, *d−3*; max / min / mean temperature, humidity |
| Accumulated heat | Excess Heat Factor (EHF) and its two parts, hot-day streak, hot days so far this season, tropical night (min ≥ 24 °C) |
| Regional | Yesterday's heat index in Philadelphia / Baltimore / Washington DC minus NYC (is heat on its way?) |
| Recent calls | log(1 + citywide heat calls before 18:00 on *d−1*) |
| Calendar and trend | Weekend, federal holiday, 3–5 July, school break, COVID years, year, day of year |

### 2.2 Two models, averaged

| | GLM: negative binomial regression | LightGBM: gradient-boosted trees |
|---|---|---|
| Form | log E[calls] = intercept + heat × lag surface (DLNM) + 12 linear terms + year and season splines | 500 trees, Poisson objective |
| Heat × lag | Natural spline in heat index (knots 26 / 31 / 35 °C, linear beyond) × quadratic over lags 0–3: 12 coefficients | Learnt by the trees |
| Size | **33 coefficients** + dispersion α = 0.2 | 23 inputs, 500 trees × 15 leaves |
| Fitting | Penalised IRLS, ridge 0.3; α by profile likelihood | Learning rate 0.03, min. 20 days per leaf, 0.8 row and feature sampling, L2 = 1 |
| Constraints | — | Monotone increasing in heat index today and yesterday, max temperature, EHF, streak |

**Ensemble:** expected calls = ½ GLM + ½ LightGBM. The weights were fixed before looking at test results.

**Uncertainty:** tomorrow's weather is redrawn 30 times as *forecast + a real past forecast error*. The errors are
analogs: past forecasts of similar heat, already verified by the issue time. Count noise is negative binomial with
α = 0.2. The forecast is the mixture of the 30 resulting distributions.

### 2.3 Why both a GLM and gradient-boosted trees

| | GLM (DLNM negative binomial) | LightGBM |
|---|---|---|
| **Strength** | Smooth, physically shaped heat response that **extrapolates sensibly** past the hottest training days (linear in log-calls beyond 35 °C); proper count model whose dispersion gives the prediction intervals; 33 interpretable coefficients, stable with few extreme days | Finds **interactions** without specifying them (heat × humidity × streak × weekend × time of season); fits ordinary days better; monotone constraints keep it physically sensible |
| **Weakness** | Only the interactions we write into it | **Cannot extrapolate**: beyond the hottest day it has seen, a tree's prediction is flat, so it under-forecasts record heat |
| **Test result alone** (CRPS / mean abs. error) | 2.57 / 3.96 | 2.44 / 3.56 |

Their errors lean in opposite directions: over 2024–26 the GLM forecast 3,912 calls and LightGBM 3,443, against 3,541
actual. Averaging cancels part of both (ensemble 3,677):

| 2024–26, 450 test days | CRPS ↓ | twCRPS ≥ 50 calls ↓ | Mean abs. error ↓ | 90% coverage |
|---|---|---|---|---|
| **Ensemble (used)** | **2.32** | **0.59** | **3.32** | 0.96 |
| LightGBM alone | 2.44 | 0.75 | 3.56 | 0.97 |
| GLM alone | 2.57 | 0.72 | 3.96 | 0.95 |
| Temperature-only GLM | 2.97 | 0.80 | 4.37 | 0.90 |
| Climatology | 5.48 | 1.59 | 6.75 | 0.94 |

The ensemble wins on every score, most clearly on high-call days (twCRPS ≥ 50), which are the days that matter for
deployment.

### 2.4 City model results (out of sample)

- **Daily error:** median 1 call, mean 3.3 calls; correlation with actual 0.93; 90% intervals contain the actual on
  96% of days.
- **Busy days:** 6 of the 8 days with ≥ 50 calls flagged the evening before. On 2 July 2026 (231 calls) the forecast
  was 211.
- **Validation:** forecasts correlate 0.89 with an independent label (citywide heat ED visits).

Details, per-season results and references: [city_model.md](city_model.md).

---

## 3. Beach & event share model (where)

### 3.1 The idea in plain words

Every ZIP area has a **usual share** of the city's heat calls. It follows where people spend their day: Times Square
gets about 2.2%, Coney Island about 1.3%. These shares are very stable from year to year (rank correlation 0.94
between 2005–15 and 2016–26).

The model starts from that usual share and changes it only when the evening-before information says the day is
different:

- **Weekends and holidays:** beach and park areas gain; office districts lose.
- **Beach season:** beach areas gain from Memorial Day weekend to Labor Day, most on weekends.
- **Recurring events:** Pride, the Puerto Rican Day, Dominican Day and West Indian Day parades, the Mermaid Parade and the US Open have fixed dates and routes. On those days the route areas take several times their usual share.

The shares are then rescaled to add up to 100%.

**Heat is deliberately not used to move the map**, except at beaches, where very hot days slightly reduce the beach
boost. In this data, hot days bring **more** calls from the **same** places.

### 3.2 Model

$$
y_{i,d} \mid N_d \sim \text{Multinomial}(N_d, \pi_d), \qquad \text{fitted as } y_{i,d} \sim \text{Poisson}(N_d\,\pi_{i,d})
$$

$$
\log \pi_{i,d} = a_i + c(d)
+ \sum_k \beta_k Z_{ik}\,\text{leisure}_d
+ \text{beach}_i\,\text{open}_d\,(g_1 + g_2\,\text{leisure}_d + g_3\,\text{heat}_d + g_4\,\text{leisure}_d\,\text{heat}_d)
+ \sum_e \gamma_e\,\mathbb 1[e \text{ on } d]\,\text{route}_{ie}
+ u_i\,\text{leisure}_d
$$

normalised so the 177 shares sum to 1 each day. *N_d* is the day's total; the Poisson–multinomial equivalence means
fitting a Poisson model with offset log *N_d* learns only the **shares**, never the total.

### 3.3 Parameters

| Term | Count | Final values (share multiplier) | Meaning |
|---|---|---|---|
| Usual share *a_i* | 177 | 0.02% – 2.2% | Each area's normal share; recent summers weighted more (half-life 8 years) |
| Day-type constants *c(d)* | 13 | – | Make shares sum to 100% on each type of day |
| Weekend × traits *β* | 3 | beach ×1.43, visitor ratio ×1.00, park share ×1.01 | Weekend shift by area type (per standard deviation; beach is 0/1) |
| Beach season *g* | 4 | ×1.58 in season, ×1.51 more on weekends; ×0.86 and ×0.70 per heat band | Open-beach effect, trimmed on the hottest days |
| Events on route *γ* | 6 | Pride ×6.1, Puerto Rican Day ×4.1, West Indian Day ×15.9, Dominican Day ×1.8, Mermaid Parade ×1.5, US Open ×1.8 | Route areas on the event day |
| Own weekend deviation *u_i* | 177 | ridge = 100 | Lets an area differ from its type on weekends; pulled towards 0 |
| **Total** | **380** | | |

**Inputs:** past calls per area; the day type (weekend/holiday, beach season, heat band from the citywide forecast,
event); static area traits (beach flag, visitor ratio = (jobs + subway entries) ÷ residents, park share).

**Fitting:** penalised IRLS, the same solver as the city GLM ([glm.py](../src/heattents/models/glm.py)). Light ridge on
the shared terms, 2 on event effects, and on *u_i* a strength chosen on the last training summer from
{3, 10, 30, 100}. Every day input is categorical, so the 594,189 area-days collapse **exactly** (Poisson sufficiency)
to **7,788 cells** (area × day type), and one fit takes 0.3 s.

**Why a GLM and no tree model here:** the spatial signal is weak (see 3.5). A LightGBM share model was tested first and
reproduced the historical map almost exactly (gain 0.002 nats per call against this model's 0.013). With few
informative days (a parade happens once a year), a regularised GLM with explicit, interpretable terms generalises
better and its moves can be checked one by one.

### 3.4 Results (out of sample, 2018–2026)

Log-likelihood per call, compared with each area's fixed historical share (higher is better):

| Days | Calls | Gain (nats per call) |
|---|---|---|
| All days | 6,661 | **+0.013** |
| Weekends and holidays | 2,280 | +0.032 |
| Hot beach weekends | 1,345 | +0.012 |
| **Event days** | 597 | **+0.111** |
| Extreme heat (≥ 38 °C) | 1,908 | −0.002 |
| Ordinary weekdays | 4,102 | +0.003 |

**The predicted moves happen.** Area-days grouped by predicted change against the usual share:

| Predicted change | > +65% | +28..+65% | +10..+28% | −10..−22% | −22..−40% | < −40% |
|---|---|---|---|---|---|---|
| Predicted ratio | 3.00 | 1.42 | 1.16 | 0.86 | 0.72 | 0.53 |
| Observed ratio | 3.23 | 1.19 | 1.07 | 0.91 | 0.72 | 0.62 |

**Ratio error.** ½ Σ |forecast share − observed share| per day, on days with ≥ 10 calls. "Perfect model" is the
error a model that knew the true shares would still have, because a day's few calls land partly by chance:

| Level | Share model | Historical share | Perfect model |
|---|---|---|---|
| ZIP areas (177) | **0.819** | 0.824 | 0.817 |
| Boroughs (5) | **0.180** | 0.181 | 0.177 |

The model is within 0.002–0.003 of the best achievable.

### 3.5 Why the map does not change more from day to day

- **Within a summer, the scatter between areas is no larger than chance** (Pearson dispersion 0.8–0.93 around the
  season's shares; 1 = pure chance).
- **Recent calls don't predict tomorrow's location:** yesterday's or last week's local excess has correlation 0.01 with
  today's, and this morning's with this afternoon's 0.009.
- **Observed shares jump a lot between days** (0.835 at ZIP level), but they would jump just as much (0.831) if this
  model were exactly right: 20–60 calls spread over 177 areas land in different places by chance.
- **So** the model moves the map only where evidence supports it (weekends, beaches, events). Everywhere else it keeps
  the usual map.

### 3.6 Alternatives tested and not kept

| Alternative | Result |
|---|---|
| LightGBM share model with area identity and area traits | Reproduced the historical map (+0.002 nats per call) |
| Heat × vulnerability (AC, HVI, older adults alone, surface temperature), per-area heat slopes | Out-of-sample shares worse |
| Neighbourhood temperature (Daymet; NOAA HRRR analyses and forecasts) | Ratio error slightly better (0.8111 → 0.8097) but small; afternoon max helped, night min did not; history only from 2018 |
| Direct count model per UHF neighbourhood, and a 42-neighbourhood panel model (GLM + LightGBM, direct and reconciled) | Neighbourhood split no better; RMSE 0.485–0.490 vs 0.483; under-forecast the July 2026 record heatwave (164 vs 220) |

---

## 4. Runtime

Measured on a 10-core Apple Silicon laptop, CPU only.

| Step | Time |
|---|---|
| **City model:** load data and build features | 0.4 s |
| City model: fit on 2005–2026 (3,357 days) | 0.7 s |
| City model: full backtest (3 test summers, 5 comparison models, final model) | 4.8 s |
| **Share model:** load panel (594,189 area-days) | 0.3 s |
| Share model: one fit (7,788 cells) | 0.3 s |
| Share model: fit with ridge tuning (5 fits) | 1.7 s |
| Share model: `heattents area` (7-summer backtest + final model) | 14 s |
| **Daily forecast, one evening:** city model (30 weather draws) + shares + 5 demand scenarios for the optimiser (46,244 rows) | **0.15 s** |
| … of which the share model | 0.02 s |
| First forecast after start-up (loads models and data) | 0.9 s |

Retraining both models from scratch, backtests included, takes about **20 seconds**. The Gurobi deployment solve that
follows is separate (about 1 minute to a 1% optimality gap).

---

## 5. Running it

```bash
.venv/bin/heattents features   # area table, citywide series, area × day panel
.venv/bin/heattents city       # city model: 2024–26 backtest + final model → outputs/models/city/
.venv/bin/heattents area       # share model: 2018–26 backtest + final model → outputs/models/share/
.venv/bin/streamlit run app/streamlit_app.py
```

| Output | Content |
|---|---|
| `outputs/models/city/city_model.pkl` | Final city model |
| `outputs/models/city/metrics_*.csv`, `oos_predictions.csv` | City backtest |
| `outputs/models/share/share_model.pkl` | Final share model |
| `outputs/models/share/coefficients.csv` | Share multipliers (table 3.3) |
| `outputs/models/share/scores_summary.csv`, `move_calibration.csv`, `ratio_error_summary.csv` | Share backtest (section 3.4) |

---

## 6. Limitations

- EMS heat calls are a proxy for heat illness, and peak days are capped by EMS capacity (53% of calls were held on the 2026 peak).
- Past-day weather inputs use the full previous day, slightly past the 18:00 cutoff. The effect is small.
- The weather forecast is a single model (GFS). It ran hot on July 2024 peaks, so 2024 was over-forecast by 31%.
- Event effects rest on about 15 past occurrences each; West Indian Day (×15.9) is the least certain.
- Since 2024, beaches draw slightly fewer calls on weekends than the model expects (6.0% observed vs 7.1% forecast), and the usual hotspots slightly more.
- About 4% of calls have no usable location; they count in the city total only.
