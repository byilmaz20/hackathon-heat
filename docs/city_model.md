# City model: forecasting tomorrow's heat EMS calls in New York City

**Question.** At 18:00 on day *d−1*, how many heat-related EMS calls will New York City have on day *d*?

**Answer the model gives.** A full probability distribution for the citywide count, not a single number. It reports the
expected count, quantiles (5%, 25%, 50%, 75%, 95%) and the probability of ≥ 50 and ≥ 100 calls.

**Code:** [src/heattents/models/city.py](../src/heattents/models/city.py).
**Run:** `python -m heattents.models.city` (evaluation + final model, ~5 s).

---

## 1. Data

| Data | Source | What is used | Period used | Role |
|---|---|---|---|---|
| **EMS heat calls (target)** | NYC Open Data, *EMS Incident Dispatch Data*, dataset `76xm-jjuj` ([link](https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj)) | Every incident with `final_call_type = 'HEAT'`, counted per day. Held, cancelled and non-transported calls are included. 17,688 incidents in total; 17,278 in May–Sep. | May–Sep, 2005–2026 (to 21 Sep 2026) | Target, plus "calls yesterday before 18:00" as an input |
| **Observed weather, NYC** | Open-Meteo Historical Weather API ([docs](https://open-meteo.com/en/docs/historical-weather-api)), ERA5 / ERA5-Land reanalysis, Central Park (40.781, −73.967) | Daily max/min/mean temperature, max/mean apparent temperature (heat index), mean relative humidity | 2005–2026, full year (for 30-day look-backs) | Training inputs; days *d−1 … d−3* at forecast time |
| **Observed weather, upstream cities** | Same API: Philadelphia, Baltimore, Washington DC | Daily max apparent temperature | 2005–2026 | Lead-lag input ("is hotter air coming from the south-west?"). Newark and Trenton are downloaded but not used. |
| **Weather forecasts** | Open-Meteo Previous Runs API ([docs](https://open-meteo.com/en/docs/previous-runs-api)), "best match" model (GFS for NYC), Central Park | The forecast for day *d* issued the day before (`*_previous_day1`): hourly temperature, apparent temperature and humidity, aggregated to daily max/min/mean | 19 Jan 2024 – 30 Sep 2026 (humidity and apparent temperature are archived only from 19 Jan 2024) | Day-*d* weather at forecast time; real forecast errors for the uncertainty layer |
| **Heat-syndrome ED visits** | NYC Health, Environment & Health Data Portal ([GitHub: nychealth/EHDP-data](https://github.com/nychealth/EHDP-data), `key-topics/heat-syndrome`) | Citywide daily heat-related emergency department visits (100% of NYC EDs) | May–Sep 2017–2026 | **Validation only**, as an independent check; not a model input |
| **Calendar** | Rule-based, in code | Weekend; federal holidays (Memorial Day, Juneteenth from 2021, 4 July observed, Labor Day); 3–5 July; NYC school summer break (27 June to the Thursday after Labor Day); COVID years 2020–21 | — | Inputs |

**Why the test starts in 2024.** Before 19 January 2024 the forecast archive has only one model's (GFS) air
temperature: no humidity, so no heat index. Testing 2022–23 would mean testing an incomplete archive, not the model.

---

## 2. Pipeline

```
ingest (data/raw)                       features (data/processed)                 model (outputs/models/city)
─────────────────                       ─────────────────────────                 ───────────────────────────
ems.fetch ──────────────► daily call counts, calls before 18:00 ─┐
weather.upstream_daily ─► heat features (features/weather.py) ───┼─► city_daily.parquet ─► city.evaluate()
weather.forecasts ──────────────────────────────────────────────┘    (one row per day)      ├─ season-by-season test 2024-26
context.syndromic (validation only)                                                         ├─ metrics, out-of-sample predictions
                                                                                            └─ final model on all data → city_model.pkl
```

| Step | Command | Output |
|---|---|---|
| 1. Download inputs | `heattents ingest` | `data/raw/*` |
| 2. Build daily series | `heattents features` | `data/processed/city_daily.parquet` (3,357 days) |
| 3. Evaluate + fit final model | `python -m heattents.models.city` | `metrics_by_season.csv`, `metrics_summary.csv`, `oos_predictions.csv`, `city_model.pkl`, `glm_coefficients.csv` |
| 4. Readable views | `python -m heattents.models.explain` | `heat_curve.png/.csv`, `effects.csv` |
| 5. Error diagnostics | `python -m heattents.models.diagnostics 2024` | `diagnostics/cumulative_error_2024.png/.csv` |

---

## 3. Inputs the model sees (all known at 18:00 on *d−1*)

Temperatures in °C inside the model (°F in brackets). "Apparent temperature" = heat index.

| Input | Definition | Day *d* comes from |
|---|---|---|
| `atmax` | Max apparent temperature on *d* | Weather forecast |
| `atmax_l1`, `atmax_l2`, `atmax_l3` | Max apparent temperature on *d−1*, *d−2*, *d−3* | Observed |
| `tmax`, `tmin`, `tmean`, `rh` | Max / min / mean air temperature, mean relative humidity on *d* | Weather forecast |
| `ehf` | Excess Heat Factor: `max(0, EHI_sig) · max(1, EHI_accl)`. `EHI_sig` = 3-day mean temperature (*d−2..d*) minus the 95th percentile of summer daily means over 2005–2019 (27.2 °C). `EHI_accl` = 3-day mean minus the mean of the preceding 30 days. | Forecast for *d*, observed for earlier days |
| `streak` | Consecutive days with apparent max ≥ 32 °C (≈ 90 °F), capped at 7 | Forecast for *d*, observed for earlier days |
| `season_hot_days` | log(1 + number of such days earlier this year): adaptation over the summer | Observed |
| `tropical_night` | Min temperature on *d* ≥ 24 °C (≈ 75 °F) | Weather forecast |
| `up_sw_gradient` | Yesterday's max apparent temperature, mean of Philadelphia / Baltimore / DC, minus NYC's | Observed |
| `lag_count` | log(1 + heat calls on *d−1* made **before 18:00**). On average 81% of a day's calls arrive before 18:00. | Observed |
| Calendar | `weekend`, `holiday`, `july4_window`, `school_break`, `covid` | Known |
| Trend | `year`, `day_of_year` | Known |

---

## 4. Models

Two models see exactly the same inputs. The forecast is their **average** (the ensemble). This choice was fixed
before looking at the test results.

### 4.1 Negative binomial regression with a distributed-lag heat term (GLM)

$$
y_d \sim \text{NegBin}(\mu_d,\ \alpha),\qquad \text{Var}(y_d) = \mu_d + \alpha\,\mu_d^2
$$

$$
\log \mu_d = \beta_0
+ \underbrace{\sum_{i,j} \theta_{ij}\, \sum_{l=0}^{3} B_i(\text{atmax}_{d-l})\, C_j(l)}_{\text{heat} \times \text{lag surface (DLNM)}}
+ \sum_k \gamma_k\, z_{k,d}
+ s_{\text{year}}(d) + s_{\text{season}}(d)
$$

| Component | Specification | Coefficients |
|---|---|---|
| Intercept | — | 1 |
| Heat × lag cross-basis (DLNM) | $B$ = natural cubic spline in apparent max, knots 26 / 31 / 35 °C, boundary 10 / 44 °C (linear beyond); $C$ = quadratic polynomial over lags 0–3 | 4 × 3 = 12 |
| Linear terms $z_k$ (scaling in brackets) | EHF (/10); streak (/7); season hot days (/3); tropical night; max(0, upstream gradient) (/5 °C); humid heat = (RH − 70)/10 · 1[atmax ≥ 30 °C]; weekend; holiday; 4 July window; school break; COVID; lag count (/3) | 12 |
| $s_{\text{year}}$ | Natural spline, knots 2010 / 2015 / 2020, boundary 2005 / 2026 | 4 |
| $s_{\text{season}}$ | Natural spline in day of year, knots 170 / 200 / 230, boundary 121 / 273 | 4 |
| **Total** | | **33** |

**Fitting.** Penalised iteratively reweighted least squares (IRLS), ridge penalty 0.3 on every coefficient except the
intercept. The dispersion α is chosen by profile likelihood over the grid
{0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5}; the final model has **α = 0.2**. Code:
[glm.py](../src/heattents/models/glm.py).

### 4.2 Gradient-boosted trees (LightGBM)

| Setting | Value |
|---|---|
| Objective | Poisson |
| Inputs | 23: `atmax`, `atmax_l1..l3`, `tmax`, `tmin`, `tmean`, `rh`, `ehf`, `ehi_sig`, `ehi_accl`, `streak`, `season_hot_days`, `tropical_night`, `up_sw_gradient`, `lag_count`, `weekend`, `holiday`, `july4_window`, `school_break`, `covid`, `year`, `day_of_year` |
| Monotone constraints | Increasing in `atmax`, `atmax_l1`, `tmax`, `ehf`, `streak` |
| Trees / learning rate | 500 / 0.03 |
| Leaves / min. data per leaf | 15 / 20 |
| Row / feature subsampling | 0.8 / 0.8 |
| L2 regularisation | 1.0 |
| Seed | 7 |

### 4.3 Ensemble

$\hat\mu_d = \tfrac12(\mu^{\text{GLM}}_d + \mu^{\text{GBM}}_d)$. The GLM extrapolates sensibly to record heat; the trees
fit typical hot days better (on ordinary days the two are tied). On test days the ensemble beats both components, on
every kind of day (Section 6; day-type breakdown in [forecast_model.md § 2.3](forecast_model.md#23-why-both-a-glm-and-gradient-boosted-trees)).

---

## 5. Uncertainty

The forecast distribution combines two sources of uncertainty.

1. **Weather uncertainty (Monte Carlo, 30 draws).** Day *d*'s weather is redrawn 30 times as
   *forecast + error*. The errors are real past forecast errors, chosen as **analogs**:
   - only forecasts already verified at 18:00 on *d−1* (target day ≤ *d−2*);
   - only May–September;
   - only days whose forecast heat index was within ±3 °C of today's.

   If fewer than 20 such errors exist (the first 21 days of May 2024), a Gaussian N(0, 2 °C) shift is used instead.
   Conditioning on the forecast level matters: very hot forecasts tend to verify cooler, and with a convex heat
   response, unconditional errors inflated the 2024–26 total by 21% (+4% with analogs).
2. **Count noise.** Negative binomial with α = 0.2 around each draw's expected count.

$$
P(y_d = k) = \frac{1}{30}\sum_{m=1}^{30} \text{NegBin}\big(k;\ \hat\mu_d(x_d^{(m)}),\ \alpha\big)
$$

---

## 6. Training, testing and metrics

**Protocol (rolling origin by season).** For each test season *s* ∈ {2024, 2025, 2026}, the model is trained on all
summers 2005 … *s*−1, then every day of *s* is forecast from the 18:00 information above.

| Test season | Training days | Test days | Actual calls |
|---|---|---|---|
| 2024 | 2,907 | 153 | 904 |
| 2025 | 3,060 | 153 | 1,194 |
| 2026 | 3,213 | 144 | 1,443 |

Training uses **observed** weather for day *d* ("perfect prog", the standard for impact models; Wilks 2011). Only
test days use the forecast.

### 6.1 Headline results (450 test days, out of sample)

| Metric | Value |
|---|---|
| Daily absolute error | **median 1 call, mean 3.3 calls** (RMSE 7.6) |
| Days within ±5 calls | 84% |
| Correlation, forecast vs actual | 0.93 (rank 0.86) |
| Correlation with independent ED heat visits | 0.89 |
| 90% prediction interval contains the actual count | 96% of days |
| Total over 2024–26 | 3,677 forecast vs 3,541 actual (**+4%**) |
| Days with ≥ 50 calls (8 days) | **6 caught** by an alert P(≥50) ≥ 50%, 6 false alarms; total on those days forecast at 83% of actual |
| 10 busiest actual days among the 10 highest forecasts | 7 of 10 |

### 6.2 Comparison with simple alternatives (lower is better)

**CRPS** (continuous ranked probability score; Gneiting & Raftery 2007) scores the whole forecast distribution in
units of calls. **twCRPS ≥ 50** is the same score weighted towards forecasts of 50 or more calls (Gneiting & Ranjan
2011). It measures, over all test days, how well the probability of a high-call day is forecast.

| Model | CRPS | twCRPS ≥ 50 | Mean abs. error | Log score ↑ | 90% coverage |
|---|---|---|---|---|---|
| **Ensemble (used)** | **2.32** | **0.59** | **3.32** | **−2.05** | 0.96 |
| LightGBM alone | 2.44 | 0.75 | 3.56 | −2.05 | 0.97 |
| GLM alone | 2.57 | 0.72 | 3.96 | −2.10 | 0.95 |
| Temperature-only Poisson GLM | 2.97 | 0.80 | 4.37 | −2.27 | 0.90 |
| Persistence (yesterday's calls) | 4.02 | 1.04 | 5.68 | −2.69 | 0.86 |
| Climatology (same time of year) | 5.48 | 1.59 | 6.75 | −2.70 | 0.94 |

The ensemble's CRPS is 22% lower than the temperature-only model and 58% lower than climatology.

### 6.3 By season

| Season | Actual | Forecast total | CRPS | Mean abs. error | 90% coverage | Corr. with ED |
|---|---|---|---|---|---|---|
| 2024 | 904 | 1,180 (+31%) | 2.27 | 3.86 | 0.97 | 0.86 |
| 2025 | 1,194 | 1,181 (−1%) | 2.21 | 3.19 | 0.95 | 0.86 |
| 2026 | 1,443 | 1,316 (−9%) | 2.48 | 2.92 | 0.97 | 0.96 |

**Why 2024 is too high** ([diagnostics](../outputs/models/city/diagnostics/cumulative_error_2024.png)). With observed
weather the model forecasts 895 for 2024 (actual 904), so its heat response was right. About +200 comes from the
evening-before weather forecasts over-shooting peak July days by 4–8 °F; about +85 from the Monte Carlo spreading that
already-too-hot forecast. In 2024 busy days are mostly over-forecast; the exception is the start of the June heatwave
(20–21 June: 21 and 46 forecast vs 37 and 56 actual).

### 6.4 Readable effects

[heat_curve.png](../outputs/models/city/heat_curve.png) shows expected calls against tomorrow's heat index: ~3 at
85 °F, ~8 at 90 °F, ~20 at 95 °F, ~37 at 100 °F, ~60 at 105 °F, ~97 at 110 °F (July weekday, yesterday's calls held at
20). [effects.csv](../outputs/models/city/effects.csv) lists the change in expected calls on a typical hot weekday when
one input changes:

| Change | Effect |
|---|---|
| Heat index +5 °F | +82% |
| Twice as many calls yesterday before 18:00 | +30% |
| Federal holiday | +21% |
| South-west 9 °F hotter than NYC yesterday | +12% |
| Weekend | +10% |
| COVID years | −32% |

These are the model's associations with other inputs held fixed, not causal effects.

---

## 7. Limitations

- **The target is recorded EMS heat calls**, not all heat illness. People who are not seen by EMS are not counted.
- **The weather forecast is a single model (GFS via Open-Meteo "best match").** It ran hot on peak days in July 2024.
  ECMWF or a multi-model mean may be better; this is untested.
- **Few extreme days.** There are only 8 test days with ≥ 50 calls, so peak-day statistics are uncertain.
- **Citywide only.** Where in the city calls happen is a separate model (the beach & event share model, [forecast_model.md](forecast_model.md)).
- **Season only.** May–September; 97.7% of heat calls fall in these months.

---

## 8. References

- Cameron, A. C. & Trivedi, P. K. (2013). *Regression Analysis of Count Data*, 2nd ed. Cambridge University Press. (Negative binomial NB2.)
- Czado, C., Gneiting, T. & Held, L. (2009). Predictive model assessment for count data. *Biometrics* 65(4), 1254–1261. (Randomised PIT for counts.)
- Delle Monache, L., Eckel, F. A., Rife, D. L., Nagarajan, B. & Searight, K. (2013). Probabilistic weather prediction with an analog ensemble. *Monthly Weather Review* 141(10), 3498–3516. (Analog forecast errors.)
- Gasparrini, A., Armstrong, B. & Kenward, M. G. (2010). Distributed lag non-linear models. *Statistics in Medicine* 29(21), 2224–2234. (DLNM cross-basis.)
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102(477), 359–378. (CRPS, log score.)
- Gneiting, T. & Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *Journal of Business & Economic Statistics* 29(3), 411–422. (Threshold-weighted CRPS.)
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed. Springer. (Natural cubic splines.)
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society* 146(730), 1999–2049.
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems* 30.
- Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data* 13(9), 4349–4383.
- Nairn, J. R. & Fawcett, R. J. B. (2015). The Excess Heat Factor: a metric for heatwave intensity and its use in classifying heatwave severity. *International Journal of Environmental Research and Public Health* 12(1), 227–253.
- Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences*, 3rd ed. Academic Press. (Perfect-prog forecasting.)
- Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo. doi:10.5281/zenodo.7970649.
