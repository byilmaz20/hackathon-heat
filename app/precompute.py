"""Precompute the data behind the slide tabs (run once; the app reads the CSVs).

    .venv/bin/python app/precompute.py

outputs/app/oos_2026_neighbourhoods.csv  every 2026 day forecast as issued the evening before, with city and share
                                         models trained on 2005-2025 only (out of sample): date, uhf, expected, observed
                                         (people)
outputs/app/input_correlations.csv       correlation of the main inputs with daily citywide calls, May-Sep 2005-2026
outputs/models/deploy_compare/coverage/  plan comparison under the active rules (heattents.deploy.compare.run_coverage)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heattents.config import PATHS  # noqa: E402

INPUTS = {"atmax": "Heat index (max)", "ehf": "Heat lasting several days", "tmin": "Night-time temperature",
          "lag_count": "Yesterday's calls", "streak": "Hot days in a row", "rh": "Humidity", "weekend": "Weekend"}


def oos_2026():
    from heattents.deploy.compare import seasonal_service
    svc = seasonal_service(2026)
    days = svc.ctx.city[svc.ctx.city.year == 2026].date
    rows = []
    for d in days:
        nb = svc.forecast_date(d).neighbourhoods
        rows.append(nb.assign(date=pd.Timestamp(d).date())[["date", "uhf", "name", "expected", "observed"]])
    out = PATHS.out("app")
    pd.concat(rows, ignore_index=True).to_csv(out / "oos_2026_neighbourhoods.csv", index=False)
    print(f"[precompute] 2026 out-of-sample neighbourhood forecasts: {len(days)} days")


def correlations():
    from heattents.models.city import observed_features
    from heattents.service import ForecastService
    ctx = ForecastService().ctx
    F = observed_features(ctx, ctx.city)
    y = ctx.city.city_y.to_numpy()
    C = pd.DataFrame([dict(input=k, label=v, correlation=float(pd.Series(F[k].to_numpy(float)).corr(pd.Series(y))))
                      for k, v in INPUTS.items()])
    C.sort_values("correlation", ascending=False).to_csv(PATHS.out("app") / "input_correlations.csv", index=False)
    print(C.round(2).to_string())


if __name__ == "__main__":
    correlations()
    oos_2026()
    from heattents.deploy.compare import run_coverage
    run_coverage()
