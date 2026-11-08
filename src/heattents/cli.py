"""Command line:  heattents <step>   (or python -m heattents <step>)

Steps, in pipeline order (``all`` runs ingest -> features -> city -> area):
    ingest     download every raw input into data/raw (cached; Open-Meteo has a daily quota, re-run to resume)
    features   static area table, citywide daily series, area x day panel (data/processed)
    eda        exploratory analysis of the label (outputs/eda)
    city       citywide model: 2024-26 season-by-season test, final model, heat curve and effects (outputs/models/city)
    area       beach & event share model: rolling-origin evaluation + final model (outputs/models/share)
    export     static JSON/GeoJSON result bundle for a front-end: forecasts + precomputed Gurobi plans (outputs/ui)
    app        print how to start the Streamlit decision-support app
"""
from __future__ import annotations

import argparse
import time

from .config import PATHS

STEPS = ["ingest", "features", "eda", "city", "area", "export", "app"]


def ingest(refresh: bool = False):
    from .ingest import activity, context, demographics, ems, geography, weather
    ems.fetch(PATHS, refresh)
    geography.load_modzcta(PATHS, refresh)
    for kind in ("NTA2010", "UHF42", "CD"):
        geography.load_polygons(kind, PATHS, refresh)
    context.indicators(PATHS, refresh)
    context.syndromic(PATHS, refresh)
    context.hvi(PATHS, refresh)
    context.cooling_sites(PATHS, refresh)
    activity.jobs(PATHS, refresh)
    activity.subway(PATHS, refresh)
    demographics.park_share(PATHS, refresh)
    demographics.acs(PATHS, refresh)  # optional, needs CENSUS_API_KEY
    weather.fetch_all(PATHS, refresh)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="heattents", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--refresh", action="store_true", help="re-download raw inputs")
    args = ap.parse_args(argv)
    steps = ["ingest", "features", "city", "area"] if args.step == "all" else [args.step]
    for step in steps:
        t0 = time.time()
        print(f"=== {step}")
        if step == "ingest":
            ingest(args.refresh)
        elif step == "features":
            from .features import panel, static
            static.build(PATHS)
            panel.build(PATHS)
        elif step == "eda":
            from . import eda
            eda.main() if hasattr(eda, "main") else eda.run()
        elif step == "city":
            from .models import city, explain
            city.evaluate(PATHS)
            explain.run(PATHS)
        elif step == "area":
            from .models import context_share
            context_share.run(PATHS)
        elif step == "export":
            from . import export
            export.run(PATHS)
        elif step == "app":
            print("Start the decision-support app with:\n    .venv/bin/streamlit run app/streamlit_app.py")
        print(f"=== {step} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
