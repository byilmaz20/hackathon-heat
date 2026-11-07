"""Label data: NYC EMS Incident Dispatch rows whose final call type is HEAT (heat-related illness).

Source: NYC Open Data, "EMS Incident Dispatch Data" (Socrata id 76xm-jjuj), 2005 onwards.
The location fields are those of the *incident* (where the person was), not of residence - exactly what a
cooling-tent deployment needs. Finest spatial unit available: ZIP code.
"""
from __future__ import annotations

import pandas as pd

from ..config import PATHS, Paths
from .http import socrata

DATASET = "76xm-jjuj"
FIELDS = ["incident_id", "incident_datetime", "initial_call_type", "final_call_type", "initial_severity_level_code",
          "final_severity_level_code", "incident_disposition_code", "held_indicator", "special_event_indicator",
          "borough", "zipcode", "communitydistrict", "incident_dispatch_area",
          "dispatch_response_seconds_qy", "incident_response_seconds_qy"]


def fetch(paths: Paths = PATHS, refresh: bool = False) -> pd.DataFrame:
    out = paths.raw / "nyc_ems_heat_incidents.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, parse_dates=["incident_datetime"], dtype={"zipcode": "string"})
    rows = socrata(DATASET, {"$select": ",".join(FIELDS), "$where": "final_call_type='HEAT'", "$order": "incident_id"})
    df = pd.DataFrame(rows).reindex(columns=FIELDS)
    df["incident_datetime"] = pd.to_datetime(df.incident_datetime)
    df = df.sort_values("incident_datetime").reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"[ingest:ems] {len(df):,} HEAT incidents {df.incident_datetime.min():%Y-%m-%d} .. "
          f"{df.incident_datetime.max():%Y-%m-%d} -> {out}")
    return df
