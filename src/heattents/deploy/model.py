"""Inputs and outputs of the daily deployment optimiser (the contract used by the app and the evaluation code)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Shift:
    """A working shift covering the hours [start_hour, end_hour) of the day (0..24)."""
    name: str
    start_hour: int
    end_hour: int

    def __post_init__(self):
        if not 0 <= self.start_hour < self.end_hour <= 24:
            raise ValueError(f"shift {self.name}: need 0 <= start < end <= 24")

    @property
    def hours(self) -> range:
        return range(self.start_hour, self.end_hour)

    def overlaps(self, other: "Shift") -> bool:
        return self.start_hour < other.end_hour and other.start_hour < self.end_hour


@dataclass(frozen=True)
class Doctor:
    doctor_id: str
    available_shifts: list[str]
    max_shifts: int = 1


@dataclass(frozen=True)
class Site:
    site_id: str
    name: str
    lat: float
    lon: float
    max_doctors: int = 4
    kind: str = "park"


@dataclass
class DeploymentRequest:
    """One day's deployment question.

    demand: one row per (area, hour, scenario) with columns [area_id, lat, lon, hour, scenario, expected_cases].
            Scenarios are equally weighted samples from the predictive distribution; missing rows mean 0.
    hour_blocks: optional list of (start, end) hour blocks; demand and tent capacity are aggregated within each
            block to shrink the model (exact when no block is split by a shift boundary).
    existing_capacity: reserved for existing cooling-site capacity (ignored in v1).
    """
    n_tents: int
    sites: list[Site]
    doctors: list[Doctor]
    shifts: list[Shift]
    demand: pd.DataFrame
    walk_radius_km: float = 1.0
    distance_decay_km: float | None = 0.6
    patients_per_doctor_hour: float = 4.0
    min_doctors_per_open_tent: int = 1
    existing_capacity: pd.DataFrame | None = None
    hour_blocks: list[tuple[int, int]] | None = None
    time_limit_s: float = 60
    mip_gap: float = 0.005


@dataclass
class DeploymentPlan:
    open_sites: pd.DataFrame  # site_id, name, lat, lon, kind, doctors_<shift>..., expected_served
    assignments: pd.DataFrame  # doctor_id, site_id, shift
    served: pd.DataFrame  # area_id, hour, hour_end, expected_served, expected_demand
    kpis: dict
    status: str
    mip_gap: float
    runtime_s: float
    stats: dict = field(default_factory=dict)
