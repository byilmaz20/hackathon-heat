"""Project paths. Modelling and decision assumptions will live here as frozen dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        p = self.root / "data" / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed(self) -> Path:
        p = self.root / "data" / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def out(self, step: str) -> Path:
        p = self.root / "outputs" / step
        p.mkdir(parents=True, exist_ok=True)
        return p


PATHS = Paths()
