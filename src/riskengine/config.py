"""Typed run config: one switch per part, baseline or upgrade."""

import tomllib
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Literal, get_args, get_type_hints

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass(frozen=True)
class Config:
    sentiment: Literal["finbert", "own", "entity"] = "finbert"
    events: Literal["keywords", "model", "model_rules", "student_rules"] = "keywords"
    impact: Literal["formula", "learned"] = "formula"
    rebalancer: Literal["equal", "sentiment"] = "sentiment"
    shock: Literal["flat", "scaled"] = "flat"
    rates: Literal["fixed", "moved"] = "fixed"  # E22: Treasury held fixed, or moved by the episode
    extended_linker: bool = False  # E38: live/API only; batch retains the original universe
    live_decisions: bool = False  # E41: opt-in dry-run previews and a separate live ledger
    impact_trigger: int = 7
    window_start: date = date(2020, 4, 9)
    window_end: date = date(2020, 6, 10)

    def __post_init__(self) -> None:
        if type(self.extended_linker) is not bool:
            raise ValueError("extended_linker must be a boolean")
        if type(self.live_decisions) is not bool:
            raise ValueError("live_decisions must be a boolean")
        hints = get_type_hints(type(self))
        for f in fields(self):
            allowed = get_args(hints[f.name])
            value = getattr(self, f.name)
            if allowed and value not in allowed:
                raise ValueError(f"{f.name}={value!r}, expected one of {allowed}")
        if not 1 <= self.impact_trigger <= 10:
            raise ValueError(f"impact_trigger={self.impact_trigger}, expected 1-10")
        if not self.window_start < self.window_end:
            raise ValueError(f"window_start {self.window_start} must be before {self.window_end}")


def load_config(path: Path = DEFAULT_PATH) -> Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)
