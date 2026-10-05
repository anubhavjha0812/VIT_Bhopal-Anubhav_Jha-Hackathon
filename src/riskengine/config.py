"""Typed run config: one switch per part, baseline or upgrade."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, get_args, get_type_hints

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass(frozen=True)
class Config:
    sentiment: Literal["finbert", "own"] = "finbert"
    events: Literal["keywords", "model"] = "keywords"
    impact: Literal["formula", "learned"] = "formula"
    rebalancer: Literal["equal", "sentiment"] = "equal"
    shock: Literal["flat", "scaled"] = "flat"
    impact_trigger: int = 7

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for f in fields(self):
            allowed = get_args(hints[f.name])
            value = getattr(self, f.name)
            if allowed and value not in allowed:
                raise ValueError(f"{f.name}={value!r}, expected one of {allowed}")
        if not 1 <= self.impact_trigger <= 10:
            raise ValueError(f"impact_trigger={self.impact_trigger}, expected 1-10")


def load_config(path: Path = DEFAULT_PATH) -> Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)
