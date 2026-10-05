"""Repo-relative paths, so code works from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
REFERENCE = DATA / "reference"
