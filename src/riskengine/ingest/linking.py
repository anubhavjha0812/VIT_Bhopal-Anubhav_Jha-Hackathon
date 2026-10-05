"""Link text to companies in our universe by cashtags ($AAPL) and company names."""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from riskengine.paths import REFERENCE

UNIVERSE_PATH = REFERENCE / "universe.csv"
CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")


@dataclass(frozen=True)
class Linker:
    cashtag_to_ticker: dict[str, str]
    alias_patterns: dict[str, re.Pattern]

    @classmethod
    def from_csv(cls, path: Path = UNIVERSE_PATH) -> "Linker":
        u = pd.read_csv(path)
        cashtags = {tag: row.ticker for row in u.itertuples() for tag in row.cashtags.split("|")}
        # Names are matched case-sensitively on word boundaries ("Apple", not "pineapple").
        aliases = {
            row.ticker: re.compile(
                r"\b(?:" + "|".join(map(re.escape, row.aliases.split("|"))) + r")\b"
            )
            for row in u.itertuples()
        }
        return cls(cashtags, aliases)

    @property
    def tickers(self) -> list[str]:
        return sorted(self.alias_patterns)

    def cashtags_in(self, text: str) -> list[str]:
        return [m.upper() for m in CASHTAG.findall(text)]

    def link(self, text: str, use_names: bool = True) -> tuple[str, ...]:
        """Tickers from our universe that the text mentions, sorted."""
        found = {
            self.cashtag_to_ticker[t] for t in self.cashtags_in(text) if t in self.cashtag_to_ticker
        }
        if use_names:
            found |= {tk for tk, pat in self.alias_patterns.items() if pat.search(text)}
        return tuple(sorted(found))
