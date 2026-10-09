"""Link text to companies in our universe by cashtags ($AAPL) and company names."""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from riskengine.paths import REFERENCE

UNIVERSE_PATH = REFERENCE / "universe.csv"
CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")

# Universe companies that also publish stock ratings (DECISIONS E12).
BROKERS = ("JPM", "BAC")
RATING_TEXT = re.compile(
    r"downgrad|upgrad|rating|price target|\bPT\b|overweight|underweight|outperform|"
    r"underperform|neutral|equal.weight|(?:market|sector) perform|\bto (?:buy|sell|hold)\b|"
    r"coverage",
    re.IGNORECASE,
)
RATING_ACTION = (
    r"(?i:downgrade[sd]?|upgrade[sd]?|initiate[sd]?|maintain(?:s|ed)?|reiterate[sd]?|"
    r"raise[sd]?|lower(?:s|ed)?|cuts?|boost(?:s|ed)?|increase[sd]?|trim(?:s|med)?|"
    r"resume[sd]?|reinstate[sd]?|assume[sd]?|start(?:s|ed)?)"
)


def _desk_pattern(names: str) -> re.Pattern:
    """The bank's research desk giving a view: 'Bank of America analysts see oil at $40'."""
    return re.compile(
        rf"\b(?:{names})(?:\s+Chase\s*&\s*Co\.?|\s+Securities|\s+Global Research)?"
        rf"\s+(?i:analysts?|strategists?|economists?)\b"
    )


def _analyst_pattern(names: str) -> re.Pattern:
    """The bank as the one giving the rating: at the start, or after by/at."""
    return re.compile(
        rf"^(?:UPDATE:\s*|Hearing\s+)?(?:{names})"
        rf"(?:\s+Chase\s*&\s*Co\.?|\s+Securities|\s+Global Research)?\s+{RATING_ACTION}\b"
        rf"|\b(?:by|at)\s+(?:{names})\b"
    )


@dataclass(frozen=True)
class Linker:
    cashtag_to_ticker: dict[str, str]
    alias_patterns: dict[str, re.Pattern]
    analyst_patterns: dict[str, re.Pattern]
    desk_patterns: dict[str, re.Pattern]

    @classmethod
    def from_csv(cls, path: Path = UNIVERSE_PATH) -> "Linker":
        u = pd.read_csv(path)
        cashtags = {tag: row.ticker for row in u.itertuples() for tag in row.cashtags.split("|")}
        names = {
            row.ticker: "|".join(map(re.escape, row.aliases.split("|"))) for row in u.itertuples()
        }
        # Names are matched case-sensitively on word boundaries ("Apple", not "pineapple").
        aliases = {tk: re.compile(rf"\b(?:{n})\b") for tk, n in names.items()}
        analysts = {tk: _analyst_pattern(names[tk]) for tk in BROKERS}
        desks = {tk: _desk_pattern(names[tk]) for tk in BROKERS}
        return cls(cashtags, aliases, analysts, desks)

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

    def analyst_only(self, text: str) -> set[str]:
        """Banks named only as the analyst of a stock rating, not as what the text is about.

        "iRobot $IRBT Downgraded by JPMorgan Chase & Co. to Underweight" -> {"JPM"}. A bank
        whose own cashtag is in the text is the subject and is kept. "<bank> analysts /
        strategists / economists" counts even without rating words (E13).
        """
        found = {tk for tk, pat in self.desk_patterns.items() if pat.search(text)}
        if RATING_TEXT.search(text):
            found |= {tk for tk, pat in self.analyst_patterns.items() if pat.search(text)}
        own = {self.cashtag_to_ticker.get(t) for t in self.cashtags_in(text)}
        return found - own
