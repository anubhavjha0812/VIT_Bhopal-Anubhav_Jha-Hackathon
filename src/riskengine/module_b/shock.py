"""Turn a gated credit alert into a scenario per obligor (docs/DECISIONS.md, E14).

1. Notches: parsed from the text ("to Baa2 from Baa1", "to 'A-' from 'A'"); none found = 1 notch;
   Default = D. The count is applied to our S&P rating at the book date.
2. The obligor moves to the new rating's spread. Every other obligor widens by that widening ×
   its daily return correlation with the obligor (2008-2019, negatives set to 0).
3. Market episode (real FRED and price moves between two dates): the Aaa and Baa spread change,
   the equal-weight 15-stock fall as z (in one-year standard deviations: 2008-2019 daily vol ×
   √252, since Vasicek's market factor is for the one-year PD) and the USD/INR change.
   `multiple` scales the episode: 1 = flat, impact/10 = scaled, k = reverse stress.
4. PD after = Vasicek stressed PD of the new rating at z, never below its normal one-year PD.
"""

import math
import re
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from riskengine.module_b.params import notch, pd_1y, rating_at, spread, stressed_pd
from riskengine.module_b.sources import BOOK_DATE, MARKET_PATH
from riskengine.paths import DATA

PRICES_PATH = DATA / "prices" / "prices_history.csv"
CALM = ("2008-01-01", BOOK_DATE)  # correlation and daily vol window
EPISODES = {
    "gfc2008": ("2008-08-15", "2008-12-15"),
    "euro2011": ("2011-07-01", "2011-10-04"),
    "covid2020": ("2020-02-14", "2020-03-23"),
}
DEFAULT_EPISODE = "covid2020"
TRADING_DAYS = 252

MOODYS = {
    "Aaa": "AAA", "Aa1": "AA+", "Aa2": "AA", "Aa3": "AA-", "A1": "A+", "A2": "A", "A3": "A-",
    "Baa1": "BBB+", "Baa2": "BBB", "Baa3": "BBB-", "Ba1": "BB+", "Ba2": "BB", "Ba3": "BB-",
    "B1": "B+", "B2": "B", "B3": "B-", "Caa1": "CCC+", "Caa2": "CCC", "Caa3": "CCC-", "Ca": "CC",
}  # fmt: skip
_TOKENS = sorted({*MOODYS, *(r for r in MOODYS.values())}, key=len, reverse=True)
_R = r"['‘’\"]?(" + "|".join(re.escape(t) for t in _TOKENS) + r")(?![\w+-])['‘’\"]?"
TO_FROM = re.compile(rf"\bto\s+{_R}\s+from\s+{_R}")
FROM_TO = re.compile(rf"\bfrom\s+{_R}\s+to\s+{_R}")
BARE_TO = re.compile(rf"{_R}\s+to\s+{_R}")


def _sp(token: str) -> str:
    return MOODYS.get(token, token)


def parse_notches(text: str, event: str) -> int:
    """How many notches the text says the rating fell (Default = all the way to D, else ≥ 1)."""
    if event == "Default":
        return 99
    for pattern, new_first in ((TO_FROM, True), (FROM_TO, False), (BARE_TO, False)):
        if m := pattern.search(text):
            new, old = (m.group(1), m.group(2)) if new_first else (m.group(2), m.group(1))
            cut = notch(_sp(new)) - notch(_sp(old))
            if cut > 0:
                return cut
    return 1


@dataclass(frozen=True)
class Market:
    episode: str
    aaa0: float  # spreads at the book date (% points)
    baa0: float
    d_aaa: float  # episode change (% points)
    d_baa: float
    z: float  # equal-weight 15-stock move, standard deviations, positive = fall
    fx0: float  # USD/INR at the book date
    fx_move: float  # episode USD/INR ratio - 1
    rate0: float  # 10-year Treasury at the book date (%); held fixed, only credit is shocked


@cache
def market_history() -> pd.DataFrame:
    return pd.read_csv(MARKET_PATH, parse_dates=["date"]).set_index("date")


@cache
def returns() -> pd.DataFrame:
    """Daily log returns of the 15 stocks (columns), 2008-2020."""
    p = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    wide = p[~p["ticker"].str.startswith("^")].pivot(index="date", columns="ticker", values="close")
    return np.log(wide).diff().iloc[1:]


@cache
def correlation() -> pd.DataFrame:
    """Return correlation over the calm window; negatives set to 0 (E14)."""
    return returns().loc[CALM[0] : CALM[1]].corr().clip(lower=0)


def _on_or_before(df: pd.DataFrame, day: str) -> pd.Series:
    return df.loc[:day].iloc[-1]


@cache
def market(episode: str = DEFAULT_EPISODE) -> Market:
    start, end = EPISODES[episode]
    m = market_history()
    book, a, b = _on_or_before(m, BOOK_DATE), _on_or_before(m, start), _on_or_before(m, end)
    ew = returns().mean(axis=1)  # equal weight over the names listed that day
    sigma_1y = ew.loc[CALM[0] : CALM[1]].std() * math.sqrt(TRADING_DAYS)
    z = -ew.loc[start:end].iloc[1:].sum() / sigma_1y
    return Market(episode, book["aaa"], book["baa"], b["aaa"] - a["aaa"], b["baa"] - a["baa"],
                  float(z), book["usdinr"], b["usdinr"] / a["usdinr"] - 1,
                  book["ust10y"])  # fmt: skip


def scenario(
    ratings: pd.Series,
    obligor: str | None = None,
    notches: int = 0,
    episode: str = DEFAULT_EPISODE,
    multiple: float = 1.0,
) -> tuple[pd.DataFrame, Market, float]:
    """Per ticker: rating, spread (% points) and PD before and after; plus the market and FX after.

    ratings: ticker -> S&P rating at the book date. obligor/notches: the alert (None = market only).
    """
    mk = market(episode)
    aaa1 = max(mk.aaa0 + multiple * mk.d_aaa, 0.01)
    baa1 = max(mk.baa0 + multiple * mk.d_baa, aaa1 + 0.01)
    z = multiple * mk.z
    out = pd.DataFrame({"rating_before": ratings})
    out["rating_after"] = out["rating_before"]
    if obligor is not None and notches > 0:
        out.loc[obligor, "rating_after"] = rating_at(notch(ratings[obligor]) + notches)
    out["spread_before"] = [spread(r, mk.aaa0, mk.baa0) for r in out["rating_before"]]
    market_after = pd.Series([spread(r, aaa1, baa1) for r in out["rating_before"]], out.index)
    out["spread_after"] = market_after
    if obligor is not None and notches > 0:
        own = spread(out.loc[obligor, "rating_after"], mk.aaa0, mk.baa0)
        widening = own - out.loc[obligor, "spread_before"]
        rho = correlation()[obligor].reindex(out.index).fillna(0)
        out["contagion"] = widening * rho
        out.loc[obligor, "contagion"] = widening
        out["spread_after"] = market_after + out["contagion"]
    else:
        out["contagion"] = 0.0
    out["pd_before"] = [pd_1y(r) for r in out["rating_before"]]
    out["pd_after"] = [max(pd_1y(r), stressed_pd(pd_1y(r), z)) for r in out["rating_after"]]
    fx_after = mk.fx0 * (1 + multiple * mk.fx_move)
    return out, mk, fx_after
