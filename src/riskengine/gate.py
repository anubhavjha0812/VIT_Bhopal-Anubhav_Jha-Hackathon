"""Credit Chhanni: which (text, company) rows may trigger Module B (docs/DECISIONS.md, E13).

A row fires only if all hold:
1. the event is a credit class (Default, Credit downgrade, Covenant breach)
2. the company is the subject, not a bank that only gives a stock rating (E12)
3. it is the obligor: named in the text and, if the text has cashtags, the first is its own
   ("$DO $XOM $CVX: Diamond Offshore files for bankruptcy" is not about Exxon)
4. impact is above the trigger (7 in the case study)
5. a rating agency is named, or a different text on the same company and credit class arrived
   in the previous 24 hours (one tweet alone is a rumour)

Rows get `fired` and `reason` (the first check that failed, or how it passed).
"""

import re

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.nlp.events import CREDIT_CLASSES, names_agency

CONFIRM_WINDOW = pd.Timedelta(hours=24)
_URL = re.compile(r"https?://\S+")


def is_obligor(linker: Linker, text: str, ticker: str) -> bool:
    if ticker not in linker.link(text):
        return False
    tags = [linker.cashtag_to_ticker.get(t) for t in linker.cashtags_in(text)]
    return not tags or tags[0] == ticker


def _same_text(text: str) -> str:
    return " ".join(_URL.sub("", text).lower().split())


def gate(signals: pd.DataFrame, trigger: int, linker: Linker | None = None) -> pd.DataFrame:
    """`signals` plus `fired` and `reason` per row."""
    linker = linker or Linker.from_csv()
    s = signals.copy()
    if "role" not in s:
        s["role"] = "subject"
    reason = pd.Series("", index=s.index, dtype=object)
    credit = s["event"].isin(CREDIT_CLASSES)
    reason[~credit] = "not credit"
    analyst = credit & (s["role"] != "subject")
    reason[analyst] = "analyst bank"
    rest = credit & ~analyst
    obligor = pd.Series(
        [
            bool(c) and is_obligor(linker, t, tk)
            for c, t, tk in zip(rest, s["text"], s["ticker"], strict=True)
        ],
        index=s.index,
    )
    reason[rest & ~obligor] = "not the obligor"
    candidates = rest & obligor  # credit news about this company, any impact
    reason[candidates & (s["impact"] <= trigger)] = f"impact ≤ {trigger}"

    pool = s[candidates]
    for i in s.index[candidates & (s["impact"] > trigger)]:
        row = s.loc[i]
        if names_agency(row["text"]):
            reason[i] = "fired: rating agency"
            continue
        earlier = pool[
            (pool["ticker"] == row["ticker"])
            & (pool["event"] == row["event"])
            & (pool["published_at"] < row["published_at"])
            & (pool["published_at"] >= row["published_at"] - CONFIRM_WINDOW)
        ]
        own = _same_text(row["text"])
        if any(_same_text(t) != own for t in earlier["text"]):
            reason[i] = "fired: 2 sources in 24 h"
        else:
            reason[i] = "1 source, no agency"
    s["reason"] = reason
    s["fired"] = reason.str.startswith("fired")
    return s
