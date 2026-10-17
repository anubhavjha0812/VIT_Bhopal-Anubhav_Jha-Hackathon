"""Market gate: which Geopolitical / Macroeconomic items may trigger a market-wide stress test.

The credit gate (gate.py) needs an obligor. A war or a rate shock has none, so this gate works on
documents, not (document, company) rows. It follows the case study's own example, "Geopolitical
with an Impact Score > 7". An item fires only if all hold:
1. the event is Geopolitical or Macroeconomic
2. sentiment is negative (good news must not trigger stress)
3. market impact is above the trigger: MARKET_SEVERITY x (0.6 + 0.4 |sentiment|), +1 for news
   (credit and other classes keep their own severity in impact.py; this does not change them)
4. a different Geo/Macro text of the same class arrived in the previous 24 hours (one item alone
   is a rumour; a tweet alone can never pass 3)

Rows get `fired` and `reason`, one row per document.
"""

import pandas as pd

from riskengine.gate import CONFIRM_WINDOW, _same_text
from riskengine.nlp.events import _COMPILED
from riskengine.nlp.impact import NEWS_BONUS

MARKET_CLASSES = ("Geopolitical", "Macroeconomic")
MARKET_SEVERITY = 7  # a market-wide event is worth more than a single-company item (severity 6)


def market_impact(sentiment: float, source: str) -> int:
    raw = MARKET_SEVERITY * (0.6 + 0.4 * abs(sentiment)) + (NEWS_BONUS if source == "news" else 0)
    return int(min(10, max(1, round(raw))))


def evidence(text: str, event: str) -> list[str]:
    """Words in the text that the keyword rule for this class also looks for (the model decides
    the class; an empty list means the model had no keyword to point to)."""
    found: list[str] = []
    for label, rx in _COMPILED:
        if label == event:
            found += [" ".join(m.group(0).split()).lower() for m in rx.finditer(text)]
    return list(dict.fromkeys(found))


def market_gate(signals: pd.DataFrame, trigger: int) -> pd.DataFrame:
    """One row per Geopolitical / Macroeconomic document, with `market_impact`, `fired` and
    `reason` (the first check that failed, or how it passed)."""
    docs = (
        signals[signals["event"].isin(MARKET_CLASSES)]
        .drop_duplicates("doc_id")
        .sort_values("published_at")
        .reset_index(drop=True)
    )
    docs["market_impact"] = [
        market_impact(s, src) for s, src in zip(docs["sentiment"], docs["source"], strict=True)
    ]
    reason = pd.Series("", index=docs.index, dtype=object)
    reason[docs["sentiment"] >= 0] = "not negative"
    reason[(reason == "") & (docs["market_impact"] <= trigger)] = f"impact ≤ {trigger}"
    for i in docs.index[reason == ""]:
        row = docs.loc[i]
        earlier = docs[
            (docs["event"] == row["event"])
            & (docs["published_at"] < row["published_at"])
            & (docs["published_at"] >= row["published_at"] - CONFIRM_WINDOW)
        ]
        own = _same_text(row["text"])
        corroborated = any(_same_text(t) != own for t in earlier["text"])
        reason[i] = "fired: 2 items in 24 h" if corroborated else "1 item, no confirmation"
    docs["reason"] = reason
    docs["fired"] = reason.str.startswith("fired")
    return docs
