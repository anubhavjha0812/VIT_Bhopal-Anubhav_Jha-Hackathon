"""FNSPID headlines -> clean, deduplicated, company-linked rows.

A headline keeps its FNSPID ticker (old symbols mapped: FB -> META, GOOG -> GOOGL) when it is
one of ours, and also gets any universe company named in the title. A headline filed under
another symbol is kept only if it names one of ours (the --by-name slice, DECISIONS E17).
"""

from pathlib import Path

import pandas as pd

from riskengine.ingest.common import finish
from riskengine.ingest.fetch_fnspid import OUT_PATH
from riskengine.ingest.linking import Linker


def load_fnspid(
    path: Path = OUT_PATH, linker: Linker | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (rows, counts at each cleaning step)."""
    linker = linker or Linker.from_csv()
    raw = pd.read_csv(path, dtype=str)
    stats = {"raw": len(raw)}

    df = pd.DataFrame(
        {
            "published_at": pd.to_datetime(raw["Date"], utc=True, errors="coerce"),
            "text": raw["Article_title"].fillna("").str.strip(),
            "symbol": raw["Stock_symbol"].map(linker.cashtag_to_ticker),
        }
    )
    df = df.dropna(subset=["published_at"])
    df = df[df["text"] != ""]

    df["tickers"] = [
        tuple(sorted({*([sym] if isinstance(sym, str) else []), *linker.link(text)}))
        for sym, text in zip(df["symbol"], df["text"], strict=True)
    ]
    df = df[df["tickers"].map(len) > 0]
    # Same headline syndicated under several symbols on one day -> one row, tickers merged.
    df["day"] = df["published_at"].dt.date
    df = (
        df.sort_values("published_at")
        .groupby(["day", "text"], as_index=False)
        .agg(
            published_at=("published_at", "first"),
            tickers=("tickers", lambda ts: tuple(sorted(set().union(*ts)))),
        )
        .sort_values("published_at", kind="stable")
    )
    stats["after_dedupe"] = len(df)
    return finish(df, "news"), stats
