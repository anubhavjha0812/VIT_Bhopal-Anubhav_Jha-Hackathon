"""Kaggle stock tweets (equinxx, CC0) -> clean, deduplicated, company-linked rows.

The dataset's own "Stock Name" column is not trusted: AMZN, MSFT and PG hold the same
4,080 tweets, and F and NOC are copies too. Tweets are linked from their text instead.
"""

import hashlib
import html
from pathlib import Path

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.paths import RAW

TWEETS_PATH = RAW / "equinxx" / "stock_tweets.csv"
MAX_CASHTAGS = 5  # more than this is a watchlist or market recap, not a view on one company


def _doc_id(ts: pd.Timestamp, text: str) -> str:
    return hashlib.sha1(f"{ts.isoformat()}|{text}".encode()).hexdigest()[:16]


def load_tweets(
    path: Path = TWEETS_PATH, linker: Linker | None = None, max_cashtags: int = MAX_CASHTAGS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (rows, counts at each cleaning step)."""
    linker = linker or Linker.from_csv()
    raw = pd.read_csv(path)
    stats = {"raw": len(raw)}

    df = pd.DataFrame(
        {
            "published_at": pd.to_datetime(raw["Date"], utc=True),
            "text": raw["Tweet"].astype(str).map(html.unescape).str.strip(),
        }
    )
    df = df.sort_values("published_at").drop_duplicates("text", keep="first")
    stats["after_dedupe"] = len(df)

    n_tags = df["text"].map(lambda t: len(linker.cashtags_in(t)))
    df = df[n_tags <= max_cashtags]
    stats["after_spam_filter"] = len(df)

    df["tickers"] = df["text"].map(linker.link)
    df = df[df["tickers"].map(len) > 0]
    stats["linked"] = len(df)

    df["source"] = "tweets"
    df["doc_id"] = [_doc_id(ts, t) for ts, t in zip(df["published_at"], df["text"], strict=True)]
    cols = ["doc_id", "published_at", "source", "text", "tickers"]
    return df[cols].reset_index(drop=True), stats
