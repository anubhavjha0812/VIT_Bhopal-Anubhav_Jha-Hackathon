"""Social posts -> clean, deduplicated, company-linked rows.

Sources:
- equinxx (Kaggle, CC0): 30 Sep 2021 - 29 Sep 2022. Its "Stock Name" column is not trusted:
  AMZN, MSFT and PG hold the same 4,080 tweets, and F and NOC are copies too.
- akkerman (Hugging Face, CC BY 4.0): 9 Apr - 16 Jul 2020, no ticker column.
Both are linked from the tweet text instead.
"""

import html
import re
from pathlib import Path

import pandas as pd

from riskengine.ingest.common import finish
from riskengine.ingest.linking import Linker
from riskengine.paths import RAW

EQUINXX_PATH = RAW / "equinxx" / "stock_tweets.csv"
AKKERMAN_PATH = RAW / "akkerman" / "stock-market-tweets-data.csv"
MAX_CASHTAGS = 5  # more than this is a watchlist or market recap, not a view on one company
RETWEET = re.compile(r"^RT @\w+:\s*")


def clean_social(
    published_at: pd.Series,
    text: pd.Series,
    linker: Linker | None = None,
    max_cashtags: int = MAX_CASHTAGS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (rows, counts at each cleaning step)."""
    linker = linker or Linker.from_csv()
    stats = {"raw": len(text)}

    df = pd.DataFrame(
        {
            "published_at": pd.to_datetime(published_at, utc=True, errors="coerce"),
            "text": text.astype(str).map(html.unescape).str.replace(RETWEET, "", regex=True),
        }
    )
    df["text"] = df["text"].str.strip()
    df = df.dropna(subset=["published_at"])
    df = df.sort_values("published_at").drop_duplicates("text", keep="first")
    stats["after_dedupe"] = len(df)

    n_tags = df["text"].map(lambda t: len(linker.cashtags_in(t)))
    df = df[n_tags <= max_cashtags]
    stats["after_spam_filter"] = len(df)

    df["tickers"] = df["text"].map(linker.link)
    df = df[df["tickers"].map(len) > 0]
    stats["linked"] = len(df)
    return finish(df, "tweets"), stats


def load_equinxx(path: Path = EQUINXX_PATH, **kw) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path)
    return clean_social(raw["Date"], raw["Tweet"], **kw)


def load_akkerman(path: Path = AKKERMAN_PATH, **kw) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path, usecols=["created_at", "text"])
    return clean_social(raw["created_at"], raw["text"], **kw)
