"""E17 history set: FNSPID headlines, 2010 – 10 Jun 2020, for our 15 companies.

uv run python -m riskengine.history --export   # unique headlines for the Kaggle scoring job
uv run python -m riskengine.history            # merge the Kaggle scores into the history set
uv run python -m riskengine.history --export-tweets / --tweets   # the same for the 2021-22 tweets

One row per (headline, company): whether FNSPID filed it under that company (`by_symbol`), the
first relevance rule it fails (`why`, empty if kept), the company's role (E12) and the engine's
sentiment for the headline. Syndicated copies (same day, same text) count once.
"""

import logging

import pandas as pd

from riskengine.ingest.fetch_fnspid import NAMED_PATH
from riskengine.ingest.linking import Linker
from riskengine.ingest.relevance import reason
from riskengine.ingest.tweets import load_equinxx
from riskengine.paths import DATA

log = logging.getLogger(__name__)
START, END = "2010-01-01", "2020-06-11"  # end is exclusive: FNSPID stops on 10 Jun 2020
HISTORY_PATH = DATA / "cache" / "history_signals.csv.gz"
TEXTS_PATH = DATA / "cache" / "history_texts.csv.gz"  # input of the Kaggle scoring job
SCORES_PATH = DATA / "cache" / "history_scores.csv.gz"  # its output: text_id, sentiment
TWEET_TEXTS_PATH = DATA / "cache" / "tweet_texts.csv.gz"
TWEET_SCORES_PATH = DATA / "cache" / "tweet_scores.csv.gz"
TWEET_SIGNALS_PATH = DATA / "cache" / "tweet_signals.csv.gz"


def pairs(path=NAMED_PATH, linker: Linker | None = None) -> pd.DataFrame:
    """(headline, company) pairs with the rule outcome and role; no sentiment yet."""
    linker = linker or Linker.from_csv()
    raw = pd.read_csv(path, dtype=str, usecols=["Date", "Article_title", "Stock_symbol"])
    df = pd.DataFrame(
        {
            "published_at": pd.to_datetime(raw["Date"], utc=True, errors="coerce"),
            "text": raw["Article_title"].fillna("").str.strip(),
            "symbol": raw["Stock_symbol"].map(linker.cashtag_to_ticker),
        }
    )
    df = df[(df["published_at"] >= START) & (df["published_at"] < END) & (df["text"] != "")]
    df["day"] = df["published_at"].dt.date
    heads = (
        df.sort_values("published_at")
        .groupby(["day", "text"], as_index=False)
        .agg(published_at=("published_at", "first"), symbols=("symbol", lambda s: set(s.dropna())))
    )
    heads["ticker"] = [
        sorted(syms | set(linker.link(t)))
        for syms, t in zip(heads["symbols"], heads["text"], strict=True)
    ]
    rows = heads.explode("ticker").dropna(subset=["ticker"])
    rows["by_symbol"] = [tk in s for tk, s in zip(rows["ticker"], rows["symbols"], strict=True)]
    rows["why"] = [
        reason(linker, t, tk, by_symbol=b)
        for t, tk, b in zip(rows["text"], rows["ticker"], rows["by_symbol"], strict=True)
    ]
    rows["role"] = [
        "analyst" if tk in linker.analyst_only(t) else "subject"
        for t, tk in zip(rows["text"], rows["ticker"], strict=True)
    ]
    cols = ["published_at", "text", "ticker", "by_symbol", "why", "role"]
    return rows[cols].sort_values("published_at", kind="stable").reset_index(drop=True)


def export_texts(rows: pd.DataFrame | None = None, path=TEXTS_PATH) -> pd.DataFrame:
    """Unique texts to score, for the Kaggle job (notebooks/history_sentiment)."""
    rows = pairs() if rows is None else rows
    texts = rows["text"].drop_duplicates().reset_index(drop=True)
    out = pd.DataFrame({"text_id": texts.index, "text": texts})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def build(
    rows: pd.DataFrame | None = None,
    texts_path=TEXTS_PATH,
    scores_path=SCORES_PATH,
    path=HISTORY_PATH,
) -> pd.DataFrame:
    """Pairs plus the engine's sentiment for every text (kept or not, so the rules' effect can
    be shown), from the scores the Kaggle job wrote."""
    rows = pairs() if rows is None else rows
    texts = pd.read_csv(texts_path)
    scores = pd.read_csv(scores_path).merge(texts, on="text_id", validate="one_to_one")
    by_text = dict(zip(scores["text"], scores["sentiment"], strict=True))
    rows["sentiment"] = rows["text"].map(by_text)
    if rows["sentiment"].isna().any():
        raise ValueError(f"{rows['sentiment'].isna().sum()} pairs have no score")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.round({"sentiment": 4}).to_csv(path, index=False)
    return rows


def tweet_pairs(linker: Linker | None = None) -> pd.DataFrame:
    """E17 fresh test: equinxx tweets (30 Sep 2021 - 29 Sep 2022), linked by their own text
    (`load_equinxx`), one row per (tweet, company), with the E12 role."""
    linker = linker or Linker.from_csv()
    rows = load_equinxx(linker=linker)[0].explode("tickers").rename(columns={"tickers": "ticker"})
    rows = rows.dropna(subset=["ticker"])
    rows["by_symbol"], rows["why"] = False, None
    rows["role"] = [
        "analyst" if tk in linker.analyst_only(t) else "subject"
        for t, tk in zip(rows["text"], rows["ticker"], strict=True)
    ]
    cols = ["published_at", "text", "ticker", "by_symbol", "why", "role"]
    return rows[cols].sort_values("published_at", kind="stable").reset_index(drop=True)


def load_history(path=HISTORY_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values={"sentiment": [""]})
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["why"] = df["why"].replace("", None)
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--export" in sys.argv:
        export_texts()
    elif "--export-tweets" in sys.argv:
        export_texts(tweet_pairs(), TWEET_TEXTS_PATH)
    elif "--tweets" in sys.argv:
        build(tweet_pairs(), TWEET_TEXTS_PATH, TWEET_SCORES_PATH, TWEET_SIGNALS_PATH)
    else:
        build()
