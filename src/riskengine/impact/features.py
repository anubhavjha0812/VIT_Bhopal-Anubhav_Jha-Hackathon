"""One row per (headline, stock): features known at publication + the real move (DECISIONS E6)."""

from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.gold import TO_LABEL as GOLD_SAMPLE
from riskengine.impact.event_study import abnormal_moves, returns
from riskengine.impact.scores import load_scores
from riskengine.ingest.common import doc_id
from riskengine.ingest.fetch_fnspid import OUT_PATH as FNSPID_SLICE
from riskengine.ingest.news import load_fnspid
from riskengine.ingest.prices import HISTORY_PATH, MARKET, VIX, load_prices

TOP_PUBLISHERS = 10
CATEGORICAL = ["event", "publisher"]
NUMERIC = ["sentiment", "abs_sentiment", "title_words", "n_prev_24h", "vol20", "vix_prev"]
NUMERIC_V2 = [*NUMERIC, "novelty"]  # DECISIONS E8
NOVELTY_WINDOW_H = 72


def publishers(path: Path = FNSPID_SLICE) -> pd.Series:
    """doc_id -> publisher, rebuilt from the raw FNSPID slice (same doc_id recipe as the loader)."""
    raw = pd.read_csv(path, dtype=str)
    ts = pd.to_datetime(raw["Date"], utc=True, errors="coerce")
    text = raw["Article_title"].fillna("").str.strip()
    ok = ts.notna() & (text != "")
    ids = [doc_id(t, x) for t, x in zip(ts[ok], text[ok], strict=True)]
    pub = raw.loc[ok, "Publisher"].str.replace("_", " ").str.strip()
    return pd.Series(pub.to_numpy(), index=ids).groupby(level=0).first()


def _ns(ts: pd.Series) -> np.ndarray:
    """Timestamps as int nanoseconds (pandas 3 may store microseconds; be explicit)."""
    return ts.dt.as_unit("ns").astype("int64").to_numpy()


def headlines_last_24h(rows: pd.DataFrame) -> np.ndarray:
    """Headlines for the same stock in the 24 h *before* each row (no look-ahead)."""
    out = np.zeros(len(rows), dtype=int)
    for _, g in rows.groupby("ticker"):
        t = _ns(g["published_at"])
        order = np.argsort(t)
        ts = t[order]
        lo = np.searchsorted(ts, ts - 24 * 3600 * 10**9, side="left")
        hi = np.searchsorted(ts, ts, side="left")
        counts = np.empty_like(order)
        counts[order] = hi - lo
        out[rows.index.get_indexer(g.index)] = counts
    return out


def novelty(rows: pd.DataFrame, hours: int = NOVELTY_WINDOW_H) -> np.ndarray:
    """1 - max cosine similarity to the same stock's headlines in the previous `hours` (1 if none).

    Word 1-2-gram hashing vectoriser: nothing is fitted, so no information leaks across time.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(ngram_range=(1, 2), n_features=2**18, alternate_sign=False)
    out = np.ones(len(rows))
    for _, g in rows.groupby("ticker"):
        g = g.sort_values("published_at", kind="stable")
        t = _ns(g["published_at"])
        lo = np.searchsorted(t, t - hours * 3600 * 10**9, side="left")
        hi = np.searchsorted(t, t, side="left")  # strictly earlier headlines only
        x = vec.transform(g["text"])
        pos = rows.index.get_indexer(g.index)
        for i in np.flatnonzero(hi > lo):
            sim = (x[lo[i] : hi[i]] @ x[i].T).toarray()
            out[pos[i]] = 1 - sim.max()
    return np.clip(out, 0.0, 1.0)  # float rounding can give -2e-16 for identical text


def build_dataset(
    news: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    publisher_src: Path = FNSPID_SLICE,
    top_publishers: list[str] | None = None,
) -> pd.DataFrame:
    """Defaults: our 15 stocks (E6). Pass news/scores/prices for another set (E8 test F),
    with the training set's `top_publishers` so the publisher column means the same thing."""
    if news is None:
        news = load_fnspid()[0]
        if GOLD_SAMPLE.exists():  # E3: gold headlines never used for fitting
            news = news[~news["doc_id"].isin(pd.read_csv(GOLD_SAMPLE, dtype=str)["doc_id"])]
    scores = load_scores() if scores is None else scores
    rows = (
        news.explode("tickers")
        .rename(columns={"tickers": "ticker"})
        .merge(scores, on="doc_id", how="inner")
        .reset_index(drop=True)
    )

    pubs = publishers(publisher_src)
    rows["publisher"] = rows["doc_id"].map(pubs).fillna("other")
    top = top_publishers or rows["publisher"].value_counts().index[:TOP_PUBLISHERS]
    rows.loc[~rows["publisher"].isin(top), "publisher"] = "other"

    rows["abs_sentiment"] = rows["sentiment"].abs()
    rows["title_words"] = rows["text"].str.split().str.len()
    rows["n_prev_24h"] = headlines_last_24h(rows)
    rows["novelty"] = novelty(rows)

    prices = load_prices(HISTORY_PATH) if prices is None else prices
    rets = returns(prices[prices["ticker"] != VIX])
    rows = rows.join(abnormal_moves(rows, rets, MARKET))
    rows = rows.dropna(subset=["abs_car01"]).reset_index(drop=True)

    # state of the market on the day *before* the event day
    vol20 = rets.rolling(20).std().shift(1)
    vix = prices[prices["ticker"] == VIX].set_index("date")["close"].shift(1)
    rows["vol20"] = [vol20.at[d, t] for d, t in zip(rows["event_day"], rows["ticker"], strict=True)]
    rows["vix_prev"] = rows["event_day"].map(vix)

    rows["year"] = rows["published_at"].dt.year
    rows["y"] = np.log1p(rows["abs_car01"] * 100)  # v1 fit target: log(1 + |CAR| in %)
    rows["z"] = rows["abs_car01"] / (np.sqrt(2) * rows["vol20"])  # v2: move in normal 2-day moves
    return rows.dropna(subset=NUMERIC).reset_index(drop=True)
