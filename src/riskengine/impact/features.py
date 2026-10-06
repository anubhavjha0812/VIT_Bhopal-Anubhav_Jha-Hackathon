"""One row per (headline, stock): features known at publication + the real move (DECISIONS E6)."""

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


def publishers() -> pd.Series:
    """doc_id -> publisher, rebuilt from the raw FNSPID slice (same doc_id recipe as the loader)."""
    raw = pd.read_csv(FNSPID_SLICE, dtype=str)
    ts = pd.to_datetime(raw["Date"], utc=True, errors="coerce")
    text = raw["Article_title"].fillna("").str.strip()
    ok = ts.notna() & (text != "")
    ids = [doc_id(t, x) for t, x in zip(ts[ok], text[ok], strict=True)]
    pub = raw.loc[ok, "Publisher"].str.replace("_", " ").str.strip()
    return pd.Series(pub.to_numpy(), index=ids).groupby(level=0).first()


def headlines_last_24h(rows: pd.DataFrame) -> np.ndarray:
    """Headlines for the same stock in the 24 h *before* each row (no look-ahead)."""
    out = np.zeros(len(rows), dtype=int)
    for _, g in rows.groupby("ticker"):
        t = g["published_at"].astype("int64").to_numpy()
        order = np.argsort(t)
        ts = t[order]
        lo = np.searchsorted(ts, ts - 24 * 3600 * 10**9, side="left")
        hi = np.searchsorted(ts, ts, side="left")
        counts = np.empty_like(order)
        counts[order] = hi - lo
        out[rows.index.get_indexer(g.index)] = counts
    return out


def build_dataset() -> pd.DataFrame:
    news = load_fnspid()[0]
    if GOLD_SAMPLE.exists():  # E3: gold headlines never used for fitting
        news = news[~news["doc_id"].isin(pd.read_csv(GOLD_SAMPLE, dtype=str)["doc_id"])]
    scores = load_scores()
    rows = (
        news.explode("tickers")
        .rename(columns={"tickers": "ticker"})
        .merge(scores, on="doc_id", how="inner")
        .reset_index(drop=True)
    )

    pubs = publishers()
    rows["publisher"] = rows["doc_id"].map(pubs).fillna("other")
    top = rows["publisher"].value_counts().index[:TOP_PUBLISHERS]
    rows.loc[~rows["publisher"].isin(top), "publisher"] = "other"

    rows["abs_sentiment"] = rows["sentiment"].abs()
    rows["title_words"] = rows["text"].str.split().str.len()
    rows["n_prev_24h"] = headlines_last_24h(rows)

    prices = load_prices(HISTORY_PATH)
    rets = returns(prices[prices["ticker"] != VIX])
    rows = rows.join(abnormal_moves(rows, rets, MARKET))
    rows = rows.dropna(subset=["abs_car01"]).reset_index(drop=True)

    # state of the market on the day *before* the event day
    vol20 = rets.rolling(20).std().shift(1)
    vix = prices[prices["ticker"] == VIX].set_index("date")["close"].shift(1)
    rows["vol20"] = [vol20.at[d, t] for d, t in zip(rows["event_day"], rows["ticker"], strict=True)]
    rows["vix_prev"] = rows["event_day"].map(vix)

    rows["year"] = rows["published_at"].dt.year
    rows["y"] = np.log1p(rows["abs_car01"] * 100)  # fit target: log(1 + |CAR| in %)
    return rows.dropna(subset=NUMERIC).reset_index(drop=True)
