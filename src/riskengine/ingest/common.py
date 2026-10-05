"""Shared output schema for every text source."""

import hashlib

import pandas as pd

COLUMNS = ["doc_id", "published_at", "source", "text", "tickers"]


def doc_id(ts: pd.Timestamp, text: str) -> str:
    return hashlib.sha1(f"{ts.isoformat()}|{text}".encode()).hexdigest()[:16]


def finish(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add source + doc_id and return rows in the shared column order."""
    df = df.copy()
    df["source"] = source
    df["doc_id"] = [doc_id(ts, t) for ts, t in zip(df["published_at"], df["text"], strict=True)]
    return df[COLUMNS].reset_index(drop=True)
