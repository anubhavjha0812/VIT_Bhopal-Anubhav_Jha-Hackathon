"""Replay the stored window as a time-ordered event stream (served by the API as SSE).

Events, in the order a live desk would have seen them:
- signal: one text with its tickers, sentiment, event class and impact
- alert: the same text again when its impact is above the trigger (Module B input)
- weights: Module A weights for a trading day, sent once every text they use has been sent

Module A's no-look-ahead rule decides the order: weights for trading day T use only texts
with a New York date before T, so they are sent just before the first text dated T or later.
"""

from collections.abc import Iterator

import pandas as pd

from riskengine.module_a import NY

Event = tuple[str, dict]


def texts(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per text (signals have one row per text and ticker), in time order."""
    return (
        signals.groupby("doc_id", sort=False)
        .agg(
            published_at=("published_at", "first"),
            source=("source", "first"),
            text=("text", "first"),
            tickers=("ticker", list),
            sentiment=("sentiment", "first"),
            event=("event", "first"),
            impact=("impact", "first"),
        )
        .reset_index()
        .sort_values(["published_at", "doc_id"], kind="stable")
        .reset_index(drop=True)
    )


def events(signals: pd.DataFrame, weights: pd.DataFrame, trigger: int) -> Iterator[Event]:
    """`weights` has a `date` column (trading day) and one column per ticker."""
    days = pd.to_datetime(weights["date"])
    by_day = weights.drop(columns="date").round(4)
    pending = 0  # index of the next weights row to send

    def weights_until(day) -> Iterator[Event]:
        nonlocal pending
        while pending < len(days) and (day is None or days.iloc[pending] <= day):
            row = by_day.iloc[pending].to_dict()
            yield "weights", {"date": days.iloc[pending].date().isoformat(), "weights": row}
            pending += 1

    for t in texts(signals).itertuples(index=False):
        ny_day = t.published_at.tz_convert(NY).tz_localize(None).normalize()
        yield from weights_until(ny_day)
        payload = {
            "doc_id": t.doc_id,
            "published_at": t.published_at.isoformat(),
            "source": t.source,
            "text": t.text,
            "tickers": t.tickers,
            "sentiment": t.sentiment,
            "event": t.event,
            "impact": int(t.impact),
        }
        yield "signal", payload
        if t.impact > trigger:
            yield "alert", payload
    yield from weights_until(None)


def sse(kind: str, data: str) -> str:
    """One Server-Sent Events message."""
    return f"event: {kind}\ndata: {data}\n\n"
