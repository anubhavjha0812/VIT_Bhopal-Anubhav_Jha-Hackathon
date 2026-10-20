"""Replay the stored window as a time-ordered event stream (served by the API as SSE).

Events, in the order a live desk would have seen them:
- signal: one text with its tickers (subjects), analysts, sentiment, event class and impact
- alert: the same text again when the credit gate fires for one of its companies
  (Module B input; `signals` must carry the gate's `fired` and `reason`, see gate.py), with
  Module B's result per obligor when `on_alert` is given
- weights: Module A weights for a trading day, sent once every text they use has been sent

Module A's no-look-ahead rule decides the order: weights for trading day T use only texts
with a New York date before T, so they are sent just before the first text dated T or later.
"""

from collections.abc import Callable, Iterator

import pandas as pd

from riskengine.module_a import NY

Event = tuple[str, dict]
OnAlert = Callable[[str, str, str, int], dict]  # (ticker, text, event, impact) -> Module B result


def texts(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per text (signals have one row per text and ticker), in time order."""
    if "role" not in signals:
        signals = signals.assign(role="subject")
    if "fired" not in signals:
        signals = signals.assign(fired=False, reason="")
    if "impact_formula" not in signals:  # E28: the alert size follows severity, not the stock move
        signals = signals.assign(impact_formula=signals["impact"])
    out = (
        signals.groupby("doc_id", sort=False)
        .agg(
            published_at=("published_at", "first"),
            source=("source", "first"),
            text=("text", "first"),
            tickers=("ticker", list),
            roles=("role", list),
            fired=("fired", list),
            reasons=("reason", list),
            sentiment=("sentiment", "first"),
            event=("event", "first"),
            impact=("impact", "first"),
            impact_formula=("impact_formula", "first"),
        )
        .reset_index()
        .sort_values(["published_at", "doc_id"], kind="stable")
        .reset_index(drop=True)
    )
    # per text: (ticker, role, fired, reason) for each company it names
    per_text = [
        list(zip(*cols, strict=True))
        for cols in zip(
            out.pop("tickers"), out.pop("roles"), out.pop("fired"), out.pop("reasons"), strict=True
        )
    ]
    # A bank that only gave a stock rating is shown apart, not as a subject (DECISIONS E12).
    out["tickers"] = [[t for t, role, _, _ in rows if role == "subject"] for rows in per_text]
    out["analysts"] = [[t for t, role, _, _ in rows if role == "analyst"] for rows in per_text]
    out["obligors"] = [[t for t, _, fired, _ in rows if fired] for rows in per_text]
    out["gate"] = [next((why for _, _, f, why in rows if f), "") for rows in per_text]
    return out


def events(
    signals: pd.DataFrame, weights: pd.DataFrame, on_alert: OnAlert | None = None
) -> Iterator[Event]:
    """`weights` has a `date` column (trading day) and one column per ticker."""
    return stream(texts(signals), weights, on_alert)


def stream(
    per_text: pd.DataFrame, weights: pd.DataFrame, on_alert: OnAlert | None = None
) -> Iterator[Event]:
    """Like `events`, from the frame `texts` returns, so a caller can build it once."""
    days = pd.to_datetime(weights["date"])
    by_day = weights.drop(columns="date").round(4)
    pending = 0  # index of the next weights row to send

    def weights_until(day) -> Iterator[Event]:
        nonlocal pending
        while pending < len(days) and (day is None or days.iloc[pending] <= day):
            row = by_day.iloc[pending].to_dict()
            yield "weights", {"date": days.iloc[pending].date().isoformat(), "weights": row}
            pending += 1

    for t in per_text.itertuples(index=False):
        ny_day = t.published_at.tz_convert(NY).tz_localize(None).normalize()
        yield from weights_until(ny_day)
        payload = {
            "doc_id": t.doc_id,
            "published_at": t.published_at.isoformat(),
            "source": t.source,
            "text": t.text,
            "tickers": t.tickers,
            "analysts": t.analysts,
            "sentiment": t.sentiment,
            "event": t.event,
            "impact": int(t.impact),
        }
        yield "signal", payload
        if t.obligors:
            alert = {**payload, "obligors": t.obligors, "gate": t.gate}
            if on_alert:
                alert["module_b"] = {
                    o: on_alert(o, t.text, t.event, int(t.impact_formula)) for o in t.obligors
                }
            yield "alert", alert
    yield from weights_until(None)


def sse(kind: str, data: str) -> str:
    """One Server-Sent Events message."""
    return f"event: {kind}\ndata: {data}\n\n"
