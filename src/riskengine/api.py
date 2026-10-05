"""Signal API.

    uv run uvicorn riskengine.api:app --port 8000

GET  /health                         status + which models produced the stored signals
GET  /signals?ticker=JPM&min_impact=8 stored per-text signals (replay)
GET  /daily?ticker=JPM               daily mean sentiment, count and max impact per stock
POST /analyze {"text": "..."}        score new text live
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from riskengine.config import load_config
from riskengine.ingest.linking import Linker
from riskengine.nlp.events import classify_keywords
from riskengine.nlp.impact import impact_formula
from riskengine.nlp.sentiment import get_scorer
from riskengine.pipeline import SIGNALS_PATH, load_signals

app = FastAPI(title="Risk Engine API")


@lru_cache
def stored_signals() -> pd.DataFrame:
    if not SIGNALS_PATH.exists():
        raise HTTPException(503, "no signals yet: run `uv run python run_demo.py` first")
    return load_signals()


@lru_cache
def scorer():
    return get_scorer(load_config().sentiment)


@lru_cache
def linker() -> Linker:
    return Linker.from_csv()


class AnalyzeIn(BaseModel):
    text: str
    source: str = "news"


@app.get("/health")
def health() -> dict:
    models = stored_signals()["model"].unique().tolist() if SIGNALS_PATH.exists() else []
    return {"status": "ok", "signal_models": models}


@app.get("/signals")
def signals(
    ticker: str | None = None,
    min_impact: int = Query(1, ge=1, le=10),
    limit: int = Query(100, ge=1, le=5000),
) -> list[dict]:
    df = stored_signals()
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    df = df[df["impact"] >= min_impact].sort_values("published_at").head(limit)
    return df.assign(published_at=df["published_at"].astype(str)).to_dict("records")


@app.get("/daily")
def daily(ticker: str | None = None) -> list[dict]:
    df = stored_signals()
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    day = df["published_at"].dt.date.astype(str)
    out = (
        df.groupby([day, "ticker"])
        .agg(
            sentiment=("sentiment", "mean"), texts=("doc_id", "size"), max_impact=("impact", "max")
        )
        .round(4)
        .reset_index()
        .rename(columns={"published_at": "date"})
    )
    return out.to_dict("records")


@app.post("/analyze")
def analyze(item: AnalyzeIn) -> dict:
    s = scorer().score([item.text])[0]
    event = classify_keywords(item.text)
    return {
        "tickers": list(linker().link(item.text)),
        "sentiment": round(s, 4),
        "event": event,
        "impact": impact_formula(event, s, item.source),
        "model": f"{scorer().name}+keywords+formula",
    }
