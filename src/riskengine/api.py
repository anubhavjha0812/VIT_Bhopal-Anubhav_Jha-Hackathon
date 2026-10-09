"""Signal API.

    uv run uvicorn riskengine.api:app --port 8000

GET  /health                         status + which models produced the stored signals
GET  /signals?ticker=JPM&min_impact=8 stored per-text signals (replay)
GET  /daily?ticker=JPM               daily mean sentiment, count and max impact per stock
POST /analyze {"text": "..."}        score new text live
GET  /tickers                        stocks that have signals
GET  /module-a/weights               Module A weights per day (set at that day's close)
GET  /module-a/metrics               equal weight vs sentiment tilt: return, vol, Sharpe, drawdown
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from riskengine.config import load_config
from riskengine.ingest.linking import Linker
from riskengine.nlp.events import get_event_classifier
from riskengine.nlp.impact import impact_formula
from riskengine.nlp.sentiment import get_scorer
from riskengine.paths import DATA
from riskengine.pipeline import SIGNALS_PATH, load_signals

app = FastAPI(title="Risk Engine API")

OUTPUTS = DATA / "outputs"


@lru_cache
def stored_signals() -> pd.DataFrame:
    if not SIGNALS_PATH.exists():
        raise HTTPException(503, "no signals yet: run `uv run python run_demo.py` first")
    return load_signals()  # load_signals returns df


@lru_cache
def scorer():
    return get_scorer(load_config().sentiment)


@lru_cache
def event_classifier():
    return get_event_classifier(load_config().events)


@lru_cache
def linker() -> Linker:
    return Linker.from_csv()


@lru_cache
def module_a_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights_path = OUTPUTS / "weights_history.csv"
    metrics_path = OUTPUTS / "module_a_metrics.csv"

    if not weights_path.exists():
        raise HTTPException(503, "no Module A output yet: run ` uv run python run_demo.py ` first")
    return pd.read_csv(weights_path), pd.read_csv(metrics_path, index_col=0)


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
    event = event_classifier()([item.text])[0]
    return {
        "tickers": list(linker().link(item.text)),
        "sentiment": round(s, 4),
        "event": event,
        "impact": impact_formula(event, s, item.source),
        "model": f"{scorer().name}+{load_config().events}+formula",
    }


@app.get("/tickers")
def tickers() -> list[str]:
    return sorted(stored_signals()["ticker"].unique().tolist())


@app.get("/module-a/weights")
def module_a_weights() -> list[dict]:
    """One row per day, e.g. [{"date": "2020-04-09", "AAPL": 0.0937, ...}, ...]."""
    weights, _ = module_a_outputs()  # it returns two df
    return weights.to_dict(orient="records")  # gives json like op of the df inside a list


@app.get("/module-a/metrics")
def module_a_metrics() -> dict:
    """One entry per strategy, e.g. {"equal": {"total_return": 0.2552, "sharpe": 4.88, ...}}."""
    _, metrics = module_a_outputs()
    return metrics.round(4).to_dict(orient="index")
