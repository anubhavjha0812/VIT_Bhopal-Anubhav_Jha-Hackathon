"""Signal API.

    uv run uvicorn riskengine.api:app --port 8000

GET  /health                         status + which models produced the stored signals
GET  /signals?ticker=JPM&min_impact=8 stored per-text signals (replay)
GET  /daily?ticker=JPM               daily mean sentiment, count and max impact per stock
POST /analyze {"text": "..."}        score new text live
GET  /tickers                        stocks that have signals
GET  /module-a/weights               Module A weights per day (set at that day's close)
GET  /module-a/metrics               equal weight vs sentiment tilt: return, vol, Sharpe, drawdown
GET  /replay?rate=20&start=2020-05-01 Server-Sent Events: the window replayed in time order
GET  /overview                       engine run at a glance: config, counts, event mix, alerts
GET  /scoreboard                     held-out test scores of every model (docs/RESULTS.md)
GET  /prices?ticker=JPM              daily closes in the replay window
GET  /module-a/equity                growth of $1 per strategy, after costs
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from riskengine.config import load_config
from riskengine.ingest.linking import Linker
from riskengine.ingest.prices import load_prices
from riskengine.module_a import NY, Backtest, run_module_a
from riskengine.nlp.events import OWN_TOPIC, get_event_classifier
from riskengine.nlp.impact import impact_formula
from riskengine.nlp.sentiment import OWN_SENTIMENT, get_scorer
from riskengine.paths import DATA
from riskengine.pipeline import SIGNALS_PATH, load_signals
from riskengine.replay import events, sse

log = logging.getLogger(__name__)


def warm_up() -> None:
    """Load the live models once, so the first /analyze answers in milliseconds, not ~20 s."""
    try:
        scorer().score(["warm up"])
        event_classifier()(["warm up"])
    except Exception as e:  # a missing model must not stop the API; /analyze reports it
        log.warning("model warm-up failed: %s", e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.get_running_loop().run_in_executor(None, warm_up)  # in the background
    yield


app = FastAPI(title="Risk Engine API", lifespan=lifespan)

OUTPUTS = DATA / "outputs"
SCORE_FIELDS = [
    "model", "n", "accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95",
    "texts_per_s", "model_size_mb",
]  # fmt: skip


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


@lru_cache
def eval_results() -> dict:
    path = OUTPUTS / "eval_results.json"
    if not path.exists():
        raise HTTPException(503, "no test scores yet: run `uv run python -m riskengine.eval` first")
    return json.loads(path.read_text())


@lru_cache
def window_prices() -> pd.DataFrame:
    config = load_config()
    df = load_prices()
    start, end = pd.Timestamp(config.window_start), pd.Timestamp(config.window_end)
    return df[(df["date"] >= start) & (df["date"] <= end)]


@lru_cache
def module_a_run() -> dict[str, Backtest]:
    """Module A re-run on the stored signals (43 days, well under a second)."""
    return run_module_a(stored_signals(), load_prices(), load_config())


def engine_models(config) -> dict[str, str]:
    """Scoreboard names of the models the engine runs, so the dashboard can mark them."""
    sentiment = OWN_SENTIMENT.resolve().name if config.sentiment == "own" else "finbert (baseline)"
    events = OWN_TOPIC.resolve().name if config.events == "model" else "keywords (baseline)"
    return {"sentiment": sentiment, "events": events}


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


@app.get("/replay")
async def replay(
    rate: float = Query(20.0, gt=0, le=1000, description="texts per second"),
    start: date | None = None,
    after: datetime | None = None,
    ticker: str | None = None,
    limit: int | None = Query(None, ge=1, description="stop after this many texts"),
) -> StreamingResponse:
    """Events `signal`, `alert` (impact above the trigger), `weights` (Module A) and `end`.

    `start` is the first New York date to replay. `after` resumes a paused replay from that
    time (texts at exactly that time are sent again; the client skips ids it has seen).
    `ticker` keeps texts that mention it.
    """
    df = stored_signals()
    weights, _ = module_a_outputs()
    begin = pd.Timestamp(start, tz=NY) if start else None
    if after:
        resume = pd.Timestamp(after)
        resume = resume.tz_localize("UTC") if resume.tzinfo is None else resume
        begin = max(begin, resume) if begin is not None else resume
    if begin is not None:
        df = df[df["published_at"] >= begin]
        first_day = begin.tz_convert(NY).tz_localize(None).normalize()
        weights = weights[pd.to_datetime(weights["date"]) >= first_day]
    if ticker:
        # keep every ticker of a matching text, so the event shows who else it names
        df = df[df["doc_id"].isin(df.loc[df["ticker"] == ticker.upper(), "doc_id"])]
    trigger = load_config().impact_trigger

    async def stream():
        sent = 0
        for kind, payload in events(df, weights, trigger):
            if kind == "signal":
                if limit and sent == limit:
                    break
                if sent:
                    await asyncio.sleep(1 / rate)
                sent += 1
            yield sse(kind, json.dumps(payload))
        yield sse("end", json.dumps({"texts": sent}))

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@app.get("/overview")
def overview() -> dict:
    """Engine run at a glance: config, models, counts per source and event class, alerts."""
    config = load_config()
    df = stored_signals()
    texts = df.drop_duplicates("doc_id")
    return {
        "config": asdict(config),
        "engine_models": engine_models(config),
        "signal_models": df["model"].unique().tolist(),
        "texts": len(texts),
        "signals": len(df),
        "texts_by_source": {k: int(v) for k, v in texts["source"].value_counts().items()},
        "events": {k: int(v) for k, v in texts["event"].value_counts().items()},
        "alerts": int((texts["impact"] > config.impact_trigger).sum()),
        "trading_days": len(module_a_outputs()[0]),
    }


@app.get("/scoreboard")
def scoreboard() -> dict:
    """Held-out test scores per task, without the per-class detail."""
    return {
        task: [{k: row[k] for k in SCORE_FIELDS} for row in rows]
        for task, rows in eval_results().items()
    }


@app.get("/prices")
def prices(ticker: str) -> list[dict]:
    """Daily closes in the replay window, e.g. [{"date": "2020-04-09", "close": 66.99}, ...]."""
    df = window_prices()
    df = df[df["ticker"] == ticker.upper()]
    if df.empty:
        raise HTTPException(404, f"no prices for {ticker!r}")
    return [
        {"date": d.date().isoformat(), "close": round(float(c), 4)}
        for d, c in zip(df["date"], df["close"], strict=True)
    ]


@app.get("/module-a/equity")
def module_a_equity() -> list[dict]:
    """Growth of $1 after costs, e.g. [{"date": "2020-04-09", "equal": 1.0, "sentiment": 1.0}, ...].

    Weights set at a day's close earn the next day's return, so each value is dated the day
    the return is known.
    """
    runs = module_a_run()
    days = next(iter(runs.values())).weights.index
    curves = {k: [1.0, *(1 + bt.returns).cumprod().round(5).tolist()] for k, bt in runs.items()}
    return [
        {"date": d.date().isoformat(), **{k: v[i] for k, v in curves.items()}}
        for i, d in enumerate(days)
    ]
