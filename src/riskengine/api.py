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
GET  /gate                           credit gate: credit / high-impact rows, fired or why not
GET  /scoreboard                     held-out test scores of every model (docs/RESULTS.md)
GET  /prices?ticker=JPM              daily closes in the replay window
GET  /module-a/equity                growth of $1 per strategy, after costs
GET  /module-b/alerts?episode=covid2020 gated alerts with loss and CET1 before → after
GET  /module-b?doc_id=...            one alert through Module B: capital, Monte Carlo, memo
GET  /module-b?ticker=BA&notches=2   a what-if downgrade (no ticker: the market episode alone)
GET  /module-b/book                  the 31 Dec 2019 book, one row per position
GET  /module-b/checks                predicted vs actual: real alerts, crises, monthly (E14, E15)
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from riskengine.config import load_config
from riskengine.gate import gate
from riskengine.ingest.gdelt import GdeltClient
from riskengine.ingest.linking import Linker
from riskengine.ingest.prices import load_prices
from riskengine.live import LiveFeed
from riskengine.market_gate import evidence, market_gate
from riskengine.module_a import NY, Backtest, run_module_a
from riskengine.module_b import report as module_b
from riskengine.module_b.backtest import ALERTS_PATH, CRISES_PATH
from riskengine.module_b.book import load_book
from riskengine.module_b.outlook import MONTHLY_PATH
from riskengine.module_b.outlook import summary as monthly_summary
from riskengine.module_b.shock import DEFAULT_EPISODE, parse_notches
from riskengine.nlp.event_engine import get_engine_classifier
from riskengine.nlp.events import (
    CREDIT_CLASSES,
    OWN_TOPIC,
    is_analyst_rating,
    names_agency,
)
from riskengine.nlp.impact import impact_formula
from riskengine.nlp.sentiment import OWN_SENTIMENT, get_scorer
from riskengine.paths import DATA
from riskengine.pipeline import SIGNALS_PATH, load_signals, subjects
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
    if live_feed.cache_info().currsize:
        live_feed().close()


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


def gated_signals() -> pd.DataFrame:
    """Stored signals with the credit gate decision per row (`fired`, `reason`)."""
    return gate(stored_signals(), load_config().impact_trigger, linker())


@lru_cache
def scorer():
    return get_scorer(load_config().sentiment)


@lru_cache
def event_classifier():
    return get_engine_classifier(load_config().events)


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
    events = OWN_TOPIC.resolve().name if config.events != "keywords" else "keywords (baseline)"
    if config.events == "model_rules":
        events += " + rules (E24 candidate C)"
    return {"sentiment": sentiment, "events": events}


Episode = Literal["gfc2008", "euro2011", "covid2020"]


@lru_cache
def book() -> pd.DataFrame:
    return load_book()


def fired_alerts() -> pd.DataFrame:
    df = gated_signals()
    return df[df["fired"]].sort_values("published_at")


def alert_row(row: pd.Series) -> dict:
    """A fired (text, obligor) row as the memo quotes it, with the parsed notches."""
    return {
        "doc_id": row["doc_id"],
        "published_at": str(row["published_at"]),
        "source": row["source"],
        "ticker": row["ticker"],
        "text": row["text"],
        "event": row["event"],
        "impact": int(row["impact"]),
        "gate": row["reason"],
        "notches": parse_notches(row["text"], row["event"]),
        "matched": module_b.matched_words(row["text"]),
    }


@lru_cache(maxsize=256)
def module_b_summary(ticker: str, notches: int, episode: str, multiple: float) -> dict:
    return module_b.summary(book(), ticker, notches, episode, multiple, load_config().rates)


@lru_cache(maxsize=64)
def module_b_run(ticker: str | None, notches: int, episode: str, multiple: float) -> dict:
    return module_b.run(book(), ticker, notches, episode, multiple, load_config().rates)


def alert_capital(ticker: str, text: str, event: str, impact: int) -> dict:
    """Module B in the replay: loss and CET1 before → after on the default episode."""
    if ticker not in set(book()["ticker"]):
        return {}
    multiple = module_b.alert_multiple(impact, load_config().shock)
    c = module_b_summary(ticker, parse_notches(text, event), DEFAULT_EPISODE, multiple)
    return {k: c[k] for k in ("loss_usd", "cet1_before", "cet1_after")}


class AnalyzeIn(BaseModel):
    text: str
    source: str = "news"


@lru_cache
def live_feed() -> LiveFeed:
    return LiveFeed(GdeltClient(linker()), lambda text: analyze(AnalyzeIn(text=text)))


@app.get("/live")
def live_signals(
    ticker: str | None = None, min_impact: int = Query(1, ge=1, le=10),
) -> dict:
    feed = live_feed()
    feed.start()
    return feed.snapshot(ticker, min_impact)


@app.post("/live/refresh")
def refresh_live() -> dict:
    live_feed().refresh()
    return {"status": "scheduled"}


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
    df = gated_signals()
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    df = df[df["impact"] >= min_impact].sort_values("published_at").head(limit)
    return df.assign(published_at=df["published_at"].astype(str)).to_dict("records")


@app.get("/daily")
def daily(ticker: str | None = None) -> list[dict]:
    df = subjects(stored_signals())
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
    analysts = linker().analyst_only(item.text) & set(linker().link(item.text))
    return {
        "tickers": [t for t in linker().link(item.text) if t not in analysts],
        "analysts": sorted(analysts),
        "sentiment": round(s, 4),
        "event": event,
        "analyst_rating": is_analyst_rating(item.text),
        "impact": impact_formula(event, s, item.source, names_agency(item.text)),
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
    `ticker` keeps texts that mention it. Alerts are the credit gate's (gate.py).
    """
    df = gated_signals()
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

    async def stream():
        sent = 0
        for kind, payload in events(df, weights, alert_capital):
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
    df = gated_signals()
    texts = df.drop_duplicates("doc_id")
    return {
        "config": asdict(config),
        "engine_models": engine_models(config),
        "signal_models": df["model"].unique().tolist(),
        "texts": len(texts),
        "signals": len(df),
        "texts_by_source": {k: int(v) for k, v in texts["source"].value_counts().items()},
        "events": {k: int(v) for k, v in texts["event"].value_counts().items()},
        # before the gate: any text above the trigger; after: texts the credit gate passes
        "alerts_naive": int((texts["impact"] > config.impact_trigger).sum()),
        "alerts": int(df.loc[df["fired"], "doc_id"].nunique()),
        "trading_days": len(module_a_outputs()[0]),
    }


@app.get("/gate")
def gate_table() -> list[dict]:
    """Every row the naive trigger (impact > 7) or the gate looks at: credit class or high
    impact, with `fired` and the `reason` (the first check that failed, or how it passed)."""
    config = load_config()
    df = gated_signals()
    df = df[df["event"].isin(CREDIT_CLASSES) | (df["impact"] > config.impact_trigger)]
    cols = ["doc_id", "published_at", "source", "ticker", "role", "text", "event", "impact"]
    df = df[[*cols, "fired", "reason"]].sort_values("published_at")
    return df.assign(published_at=df["published_at"].astype(str)).to_dict("records")


@app.get("/scoreboard")
def scoreboard() -> dict:
    """Held-out test scores per task. Per-class F1 only where it has intervals (E16 rows)."""

    def summary(row: dict) -> dict:
        out = {k: row[k] for k in SCORE_FIELDS}
        if "per_class_f1_ci95" in row:
            out |= {k: row[k] for k in ("per_class_f1", "per_class_f1_ci95")}
        return out

    # E18 rows are per (model, test set) and live in docs/RESULTS.md, not on the scoreboard.
    tasks = {t: rows for t, rows in eval_results().items() if "(E18)" not in t}
    return {task: [summary(r) for r in rows] for task, rows in tasks.items()}


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


@app.get("/module-b/alerts")
def module_b_alerts(episode: Episode = DEFAULT_EPISODE) -> list[dict]:
    """Every gated alert with its loss and CET1 before → after on the episode."""
    shock = load_config().shock
    out = []
    for _, row in fired_alerts().iterrows():
        a = alert_row(row)
        multiple = module_b.alert_multiple(a["impact"], shock)
        c = module_b_summary(a["ticker"], a["notches"], episode, multiple)
        out.append({**a, "episode": episode, "multiple": multiple, **c})
    return out


@app.get("/module-b/market-alerts")
def module_b_market_alerts(episode: Episode = DEFAULT_EPISODE) -> list[dict]:
    """Geopolitical / Macroeconomic items that fired the market gate, with the market-wide stress
    (no single obligor): loss and CET1 before → after on the episode, scaled by config.shock."""
    cfg = load_config()
    fired = market_gate(stored_signals(), cfg.impact_trigger)
    out = []
    for _, row in fired[fired["fired"]].iterrows():
        multiple = module_b.alert_multiple(int(row["market_impact"]), cfg.shock)
        c = module_b_summary(None, 0, episode, multiple)
        out.append(
            {
                "doc_id": row["doc_id"],
                "published_at": str(row["published_at"]),
                "source": row["source"],
                "text": row["text"],
                "event": row["event"],
                "impact": int(row["market_impact"]),
                "gate": row["reason"],
                "matched": evidence(row["text"], row["event"]),
                "episode": episode,
                "multiple": multiple,
                **c,
            }
        )
    return out


@app.get("/module-b")
def module_b_report(
    doc_id: str | None = None,
    ticker: str | None = None,
    notches: int = Query(1, ge=0, le=99, description="what-if cut; 99 = default"),
    episode: Episode = DEFAULT_EPISODE,
    multiple: float | None = Query(None, ge=0, le=5, description="× the episode"),
) -> dict:
    """An alert (`doc_id`, plus `ticker` if it fired for several) or a what-if (`ticker`,
    `notches`; no ticker = the market episode alone). The multiple defaults to config.shock."""
    alert = None
    if doc_id:
        rows = fired_alerts()
        rows = rows[rows["doc_id"] == doc_id]
        if ticker:
            rows = rows[rows["ticker"] == ticker.upper()]
        if rows.empty:
            raise HTTPException(404, f"no gated alert {doc_id!r}")
        alert = alert_row(rows.iloc[0])
        ticker, notches = alert["ticker"], alert["notches"]
        if multiple is None:
            multiple = module_b.alert_multiple(alert["impact"], load_config().shock)
    elif ticker:
        ticker = ticker.upper()
        if ticker not in set(book()["ticker"]):
            raise HTTPException(404, f"{ticker} is not in the book")
    if not ticker:
        notches = 0
    rep = module_b_run(ticker, notches, episode, 1.0 if multiple is None else multiple)
    return {**rep, "alert": alert, "memo": module_b.memo(rep, alert)}


@app.get("/module-b/book")
def module_b_book() -> list[dict]:
    return book().to_dict("records")


@lru_cache
def module_b_checks() -> dict:
    paths = (ALERTS_PATH, CRISES_PATH, MONTHLY_PATH)
    if not all(p.exists() for p in paths):
        raise HTTPException(
            503, "no checks yet: run `uv run python -m riskengine.module_b.backtest` and `.outlook`"
        )
    alerts, crises, monthly = (pd.read_csv(p) for p in paths)
    mae = {
        k: {
            "loss": crises[f"{k}_loss_error"].mean(),
            "cet1": crises[f"{k}_cet1_error"].mean(),
            "sign": crises[f"{k}_sign_match"].mean(),
        }
        for k in ("predicted", "naive")
    }
    loss_cols = ["realised_loss", "model_loss", "naive_loss"]
    return {
        "alerts": alerts.drop(columns="published_at").to_dict("records"),
        "crises": {"rows": crises.to_dict("records"), "mae": mae},
        "monthly": {
            "summary": monthly_summary(monthly),
            "rows": monthly[["origin", "baa", *loss_cols]].round(1).to_dict("records"),
        },
    }


@app.get("/module-b/checks")
def module_b_checks_endpoint() -> dict:
    """Predicted vs actual, each scored once: real alerts and 12 crises (E14), 345 months (E15)."""
    return module_b_checks()
