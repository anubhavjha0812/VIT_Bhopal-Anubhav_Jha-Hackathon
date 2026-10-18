"""Learned impact (v3) in the engine: expected move of the stock after a news headline (E28).

uv run python -m riskengine.impact.learned --fit    # train on 2009-2018, save the small model
uv run python -m riskengine.impact.learned --check  # replay-window check against the real move

The model needs the stock's recent volatility, so it scores news rows of stocks with price
history only. Other rows get NaN and keep the formula impact. The 1-10 `impact` column (used by
the credit gate and Module B) is not changed; the learned score is `move_score`.
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskengine.impact.event_study import abnormal_moves, event_day_index, returns
from riskengine.impact.features import (
    CATEGORICAL,
    build_dataset,
    headlines_last_24h,
    novelty,
    publishers,
)
from riskengine.impact.scores import SCORES_PATH, score_headlines
from riskengine.impact.train_v2 import (
    FIRST_YEAR,
    LAST_FIT_YEAR,
    SQRT2,
    expected_move,
    features,
    rho,
)
from riskengine.impact.train_v3 import build
from riskengine.ingest.fetch_fnspid import OUT_PATH as FNSPID_SLICE
from riskengine.ingest.prices import HISTORY_PATH, MARKET, VIX, load_prices
from riskengine.paths import DATA, ROOT

log = logging.getLogger(__name__)

MODEL_PATH = ROOT / "models" / "impact-v3" / "impact_v3.joblib"
TWEET_SCORES = DATA / "cache" / "tweet_v3_scores.csv.gz"
TWEET_MODEL_PATH = ROOT / "models" / "impact-v3" / "impact_v3_tweets.joblib"
CHECK_PATH = DATA / "outputs" / "impact_v3_live_check.json"
CHOSEN = "d_clipped_no_vol"  # frozen in data/outputs/impact_v3_frozen.json before the test


def fit() -> dict:
    """Refit the frozen choice on 2009-2018 rows; the decile edges give the 1-10 scale."""
    frozen = json.loads((DATA / "outputs" / "impact_v3_frozen.json").read_text())
    assert frozen["chosen"] == CHOSEN, "the frozen choice changed"
    df = build_dataset()
    cols = features(df)
    top = sorted(set(df["publisher"]) - {"other"})
    rows = df[df["year"].between(FIRST_YEAR, LAST_FIT_YEAR)]
    model = build(CHOSEN, cols)
    used = model.num + CATEGORICAL
    model.fit(rows[used], np.log1p(rows["z"]))
    move = expected_move(model.predict(rows[used]), rows)
    edges = np.quantile(move, np.linspace(0.1, 0.9, 9))
    pack = {"model": model, "edges": edges, "used": used, "top_publishers": top, "kind": CHOSEN}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pack, MODEL_PATH)
    return {"rows": len(rows), "edges": [float(e) for e in edges]}


def _headline_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Own sentiment, E2 event and topic probabilities per doc_id (cache first, then compute)."""
    ids = rows["doc_id"].drop_duplicates()
    caches = [
        pd.read_csv(p, dtype={"doc_id": str}) for p in (SCORES_PATH, TWEET_SCORES) if p.exists()
    ]
    parts = [c[c["doc_id"].isin(ids)] for c in caches]
    have = pd.concat(parts) if parts else pd.DataFrame({"doc_id": []})
    missing = ids[~ids.isin(have["doc_id"])]
    if len(missing):
        new = rows.drop_duplicates("doc_id").set_index("doc_id").loc[missing].reset_index()
        with tempfile.TemporaryDirectory() as tmp:
            parts.append(score_headlines(new[["doc_id", "text"]], Path(tmp) / "scores.csv.gz"))
    return pd.concat(parts).drop_duplicates("doc_id").set_index("doc_id")


def _score(rows: pd.DataFrame, source: str, pack: dict, market: tuple) -> pd.DataFrame:
    """Expected |move| and move_score for the rows of one source (news or tweets)."""
    rets, vol20, vix = market
    out = pd.DataFrame(np.nan, index=rows.index, columns=["expected_move", "move_score"])
    news = rows[(rows["source"] == source) & rows["ticker"].isin(rets.columns)].copy()
    if news.empty:
        return out
    pos = event_day_index(news["published_at"], rets.index)
    news = news[pos < len(rets.index)].copy()
    news["event_day"] = rets.index[pos[pos < len(rets.index)]]
    news["vol20"] = [vol20.at[d, t] for d, t in zip(news["event_day"], news["ticker"], strict=True)]
    news["vix_prev"] = news["event_day"].map(vix)

    sc = _headline_scores(news)
    for col in sc.columns:
        news[col] = news["doc_id"].map(sc[col])
    news = news.reset_index().rename(columns={"index": "_row"})
    news["abs_sentiment"] = news["sentiment"].abs()
    news["title_words"] = news["text"].str.split().str.len()
    if "n_prev_24h" in pack["used"]:  # the tweet model does not use these two
        news["n_prev_24h"] = headlines_last_24h(news)
        news["novelty"] = novelty(news)
    pubs = publishers() if FNSPID_SLICE.exists() and source == "news" else pd.Series(dtype=str)
    news["publisher"] = news["doc_id"].map(pubs).fillna("other")
    news.loc[~news["publisher"].isin(pack["top_publishers"]), "publisher"] = "other"

    ok = news.dropna(subset=[c for c in pack["used"] if c != "publisher"] + ["vol20"])
    ok = ok[ok["vol20"] > 0]
    move = expected_move(pack["model"].predict(ok[pack["used"]]), ok)
    out.loc[ok["_row"].to_numpy(), "expected_move"] = move
    out.loc[ok["_row"].to_numpy(), "move_score"] = np.digitize(move, pack["edges"]) + 1
    return out


def score_rows(
    rows: pd.DataFrame,
    pack: dict | None = None,
    prices: pd.DataFrame | None = None,
    tweet_pack: dict | None = None,
) -> pd.DataFrame:
    """Expected |move| and 1-10 move_score per (doc, ticker) row; NaN where not possible.

    News rows use the news model, tweet rows the tweet model (E28d) if it was adopted.
    """
    if pack is None:
        pack = joblib.load(MODEL_PATH)
        if tweet_pack is None and TWEET_MODEL_PATH.exists():
            tweet_pack = joblib.load(TWEET_MODEL_PATH)
    prices = load_prices(HISTORY_PATH) if prices is None else prices
    rets = returns(prices[prices["ticker"] != VIX])
    market = (
        rets,
        rets.rolling(20).std().shift(1),
        prices[prices["ticker"] == VIX].set_index("date")["close"].shift(1),
    )
    out = _score(rows, "news", pack, market)
    if tweet_pack is not None:
        tw = _score(rows, "tweets", tweet_pack, market)
        out = out.fillna(tw)
    return out


def apply(rows: pd.DataFrame, **kw) -> pd.DataFrame:
    """Rows plus `impact_formula`, `expected_move`, `move_score`; `impact` = move_score if set."""
    out = rows.copy()
    if "impact_formula" not in out:
        out["impact_formula"] = out["impact"]
    scored = score_rows(out, **kw)
    out["expected_move"], out["move_score"] = scored["expected_move"], scored["move_score"]
    out["impact"] = out["move_score"].fillna(out["impact_formula"]).round().astype(int)
    return out


def check(path: Path = CHECK_PATH) -> dict:
    """Replay window: does the score order the real next-two-day moves better than the formula?"""
    from riskengine.pipeline import SIGNALS_PATH, load_signals

    sig = load_signals(SIGNALS_PATH)
    sig = sig[sig["source"] == "news"].reset_index(drop=True)
    sig["impact"] = sig.get("impact_formula", sig["impact"])  # the formula, also on applied files
    sig = sig.drop(columns=["impact_formula", "expected_move", "move_score"], errors="ignore")
    prices = load_prices(HISTORY_PATH)
    scored = score_rows(sig, prices=prices)
    sig = sig.join(scored)
    rets = returns(prices[prices["ticker"] != VIX])
    sig = sig.join(abnormal_moves(sig, rets, MARKET))
    sig["vol20"] = [
        rets.rolling(20).std().shift(1).at[d, t] if pd.notna(d) else np.nan
        for d, t in zip(sig["event_day"], sig["ticker"], strict=True)
    ]
    t = sig.dropna(subset=["move_score", "abs_car01", "vol20"])
    res = {
        "rows_news": len(sig),
        "rows_scored": int(sig["move_score"].notna().sum()),
        "rows_with_real_move": len(t),
        "spearman_vs_abs_car01": {
            "formula_impact": round(rho(t["impact"], t["abs_car01"]), 4),
            "volatility_only": round(rho(t["vol20"], t["abs_car01"]), 4),
            "v3_expected_move": round(rho(t["expected_move"], t["abs_car01"]), 4),
        },
        "note": "Apr-Jun 2020 replay window, news rows only; 2020 was already seen in E28 "
        "(development check, not a blind test). Sqrt2 scaling: " + str(round(SQRT2, 4)),
    }
    path.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply-signals", type=Path, help="add the learned impact to a signals file")
    a = ap.parse_args()
    if a.apply_signals:
        from riskengine.pipeline import load_signals

        sig = apply(load_signals(a.apply_signals))
        sig.to_json(a.apply_signals, orient="records", lines=True, date_format="iso")
        print({"rows": len(sig), "scored": int(sig["move_score"].notna().sum())})
    if a.fit:
        print(json.dumps(fit(), indent=1))
    if a.check:
        print(json.dumps(check(), indent=1))
