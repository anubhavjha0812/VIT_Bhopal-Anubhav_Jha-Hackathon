"""Impact v3 for tweets: one pre-registered fit and test (docs/experiments/IMPACT_V3.md, E28d).

uv run python -m riskengine.impact.tweets_v3 --run
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from riskengine.impact.event_study import abnormal_moves, returns
from riskengine.impact.scores import score_headlines
from riskengine.impact.train_v2 import SQRT2, expected_move, rho
from riskengine.impact.train_v3 import build
from riskengine.ingest.prices import HISTORY_PATH, MARKET, VIX, load_prices
from riskengine.ingest.tweets import load_akkerman
from riskengine.nlp.impact import impact_formula
from riskengine.paths import DATA, ROOT

CAP = 20
TRAIN = ("2020-04-09", "2020-05-15")
TEST = ("2020-06-01", "2020-07-16")
CHOSEN = "d_clipped_no_vol"
SCORES = DATA / "cache" / "tweet_v3_scores.csv.gz"
OUT = DATA / "outputs" / "impact_v3_tweets.json"
MODEL = ROOT / "models" / "impact-v3" / "impact_v3_tweets.joblib"
MIN_GAIN_CI_LOW = 0.0


def sample() -> pd.DataFrame:
    """(tweet, ticker) rows, at most CAP per stock per day (seed 0), train and test periods."""
    tw = load_akkerman()[0].explode("tickers").rename(columns={"tickers": "ticker"})
    tw["day"] = tw["published_at"].dt.strftime("%Y-%m-%d")
    tw = tw[tw["day"].between(*TRAIN) | tw["day"].between(*TEST)]
    picked = tw.groupby(["ticker", "day"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), CAP), random_state=0)
    )  # pandas drops the group keys here, so keep only the picked index
    tw = tw.loc[picked.index]
    return tw.reset_index(drop=True)


def scored(tw: pd.DataFrame) -> pd.DataFrame:
    if SCORES.exists():
        sc = pd.read_csv(SCORES, dtype={"doc_id": str})
    else:
        docs = tw.drop_duplicates("doc_id")[["doc_id", "text"]]
        sc = score_headlines(docs, SCORES)
    return tw.merge(sc, on="doc_id", how="inner")


def features(tw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prices = load_prices(HISTORY_PATH)
    rets = returns(prices[prices["ticker"] != VIX])
    tw = tw.join(abnormal_moves(tw, rets, MARKET))
    vol20 = rets.rolling(20).std().shift(1)
    tw = tw.dropna(subset=["abs_car01"]).reset_index(drop=True)
    tw["vol20"] = [vol20.at[d, t] for d, t in zip(tw["event_day"], tw["ticker"], strict=True)]
    tw["abs_sentiment"] = tw["sentiment"].abs()
    tw["title_words"] = tw["text"].str.split().str.len()
    tw["publisher"] = "other"
    tw["z"] = tw["abs_car01"] / (SQRT2 * tw["vol20"])
    cols = ["event", "publisher", "sentiment", "abs_sentiment", "title_words"]
    cols += sorted(c for c in tw if c.startswith("p_"))
    return tw.dropna(subset=[*cols, "vol20", "z"]).reset_index(drop=True), cols


def boot_gain(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int = 1000) -> list[float]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out.append(rho(a[i], y[i]) - rho(b[i], y[i]))
    return [round(float(x), 4) for x in np.quantile(out, [0.025, 0.975])]


def run() -> dict:
    assert not OUT.exists(), "the tweet test is run once"
    df, cols = features(scored(sample()))
    day = df["published_at"].dt.strftime("%Y-%m-%d")
    tr, te = df[day.between(*TRAIN)], df[day.between(*TEST)]
    model = build(CHOSEN, cols)
    used = model.num + ["event", "publisher"]
    model.fit(tr[used], np.log1p(tr["z"]))
    move = expected_move(model.predict(te[used]), te)
    formula = np.array(
        [impact_formula(e, s, "tweets") for e, s in zip(te["event"], te["sentiment"], strict=True)]
    )
    y = te["abs_car01"].to_numpy()
    r = {
        "v3_tweets": rho(move, y),
        "volatility_only": rho(te["vol20"], y),
        "formula": rho(formula, y),
    }
    gain_ci = boot_gain(move, te["vol20"].to_numpy(), y)
    adopt = (
        r["v3_tweets"] > r["volatility_only"]
        and r["v3_tweets"] > r["formula"]
        and gain_ci[0] > MIN_GAIN_CI_LOW
    )
    res = {
        "train_rows": len(tr),
        "test_rows": len(te),
        "spearman_vs_abs_car01": {k: round(v, 4) for k, v in r.items()},
        "gain_over_volatility_ci95": gain_ci,
        "adopt": bool(adopt),
        "rule": "v3 > volatility only and > formula, gain CI lower bound > 0",
    }
    if adopt:
        fit_move = expected_move(model.predict(tr[used]), tr)
        edges = np.quantile(fit_move, np.linspace(0.1, 0.9, 9))
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        pack = {"model": model, "edges": edges, "used": used, "top_publishers": []}
        joblib.dump(pack | {"kind": CHOSEN}, MODEL)
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    if ap.parse_args().run:
        print(json.dumps(run(), indent=1))
