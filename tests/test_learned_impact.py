from dataclasses import replace

import numpy as np
import pandas as pd

from riskengine.config import Config
from riskengine.impact import learned
from riskengine.ingest.linking import Linker
from riskengine.nlp.events import get_event_classifier
from riskengine.pipeline import score_docs


class ConstantModel:
    def predict(self, x):
        return np.full(len(x), 0.5)


def _prices() -> pd.DataFrame:
    days = pd.bdate_range("2020-01-01", periods=80)
    rng = np.random.default_rng(0)
    rows = []
    for ticker, scale in [("AAA", 0.02), ("^GSPC", 0.01), ("^VIX", 0.0)]:
        close = 100 * np.cumprod(1 + rng.normal(0, scale, len(days))) if scale else 20.0
        rows += [pd.DataFrame({"date": days, "ticker": ticker, "close": close, "volume": 1})]
    return pd.concat(rows, ignore_index=True)


def _pack() -> dict:
    used = ["event", "publisher", "sentiment", "abs_sentiment", "title_words", "n_prev_24h"]
    return {
        "model": ConstantModel(),
        "edges": np.linspace(0.01, 0.09, 9),
        "used": used + ["novelty", "vix_prev"],
        "top_publishers": [],
    }


def _rows() -> pd.DataFrame:
    when = pd.to_datetime(["2020-03-10 15:00+00:00", "2020-03-10 15:00+00:00"] * 2)
    return pd.DataFrame(
        {
            "doc_id": ["n1", "t1", "n2", "n3"],
            "published_at": when,
            "source": ["news", "tweets", "news", "news"],
            "ticker": ["AAA", "AAA", "ZZZ", "AAA"],
            "text": ["AAA beats estimates", "AAA to the moon", "ZZZ beats estimates", "x"],
        }
    ).assign(published_at=lambda d: d["published_at"] + pd.to_timedelta([0, 0, 0, 40], "D"))


def test_score_rows_scores_news_with_price_history_only(monkeypatch):
    monkeypatch.setattr(
        learned,
        "_headline_scores",
        lambda rows: pd.DataFrame(
            {"sentiment": 0.3, "event": "Earnings"}, index=rows["doc_id"].drop_duplicates()
        ),
    )
    monkeypatch.setattr(learned, "publishers", lambda: pd.Series(dtype=str))
    out = learned.score_rows(_rows(), _pack(), _prices())
    assert out["move_score"].notna().tolist() == [True, False, False, True]
    assert out["move_score"].dropna().between(1, 10).all()
    assert (out["expected_move"].dropna() > 0).all()


def test_unscored_rows_keep_the_formula_impact(monkeypatch):
    monkeypatch.setattr(
        learned,
        "score_rows",
        lambda rows, **kw: pd.DataFrame(
            {"expected_move": [0.02, np.nan], "move_score": [9.0, np.nan]}, index=rows.index
        ),
    )
    out = learned.apply(pd.DataFrame({"impact": [3, 5]}))
    assert out["impact"].tolist() == [9, 5] and out["impact_formula"].tolist() == [3, 5]


def test_learned_mode_replaces_impact_and_keeps_the_formula_column(monkeypatch):
    seen = {}

    def fake(rows):
        seen["called"] = True
        return pd.DataFrame({"expected_move": 0.02, "move_score": 7.0}, index=rows.index)

    monkeypatch.setattr(learned, "score_rows", fake)
    docs = pd.DataFrame(
        {
            "doc_id": ["a"],
            "published_at": pd.to_datetime(["2020-05-01 12:00+00:00"]),
            "source": ["news"],
            "text": ["Bank of America posts record loan losses"],
            "tickers": [["BAC"]],
        }
    )

    class Scorer:
        name = "fake"

        def score(self, texts):
            return [-0.5 for _ in texts]

    base = Config(sentiment="own")
    kw = get_event_classifier("keywords")
    plain = score_docs(docs, base, Scorer(), kw, Linker.from_csv())
    out = score_docs(docs, replace(base, impact="learned"), Scorer(), kw, Linker.from_csv())
    assert seen["called"] and out["move_score"].tolist() == [7.0]
    assert out["impact"].tolist() == [7]  # the learned score replaces the formula
    assert out["impact_formula"].tolist() == plain["impact"].tolist()
    assert "move_score" not in plain
