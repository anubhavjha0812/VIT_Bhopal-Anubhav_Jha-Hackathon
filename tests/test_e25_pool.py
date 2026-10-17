"""E25 training pool: the frame rule, the leak filter, the per-company cap, and the final guard.
Small synthetic frames stand in for FNSPID and the test sets."""

import pandas as pd
import pytest

from riskengine.eval import e25_pool
from riskengine.eval.e25_pool import (
    assert_clean,
    cap_per_company,
    exclusions,
    frame,
    remove_seen,
    settle,
)


def history(rows):
    return pd.DataFrame(rows, columns=["published_at", "text", "ticker", "why", "role"]).assign(
        published_at=lambda d: pd.to_datetime(d["published_at"], utc=True)
    )


def test_frame_keeps_subject_rows_in_window_and_dedupes_text():
    h = history([
        ("2012-05-01", "Apple launches a phone", "AAPL", None, "subject"),
        ("2012-05-02", "Apple launches a phone", "AAPL", None, "subject"),  # same text, later
        ("2012-06-01", "Bank names Apple as client", "AAPL", "not subject", "subject"),  # fails
        ("2013-01-01", "Tesla plant opens", "TSLA", None, "mention"),  # role not subject
        ("2017-03-01", "Exxon drills well", "XOM", None, "subject"),  # after the window
        ("2009-12-31", "Boeing wins order", "BA", None, "subject"),  # before the window
    ])  # fmt: skip
    f = frame(h)
    assert f["text"].tolist() == ["Apple launches a phone"]
    assert str(f.loc[0, "published_at"].date()) == "2012-05-01"  # first sighting


def test_frame_joins_companies_of_one_text():
    h = history([
        ("2012-05-01", "Apple and Google settle", "AAPL", None, "subject"),
        ("2012-05-01", "Apple and Google settle", "GOOGL", None, "subject"),
    ])  # fmt: skip
    assert frame(h).loc[0, "tickers"] == "AAPL GOOGL"


def test_remove_seen_drops_exact_and_near_duplicates_and_counts_each_set():
    f = pd.DataFrame({"text": [
        "Apple unveils a new phone today",       # exact match of a test text
        "Apple unveils a new phone today!",      # near duplicate (same words)
        "Boeing receives a large aircraft order",  # clean
    ]})  # fmt: skip
    corpus = {"dev": ["Apple unveils a new phone today"], "other": ["Unrelated text here"]}
    kept, info = remove_seen(f, corpus)
    assert kept["text"].tolist() == ["Boeing receives a large aircraft order"]
    assert info["excluded_by_set"] == {"dev": 2, "other": 0}
    assert info["excluded_total"] == 2 and info["after_exclusions"] == 1


def test_cap_per_company_limits_each_company_and_is_deterministic():
    f = pd.DataFrame({
        "text": [f"headline {i}" for i in range(10)],
        "tickers": ["AAPL"] * 6 + ["TSLA"] * 3 + ["BA AAPL"],
        "published_at": pd.date_range("2012-01-01", periods=10, tz="UTC"),
    })  # fmt: skip
    a, b = cap_per_company(f, cap=2, seed=1), cap_per_company(f, cap=2, seed=1)
    assert a.equals(b)
    primary = a["tickers"].str.split().str[0]
    assert primary.value_counts().max() <= 2
    assert set(primary) == {"AAPL", "TSLA", "BA"}
    assert a["pool_id"].is_unique and a["pool_id"].iloc[0] == "p000000"


def test_assert_clean_raises_on_a_leaked_text_and_passes_otherwise():
    corpus = {"E3": ["Apple unveils a new phone today"]}
    leaked = pd.DataFrame({"text": ["Apple unveils a new phone today", "Boeing wins order"]})
    with pytest.raises(RuntimeError, match="E3"):
        assert_clean(leaked, corpus)
    assert_clean(pd.DataFrame({"text": ["Boeing wins an aircraft order"]}), corpus)


def test_exclusions_adds_final_sample_texts(tmp_path):
    news, tweets = tmp_path / "news.csv", tmp_path / "tweets.csv"
    pd.DataFrame({"item_id": ["n0"], "text": ["final news text"]}).to_csv(news, index=False)
    pd.DataFrame({"item_id": ["t0"], "text": ["final tweet text"]}).to_csv(tweets, index=False)
    out = exclusions({"E3": ["dev text"]}, {"E24 final news": news, "E24 final tweets": tweets})
    assert out["E3"] == ["dev text"]
    assert out["E24 final news"] == ["final news text"]
    assert out["E24 final tweets"] == ["final tweet text"]


def test_pool_window_ends_before_the_earliest_test_text():
    assert e25_pool.END <= "2017-01-01"  # the E24 final news starts on this date


def test_settle_renumbers_ids_and_drops_a_leaked_row():
    pool = pd.DataFrame({
        "pool_id": ["p000000", "p000001", "p000002"],
        "text": ["Apple unveils a new phone today", "Boeing wins an order", "Tesla opens a plant"],
        "tickers": ["AAPL", "BA", "TSLA"],
    })  # fmt: skip
    out, dropped = settle(pool, {"E3": ["Apple unveils a new phone today"]})
    assert dropped == 1
    assert out["text"].tolist() == ["Boeing wins an order", "Tesla opens a plant"]
    assert out["pool_id"].tolist() == ["p000000", "p000001"]
