"""E25 re-test sample: random plus keyword strata, no row twice, deterministic, blind sheet."""

import pandas as pd

from riskengine.eval.e25_retest import STRATA, draw, hit


def frame_of(texts):
    return pd.DataFrame({
        "text": texts,
        "tickers": ["AAPL"] * len(texts),
        "published_at": pd.date_range("2019-01-01", periods=len(texts), tz="UTC"),
    })  # fmt: skip


def test_hit_matches_case_insensitively():
    s = pd.Series(["Fed holds interest rates", "Apple unveils a phone"])
    assert hit(s, STRATA["Macroeconomic"]).tolist() == [True, False]
    assert hit(s, STRATA["Product news"]).tolist() == [False, True]


def test_draw_has_random_and_stratum_rows_each_row_once_and_is_deterministic():
    texts = [f"Fed raises interest rates story {i}" for i in range(30)]
    texts += [f"Company unveils new gadget {i}" for i in range(30)]
    texts += [f"Quiet filler headline {i}" for i in range(60)]
    f = frame_of(texts)
    strata = {k: STRATA[k] for k in ("Macroeconomic", "Product news")}
    a = draw(f, strata, n_random=20, per_stratum=5, seed=4)
    b = draw(f, strata, n_random=20, per_stratum=5, seed=4)
    assert a.equals(b)
    assert a["text"].is_unique and a["item_id"].is_unique
    assert (a["stratum"] == "random").sum() == 20
    assert (a["stratum"] == "Macroeconomic").sum() <= 5
    assert a["item_id"].iloc[0] == "r3_0000"
