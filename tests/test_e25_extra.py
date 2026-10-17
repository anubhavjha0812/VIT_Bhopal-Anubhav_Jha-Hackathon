"""E25 extra rows for thin classes: keyword shortlist, no overlap with the pool, no labels."""

import pandas as pd

from riskengine.eval.e25_extra import GROUPS, hits, shortlist, without_pool


def frame_of(texts):
    return pd.DataFrame({
        "text": texts,
        "tickers": ["AAPL"] * len(texts),
        "published_at": pd.date_range("2012-01-01", periods=len(texts), tz="UTC"),
    })  # fmt: skip


def test_hits_counts_keyword_matches_ignoring_case():
    s = pd.Series(["Fed holds rates as inflation cools", "Apple unveils a phone"])
    assert hits(s, GROUPS["Macroeconomic"]).tolist() == [2, 0]


def test_shortlist_takes_keyword_rows_only_and_ranks_by_hits():
    f = frame_of([
        "Apple unveils a phone",                          # no keyword
        "Fed raises interest rates as inflation climbs",  # macro, 3 hits
        "Oil prices fall",                                # macro, 1 hit
        "Sanctions and war talks stall",                  # geo, 2 hits
    ])  # fmt: skip
    out = shortlist(f, per_group=1)
    assert out["text"].tolist() == [
        "Fed raises interest rates as inflation climbs",
        "Sanctions and war talks stall",
    ]
    assert out["shortlist"].tolist() == ["Macroeconomic", "Geopolitical"]
    assert out["extra_id"].tolist() == ["x000000", "x000001"]


def test_shortlist_keeps_a_row_picked_by_two_groups_once_and_is_deterministic():
    f = frame_of([
        "War pushes oil prices higher", "Sanctions hit the economy", "Tariff war hits oil",
    ])  # fmt: skip
    a, b = shortlist(f, per_group=5, seed=3), shortlist(f, per_group=5, seed=3)
    assert a.equals(b)
    assert len(a) == 3 and a["text"].is_unique  # each row once, though both groups match it


def test_without_pool_drops_pool_texts():
    f = frame_of(["war news", "tariff news"])
    assert without_pool(f, {"war news"})["text"].tolist() == ["tariff news"]
