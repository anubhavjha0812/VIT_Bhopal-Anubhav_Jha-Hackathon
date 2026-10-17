"""Market gate: Geo/Macro items fire only if negative, above the trigger and confirmed."""

import pandas as pd

from riskengine.market_gate import market_gate, market_impact


def row(doc, event, s, source, t, text, ticker="AAPL"):
    return dict(doc_id=doc, event=event, sentiment=s, source=source, ticker=ticker, text=text,
                published_at=pd.Timestamp(t, tz="UTC"))  # fmt: skip


def test_market_impact_can_pass_seven_only_for_strong_news():
    assert market_impact(-0.9, "news") == 8
    assert market_impact(-0.9, "tweets") == 7  # a tweet alone never passes 7
    assert market_impact(-0.2, "news") <= 7


def test_fires_when_negative_strong_and_confirmed():
    s = pd.DataFrame(
        [
            row("a", "Geopolitical", -0.9, "news", "2020-04-01 09:00", "War escalates in region X"),
            row("b", "Geopolitical", -0.9, "news", "2020-04-01 12:00", "Fighting spreads, markets fall"),  # noqa: E501
            row("b", "Geopolitical", -0.9, "news", "2020-04-01 12:00", "Fighting spreads, markets fall", "MSFT"),  # noqa: E501
        ]
    )  # fmt: skip
    g = market_gate(s, 7).set_index("doc_id")
    assert len(g) == 2  # one row per document
    assert not g.loc["a", "fired"] and g.loc["a", "reason"].startswith("1 item")
    assert g.loc["b", "fired"] and g.loc["b", "reason"] == "fired: 2 items in 24 h"


def test_positive_news_and_other_classes_do_not_fire():
    s = pd.DataFrame(
        [
            row("a", "Macroeconomic", 0.9, "news", "2020-04-01 09:00", "Growth beats forecasts"),
            row("b", "Macroeconomic", 0.9, "news", "2020-04-01 10:00", "Jobs surge, rates steady"),
            row("c", "Earnings", -0.9, "news", "2020-04-01 10:30", "Profit collapses"),
        ]
    )
    g = market_gate(s, 7)
    assert set(g["doc_id"]) == {"a", "b"} and not g["fired"].any()
    assert (g["reason"] == "not negative").all()


def test_same_text_does_not_confirm_itself_and_old_items_do_not_count():
    s = pd.DataFrame(
        [
            row("a", "Macroeconomic", -0.9, "news", "2020-04-01 09:00", "Oil plunges"),
            row("b", "Macroeconomic", -0.9, "news", "2020-04-01 10:00", "OIL PLUNGES"),
            row("c", "Macroeconomic", -0.9, "news", "2020-04-03 10:00", "Recession fears grow"),
        ]
    )
    g = market_gate(s, 7)
    assert not g["fired"].any()


def test_evidence_lists_keywords_of_the_class_only():
    from riskengine.market_gate import evidence

    assert evidence("Fed warns of recession as inflation rises", "Macroeconomic") == [
        "fed", "recession", "inflation"
    ]  # fmt: skip
    assert evidence("Fed warns of recession", "Geopolitical") == []
    assert evidence("Oil plunges", "Macroeconomic") == []
