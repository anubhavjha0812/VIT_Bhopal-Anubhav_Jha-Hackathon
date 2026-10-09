import pandas as pd
import pytest

from riskengine.gate import gate
from riskengine.ingest.linking import Linker
from riskengine.nlp.events import credit_event, is_analyst_rating, names_agency


@pytest.fixture(scope="module")
def linker():
    return Linker.from_csv()


@pytest.mark.parametrize(
    ("text", "event", "analyst_rating"),
    [
        ("JPMorgan Chase & Co. Downgrades iRobot $IRBT to Underweight", None, True),
        ("iRobot $IRBT Downgraded by JPMorgan Chase & Co. to Underweight", None, True),
        ("Goldman Sachs Downgrades Apple to Sell, Lowers Price Target to $233", None, True),
        ("Fitch downgrades Disney's IDR to 'A-' from 'A'", "Credit downgrade", False),
        ("Moody's cuts Boeing to junk as coronavirus hits demand", "Credit downgrade", False),
        ("$DIS - Disney's credit rating lowered at S&P", "Credit downgrade", False),
        ("Google pays Apple to be the default search engine on iOS", None, False),
        ("JCPenney misses interest payment, prepares Chapter 11 filing", "Default", False),
        ("JPMorgan profit falls 20% as loan losses rise", None, False),
        ("Dow, S&P Jump Over 3% on Encouraging Vaccine Data", None, False),
        # amended after the demo check (E13): payments far from "default", hedged bankruptcy,
        # curly apostrophe
        ("Google pays Apple to be the default search engine; such payments top $7B", None, False),
        ("$xom wont go bankrupt, neither will most decent size producers", None, False),
        ("There's rumors that Neiman Marcus may file for bankruptcy", None, False),
        ("$JPM and $WFC better believe credit defaults will be over $12B", "Default", False),
        ("On April 2nd, Moody’s cut Exxon's top credit rating", "Credit downgrade", False),
    ],
)
def test_credit_rules(text, event, analyst_rating):
    assert credit_event(text) == event
    assert is_analyst_rating(text) == analyst_rating


def test_the_sp_index_is_not_a_rating_agency():
    assert not names_agency("S&P 500 falls as Fed cuts rates")
    assert names_agency("S&P lowers Ford credit rating to junk")


def test_bank_research_desk_is_an_analyst_even_without_rating_words(linker):
    assert linker.analyst_only("Bank of America analysts see oil at $40") == {"BAC"}
    assert linker.analyst_only("Bank of America cuts 1,000 jobs") == set()


def rows(*items):
    """(minutes, ticker, text, event, impact[, role]) -> signals rows."""
    base = pd.Timestamp("2020-05-01 12:00", tz="UTC")
    return pd.DataFrame(
        [
            {
                "doc_id": f"d{i}",
                "published_at": base + pd.Timedelta(minutes=it[0]),
                "ticker": it[1],
                "text": it[2],
                "event": it[3],
                "impact": it[4],
                "role": it[5] if len(it) > 5 else "subject",
            }
            for i, it in enumerate(items)
        ]
    )


def test_agency_named_fires(linker):
    out = gate(
        rows((0, "DIS", "Fitch downgrades Disney's IDR to 'A-' $DIS", "Credit downgrade", 8)),
        7,
        linker,
    )
    assert out["fired"].tolist() == [True]
    assert out["reason"].tolist() == ["fired: rating agency"]


def test_one_rumour_is_blocked_two_sources_fire(linker):
    one = rows((0, "BA", "Boeing said to be near default on its loans", "Default", 9))
    assert gate(one, 7, linker)["reason"].tolist() == ["1 source, no agency"]
    two = rows(
        (0, "BA", "Boeing said to be near default on its loans", "Default", 6),
        (90, "BA", "Boeing misses a loan payment, lenders say", "Default", 9),
    )
    assert gate(two, 7, linker)["reason"].tolist() == ["impact ≤ 7", "fired: 2 sources in 24 h"]


def test_not_obligor_analyst_bank_and_non_credit_are_blocked(linker):
    out = gate(
        rows(
            (0, "XOM", "$DO $XOM $CVX: Diamond Offshore files for bankruptcy", "Default", 8),
            (
                1,
                "JPM",
                "Fitch: iRobot debt cut to junk by JPMorgan",
                "Credit downgrade",
                9,
                "analyst",
            ),
            (2, "AAPL", "Goldman downgrades Apple to Sell", "Other", 4),
        ),
        7,
        linker,
    )
    assert out["reason"].tolist() == ["not the obligor", "analyst bank", "not credit"]
    assert not out["fired"].any()
