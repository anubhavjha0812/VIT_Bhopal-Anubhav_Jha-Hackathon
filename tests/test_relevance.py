"""E17 relevance rules, on headlines from the design sample (seed 7)."""

import pytest

from riskengine.ingest.linking import Linker
from riskengine.ingest.relevance import reason

L = Linker.from_csv()


@pytest.mark.parametrize(
    ("ticker", "text"),
    [
        ("MSFT", "Microsoft to buy Yammer for $1.2 bln"),
        ("AAPL", "UPDATE 3-Apple cuts orders of iPhone 5C as consumers prefer 5S"),
        ("TSLA", "Tesla Announces Pricing of Initial Public Offering"),
        ("XOM", "ExxonMobil Announces Multi-Year Supply Agreement With Caterpillar"),
        ("BAC", "UPDATE 1-Bank of America $8.5 bln mortgage pact challenged"),
        ("KO", "The Coca-Cola Company`s Haiti Hope Project Momentum Continues"),
        ("JPM", "JPMorgan sells $1.25 bln in FDIC-backed debt"),
    ],
)
def test_kept(ticker, text):
    assert reason(L, text, ticker) is None


@pytest.mark.parametrize(
    ("ticker", "text", "why"),
    [
        ("TSLA", "Tesla столкнулась с новыми проблемами", "not English"),
        ("JPM", "REG - JPMorgan Russ - Net Asset Value(s)", "fund notice"),
        (
            "DIS",
            "Marsico Capital Management Llc Buys Walt Disney Co, Celgene, Wells Fargo, Sells "
            "Chipotle ...",
            "holdings list",
        ),  # fmt: skip
        ("BAC", "Caxton Associates Sells Bank of America, Monster Beverage", "holdings list"),
        (
            "JPM",
            "Eaton To Present At JPMorgan Basics And Industrials Conference On June 3",
            "bank conference",
        ),  # fmt: skip
        ("BAC", "Bank of America Upgrades Automatic Data Processing to Buy", "bank as analyst"),
        ("JPM", "J.P. Morgan Adjusts PRE 1Q And FY 2011 Operating Estimates", "bank as analyst"),
        (
            "BAC",
            "STXNEWS LATAM-Gafisa shares attractive due to strong cash flow-BofA",
            "bank as analyst",
        ),  # fmt: skip
        ("KO", "Mikuni Coca-Cola <2572.T>-2010 group results", "another company's name"),
        (
            "DIS",
            "BRIEF-Euro Disney announces results of capital increase",
            "another company's name",
        ),  # fmt: skip
        (
            "GOOGL",
            "VigLink Secures Funding From First Round Capital, Google Ventures, Angel Investors",
            "another company's name",
        ),  # fmt: skip
        ("KO", "Aujan und die Coca-Cola Company kÃ¼ndigen Unterzeichnung", "not English"),
        ("TSLA", "Auto Stock Roundup: GM, Honda, Tesla And More", "market list"),
        ("MSFT", "Stocks To Watch: Spotlight On Microsoft, Tesla And Xilinx", "market list"),
        ("MSFT", "INRIX Appoints Former Microsoft CFO John Connors to Board", "former employer"),
        ("BAC", "BofA's Stock Picks For Income Funds In Earnings Season", "bank as analyst"),
        (
            "BA",
            "U.S. Navy's 'unfunded priorities' list has Boeing, Lockheed fighters",
            "named late",
        ),  # fmt: skip
    ],
)
def test_dropped(ticker, text, why):
    assert reason(L, text, ticker) == why


def test_source_links_only_get_the_language_rule():
    text = "Seneca Foods to Present at the Bank of America Merrill Lynch Conference"
    assert reason(L, text, "BAC") == "bank conference"
    assert reason(L, text, "BAC", by_symbol=True) is None
    assert reason(L, "Tesla столкнулась с новыми проблемами", "TSLA", by_symbol=True) == (
        "not English"
    )


def test_wilson_interval():
    from riskengine.eval.relevance import wilson

    w = wilson(82, 100)
    assert w["share"] == 0.82
    assert w["ci95"][0] < 0.82 < w["ci95"][1]
    assert abs(w["ci95"][0] - 0.733) < 0.002 and abs(w["ci95"][1] - 0.883) < 0.002
    assert wilson(0, 0)["share"] is None
