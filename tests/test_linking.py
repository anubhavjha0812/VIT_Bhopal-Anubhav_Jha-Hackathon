import pytest

from riskengine.ingest.linking import Linker


@pytest.fixture(scope="module")
def linker():
    return Linker.from_csv()


def test_universe_has_15_stocks(linker):
    assert len(linker.tickers) == 15


def test_cashtag_is_case_insensitive(linker):
    assert linker.link("buying more $aapl today") == ("AAPL",)


def test_old_cashtags_map_to_current_ticker(linker):
    assert linker.link("$FB and $GOOG down") == ("GOOGL", "META")


def test_company_name_links(linker):
    assert linker.link("JPMorgan raises loan-loss reserves") == ("JPM",)


def test_company_name_uses_the_canonical_universe_name(linker):
    assert linker.company_name("AAPL") == "Apple Inc."
    assert linker.company_name("BAC") == "Bank of America Corp."
    assert linker.company_name("UNKNOWN") is None


def test_linker_without_company_names_keeps_existing_linking_working(linker):
    old = Linker(
        linker.cashtag_to_ticker,
        linker.alias_patterns,
        linker.analyst_patterns,
        linker.desk_patterns,
    )
    assert old.link("Apple $AAPL") == ("AAPL",)
    assert old.company_name("AAPL") is None


def test_name_needs_word_boundary(linker):
    assert linker.link("pineapple prices rise") == ()


def test_unknown_cashtag_ignored(linker):
    assert linker.link("$SPY $QQQ rally") == ()


def test_names_can_be_turned_off(linker):
    assert linker.link("Tesla deliveries beat", use_names=False) == ()


@pytest.mark.parametrize(
    ("text", "subjects", "analysts"),
    [
        # the bank only gives the rating -> analyst, not subject (DECISIONS E12)
        ("iRobot $IRBT Downgraded by JPMorgan Chase & Co. to Underweight", (), ("JPM",)),
        ("BofA Downgrades Tesla, Says Company 'Faces Several Hurdles'", ("TSLA",), ("BAC",)),
        ("AutoZone $AZO Lowered to “Neutral” at Bank of America", (), ("BAC",)),
        (
            "Netflix shares rise after Bank of America analysts raised their price target",
            ("NFLX",),
            ("BAC",),
        ),
        ("Apple $AAPL Buy Rating Reaffirmed at JPMorgan Chase & Co.", ("AAPL",), ("JPM",)),
        # the bank is what the text is about -> subject
        ("CFRA Maintains Buy on Bank of America, Lowers Price Target to $28", ("BAC",), ()),
        ("JPMorgan Chase & Co. $JPM Rating Reiterated by Royal Bank of Canada", ("JPM",), ()),
        ("Layoffs at JPMorgan as markets slow", ("JPM",), ()),
        ("JPMorgan profit falls 20% as loan losses rise", ("JPM",), ()),
        # only the two banks are checked
        ('"Disney Analyst Downgrades Stock, Projects Parks Won\'t Reopen"', ("DIS",), ()),
    ],
)
def test_bank_that_only_gives_a_rating_is_an_analyst(linker, text, subjects, analysts):
    found = linker.link(text)
    tagged = linker.analyst_only(text)
    assert tuple(t for t in found if t not in tagged) == subjects
    assert tuple(t for t in found if t in tagged) == analysts
