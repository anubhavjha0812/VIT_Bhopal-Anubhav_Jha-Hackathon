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


def test_name_needs_word_boundary(linker):
    assert linker.link("pineapple prices rise") == ()


def test_unknown_cashtag_ignored(linker):
    assert linker.link("$SPY $QQQ rally") == ()


def test_names_can_be_turned_off(linker):
    assert linker.link("Tesla deliveries beat", use_names=False) == ()
