from riskengine.ingest.linking import Linker
from riskengine.ingest.prices import MARKET, PRICES_PATH, load_prices


def test_committed_prices_cover_universe_and_market():
    df = load_prices()
    assert set(df["ticker"]) == {*Linker.from_csv().tickers, MARKET}
    assert list(df.columns) == ["date", "ticker", "close", "volume"]
    assert (df["close"] > 0).all()
    assert not df.duplicated(["ticker", "date"]).any()


def test_prices_file_is_small_enough_to_commit():
    assert PRICES_PATH.stat().st_size < 2_000_000
