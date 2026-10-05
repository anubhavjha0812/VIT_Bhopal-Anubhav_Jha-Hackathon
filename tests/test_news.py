import pandas as pd

from riskengine.ingest.fetch_fnspid import fetch
from riskengine.ingest.news import load_fnspid

ROWS = [
    ["2020-05-01 12:00:00 UTC", "Facebook ad sales slow", "FB", "u1", "p"],
    ["2020-05-01 13:00:00 UTC", "Banks rally: JPMorgan, Bank of America up", "JPM", "u2", "p"],
    ["2020-05-01 13:00:00 UTC", "Banks rally: JPMorgan, Bank of America up", "BAC", "u3", "p"],
    ["2020-05-02 09:00:00 UTC", "Clorox restocks shelves", "CLX", "u4", "p"],
    ["2020-05-02 10:00:00 UTC", "", "AAPL", "u5", "p"],
]
COLS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]


def _write(tmp_path, rows):
    path = tmp_path / "news.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    return path


def test_fetch_keeps_only_universe_symbols(tmp_path):
    df = fetch(str(_write(tmp_path, ROWS)))
    assert sorted(df["Stock_symbol"]) == ["AAPL", "BAC", "FB", "JPM"]


def test_load_maps_symbols_and_merges_syndicated_headlines(tmp_path):
    df, stats = load_fnspid(_write(tmp_path, ROWS))

    assert stats == {"raw": 5, "after_dedupe": 2}
    assert df["tickers"].tolist() == [("META",), ("BAC", "JPM")]
    assert set(df["source"]) == {"news"}
    assert str(df["published_at"].dt.tz) == "UTC"
