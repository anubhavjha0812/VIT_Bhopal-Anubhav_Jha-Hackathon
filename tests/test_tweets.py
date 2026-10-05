import pandas as pd

from riskengine.ingest.tweets import load_akkerman, load_equinxx


def _write(tmp_path, columns, rows):
    path = tmp_path / "tweets.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_equinxx_cleaning_steps(tmp_path):
    path = _write(
        tmp_path,
        ["Date", "Tweet", "Stock Name", "Company Name"],
        [
            ["2022-01-03 10:00:00+00:00", "$AMZN earnings beat", "PG", "Procter & Gamble"],
            ["2022-01-03 11:00:00+00:00", "$AMZN earnings beat", "MSFT", "Microsoft"],
            ["2022-01-03 12:00:00+00:00", "$A $B $C $D $E $AAPL watchlist", "AAPL", "Apple"],
            ["2022-01-03 13:00:00+00:00", "nothing about our stocks", "TSLA", "Tesla"],
            ["2022-01-04 09:00:00+00:00", "Tesla &amp; $NVDA", "TSLA", "Tesla"],
        ],
    )
    df, stats = load_equinxx(path)

    assert stats == {"raw": 5, "after_dedupe": 4, "after_spam_filter": 3, "linked": 2}
    # linked from the text, not the (wrong) Stock Name label
    assert df["tickers"].tolist() == [("AMZN",), ("NVDA", "TSLA")]
    assert df["text"].iloc[1] == "Tesla & $NVDA"
    assert df["doc_id"].is_unique
    assert str(df["published_at"].dt.tz) == "UTC"
    assert set(df["source"]) == {"tweets"}


def test_akkerman_retweets_count_as_duplicates(tmp_path):
    path = _write(
        tmp_path,
        ["id", "created_at", "text"],
        [
            [1, "2020-04-09 10:00:00+00:00", "$JPM reserves up"],
            [2, "2020-04-09 10:05:00+00:00", "RT @someone: $JPM reserves up"],
            [3, "2020-04-09 11:00:00+00:00", "Bank of America cuts costs"],
        ],
    )
    df, stats = load_akkerman(path)

    assert stats["after_dedupe"] == 2
    assert df["tickers"].tolist() == [("JPM",), ("BAC",)]
