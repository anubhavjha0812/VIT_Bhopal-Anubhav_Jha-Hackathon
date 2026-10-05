import pandas as pd

from riskengine.ingest.tweets import load_tweets


def _write(tmp_path, rows):
    path = tmp_path / "tweets.csv"
    pd.DataFrame(rows, columns=["Date", "Tweet", "Stock Name", "Company Name"]).to_csv(
        path, index=False
    )
    return path


def test_cleaning_steps(tmp_path):
    path = _write(
        tmp_path,
        [
            ["2022-01-03 10:00:00+00:00", "$AMZN earnings beat", "PG", "Procter & Gamble"],
            ["2022-01-03 11:00:00+00:00", "$AMZN earnings beat", "MSFT", "Microsoft"],
            ["2022-01-03 12:00:00+00:00", "$A $B $C $D $E $AAPL watchlist", "AAPL", "Apple"],
            ["2022-01-03 13:00:00+00:00", "nothing about our stocks", "TSLA", "Tesla"],
            ["2022-01-04 09:00:00+00:00", "Tesla &amp; $NVDA", "TSLA", "Tesla"],
        ],
    )
    df, stats = load_tweets(path)

    assert stats == {"raw": 5, "after_dedupe": 4, "after_spam_filter": 3, "linked": 2}
    # linked from the text, not the (wrong) Stock Name label
    assert df["tickers"].tolist() == [("AMZN",), ("NVDA", "TSLA")]
    assert df["text"].iloc[1] == "Tesla & $NVDA"
    assert df["doc_id"].is_unique
    assert str(df["published_at"].dt.tz) == "UTC"
