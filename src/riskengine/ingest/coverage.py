"""Per-stock data coverage inside the replay window: tweets, news and price days.

uv run python -m riskengine.ingest.coverage     # prints and writes docs/data_coverage.md
"""

import pandas as pd

from riskengine.config import Config, load_config
from riskengine.ingest.news import load_fnspid
from riskengine.ingest.prices import load_prices
from riskengine.ingest.tweets import load_akkerman
from riskengine.paths import ROOT

OUT_PATH = ROOT / "docs" / "data_coverage.md"


def in_window(df: pd.DataFrame, config: Config, col: str = "published_at") -> pd.DataFrame:
    start = pd.Timestamp(config.window_start, tz="UTC")
    end = pd.Timestamp(config.window_end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df[col] >= start) & (df[col] < end)]


def per_stock(docs: pd.DataFrame) -> pd.DataFrame:
    rows = docs.explode("tickers")
    return rows.groupby("tickers").agg(
        rows=("doc_id", "size"), days=("published_at", lambda s: s.dt.date.nunique())
    )


def coverage(config: Config) -> pd.DataFrame:
    tweets = in_window(load_akkerman()[0], config)
    news = in_window(load_fnspid()[0], config)
    prices = load_prices()
    prices["published_at"] = pd.to_datetime(prices["date"]).dt.tz_localize("UTC")
    prices = in_window(prices, config)

    t, n = per_stock(tweets), per_stock(news)
    table = pd.DataFrame(
        {
            "tweets": t["rows"],
            "tweet_days": t["days"],
            "news": n["rows"],
            "news_days": n["days"],
            "price_days": prices.groupby("ticker").size(),
        }
    )
    table = table.drop(index="^GSPC", errors="ignore").fillna(0).astype(int)
    return table.sort_values("news")


def main() -> None:
    config = load_config()
    table = coverage(config)
    trading_days = int(table["price_days"].max())
    header = (
        f"# Data coverage\n\nReplay window {config.window_start} to {config.window_end} "
        f"({trading_days} trading days). Counts after cleaning and company linking.\n\n"
    )
    text = header + table.to_markdown() + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
