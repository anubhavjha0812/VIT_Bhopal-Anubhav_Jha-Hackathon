"""Daily prices for the universe + S&P 500 index from yfinance, stored small in the repo.

uv run python -m riskengine.ingest.prices        # refresh data/prices/prices_daily.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.paths import DATA

PRICES_PATH = DATA / "prices" / "prices_daily.csv"
MARKET = "^GSPC"
START, END = "2019-01-01", "2021-01-01"  # pre-window history for baselines and betas


def fetch_prices(tickers: list[str], start: str = START, end: str = END) -> pd.DataFrame:
    """Long table: date, ticker, close (split/dividend adjusted), volume."""
    import yfinance as yf

    wide = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )
    long = wide[["Close", "Volume"]].stack(future_stack=True).reset_index()
    long.columns = ["date", "ticker", "close", "volume"]
    long = long.dropna(subset=["close"])
    missing = set(tickers) - set(long["ticker"])
    if missing:
        raise RuntimeError(f"no prices downloaded for {sorted(missing)}")
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()

    tickers = [*Linker.from_csv().tickers, MARKET]
    df = fetch_prices(tickers, args.start, args.end)
    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRICES_PATH, index=False, float_format="%.4f")
    print(f"{len(df)} rows, {df['ticker'].nunique()} tickers -> {PRICES_PATH}")


if __name__ == "__main__":
    main()
