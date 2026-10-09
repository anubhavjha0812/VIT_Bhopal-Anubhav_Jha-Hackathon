"""Daily prices for the universe + S&P 500 index from yfinance, stored small in the repo.

uv run python -m riskengine.ingest.prices            # prices_daily.csv (replay window)
uv run python -m riskengine.ingest.prices --history  # prices_history.csv (impact model)
"""

import argparse
from pathlib import Path

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.paths import DATA

PRICES_PATH = DATA / "prices" / "prices_daily.csv"
HISTORY_PATH = DATA / "prices" / "prices_history.csv"
MARKET = "^GSPC"
VIX = "^VIX"
START, END = "2019-01-01", "2021-01-01"  # pre-window history for baselines and betas
HISTORY_START = "2008-01-01"  # one year before the first headline, for market-model betas


def fetch_prices(
    tickers: list[str], start: str = START, end: str = END, strict: bool = True
) -> pd.DataFrame:
    """Long table: date, ticker, close (split/dividend adjusted), volume.

    strict: raise if any ticker has no prices; otherwise just leave it out.
    """
    import yfinance as yf

    wide = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )
    long = wide[["Close", "Volume"]].stack(future_stack=True).reset_index()
    long.columns = ["date", "ticker", "close", "volume"]
    long = long.dropna(subset=["close"])
    missing = set(tickers) - set(long["ticker"])
    if missing and strict:
        raise RuntimeError(f"no prices downloaded for {sorted(missing)}")
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="2008-2020 incl. VIX")
    args = parser.parse_args()

    tickers = [*Linker.from_csv().tickers, MARKET]
    if args.history:
        tickers, start, path = [*tickers, VIX], HISTORY_START, HISTORY_PATH
    else:
        start, path = START, PRICES_PATH
    df = fetch_prices(tickers, start, END)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"{len(df)} rows, {df['ticker'].nunique()} tickers -> {path}")


if __name__ == "__main__":
    main()
