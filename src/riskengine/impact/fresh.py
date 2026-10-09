"""Fresh test set for impact v2 (DECISIONS E8): other S&P 100 stocks, 2019 - 10 Jun 2020.

None of these stocks or dates is used to fit or choose a model.

    uv run python -m riskengine.impact.fresh --source data/raw/fnspid/All_external.csv
    (news slice -> prices -> headline scores; each step is skipped if its file exists)
"""

import argparse
import logging

import pandas as pd

from riskengine.impact.scores import score_headlines
from riskengine.ingest.common import finish
from riskengine.ingest.fetch_fnspid import COLUMNS, FNSPID_URL
from riskengine.ingest.prices import MARKET, VIX, fetch_prices
from riskengine.paths import DATA, RAW

log = logging.getLogger(__name__)

# S&P 100 members (2020 list) minus our 15 stocks (and their other share class / old symbol).
FRESH_SYMBOLS = (
    "ABBV ABT ACN ADBE AIG ALL AMGN AMT AXP BIIB BK BKNG BLK BMY BRK.B C CAT CHTR CL CMCSA COF "
    "COP COST CRM CSCO CVS CVX DD DHR DOW DUK EMR EXC F FDX GD GE GILD GM GS HD HON IBM INTC JNJ "
    "KHC KMI LLY LMT LOW MA MCD MDLZ MDT MET MMM MO MRK MS NEE NKE ORCL OXY PEP PFE PG PM PYPL "
    "QCOM RTX SBUX SLB SO SPG T TGT TMO TXN UNH UNP UPS USB V VZ WBA WFC WMT"
).split()
TEST_START, TEST_END = "2019-01-01", "2020-06-11"
NEWS_START = "2018-12-01"  # a few weeks before the test, for the 24 h / 72 h look-back features
PRICES_START = "2017-06-01"  # enough trading days for the market model (days -280..-31)

NEWS_PATH = RAW / "fnspid" / "fresh_news.csv.gz"
PRICES_PATH = DATA / "prices" / "prices_fresh.csv"
SCORES_PATH = DATA / "cache" / "headline_scores_fresh.csv.gz"


def fetch_news(source: str = FNSPID_URL) -> pd.DataFrame:
    keep = set(FRESH_SYMBOLS)
    chunks = pd.read_csv(source, usecols=COLUMNS, dtype=str, chunksize=500_000, on_bad_lines="skip")
    df = pd.concat(ch[ch["Stock_symbol"].isin(keep)] for ch in chunks)
    day = df["Date"].str[:10]
    return df[(day >= NEWS_START) & (day < TEST_END)]


def load_fresh_news(path=NEWS_PATH) -> pd.DataFrame:
    """Same cleaning as ingest/news.py, but each headline keeps only its FNSPID symbol."""
    raw = pd.read_csv(path, dtype=str)
    df = pd.DataFrame(
        {
            "published_at": pd.to_datetime(raw["Date"], utc=True, errors="coerce"),
            "text": raw["Article_title"].fillna("").str.strip(),
            "symbol": raw["Stock_symbol"],
        }
    )
    df = df.dropna(subset=["published_at"])
    df = df[df["text"] != ""]
    df["day"] = df["published_at"].dt.date
    df = (
        df.sort_values("published_at")
        .groupby(["day", "text"], as_index=False)
        .agg(
            published_at=("published_at", "first"),
            tickers=("symbol", lambda s: tuple(sorted(set(s)))),
        )
        .sort_values("published_at", kind="stable")
    )
    return finish(df, "news")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=FNSPID_URL, help="URL or local path to the CSV")
    args = parser.parse_args()

    if not NEWS_PATH.exists():
        df = fetch_news(args.source)
        NEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NEWS_PATH, index=False)
        log.info("kept %d headlines -> %s", len(df), NEWS_PATH)

    if not PRICES_PATH.exists():
        df = fetch_prices([*FRESH_SYMBOLS, MARKET, VIX], PRICES_START, "2020-07-01", strict=False)
        PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PRICES_PATH, index=False, float_format="%.4f")
        log.info("%d rows, %d tickers -> %s", len(df), df["ticker"].nunique(), PRICES_PATH)

    if not SCORES_PATH.exists():
        news = load_fresh_news()
        out = score_headlines(news, SCORES_PATH)
        log.info("scored %d headlines -> %s", len(out), SCORES_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "transformers", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
