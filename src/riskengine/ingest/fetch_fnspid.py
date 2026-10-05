"""Rebuild our news slice from the public FNSPID file (Hugging Face, CC BY-NC 4.0).

Streams the 5.7 GB CSV (or a local copy) and keeps only headlines whose Stock_symbol is in
our universe. Only titles are kept, not article bodies.

    uv run python -m riskengine.ingest.fetch_fnspid [--source PATH_OR_URL]
"""

import argparse

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.paths import RAW

FNSPID_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"
)
OUT_PATH = RAW / "fnspid" / "universe_news.csv.gz"
COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]


def fetch(source: str = FNSPID_URL, linker: Linker | None = None) -> pd.DataFrame:
    linker = linker or Linker.from_csv()
    symbols = set(linker.cashtag_to_ticker)
    chunks = pd.read_csv(source, usecols=COLUMNS, dtype=str, chunksize=500_000, on_bad_lines="skip")
    return pd.concat(ch[ch["Stock_symbol"].isin(symbols)] for ch in chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=FNSPID_URL, help="URL or local path to the CSV")
    args = parser.parse_args()

    df = fetch(args.source)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"kept {len(df)} headlines -> {OUT_PATH}")


if __name__ == "__main__":
    main()
