"""Rebuild our news slice from the public FNSPID file (Hugging Face, CC BY-NC 4.0).

Streams the 5.7 GB CSV (or a local copy) and keeps only headlines whose Stock_symbol is in
our universe. With --by-name it also keeps headlines filed under another symbol that name one of
our companies (DECISIONS E17). Only titles are kept, not article bodies.

    uv run python -m riskengine.ingest.fetch_fnspid [--source PATH_OR_URL] [--by-name]
"""

import argparse

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.paths import RAW

FNSPID_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"
)
OUT_PATH = RAW / "fnspid" / "universe_news.csv.gz"
NAMED_PATH = RAW / "fnspid" / "universe_news_named.csv.gz"
COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]


def fetch(
    source: str = FNSPID_URL, linker: Linker | None = None, by_name: bool = False
) -> pd.DataFrame:
    linker = linker or Linker.from_csv()
    symbols = set(linker.cashtag_to_ticker)
    chunks = pd.read_csv(source, usecols=COLUMNS, dtype=str, chunksize=500_000, on_bad_lines="skip")

    def keep(ch: pd.DataFrame) -> pd.Series:
        ours = ch["Stock_symbol"].isin(symbols)
        if not by_name:
            return ours
        return ours | ch["Article_title"].fillna("").map(lambda t: bool(linker.link(t)))

    return pd.concat(ch[keep(ch)] for ch in chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=FNSPID_URL, help="URL or local path to the CSV")
    parser.add_argument("--by-name", action="store_true", help="also keep headlines naming us")
    args = parser.parse_args()

    df = fetch(args.source, by_name=args.by_name)
    out = NAMED_PATH if args.by_name else OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"kept {len(df)} headlines -> {out}")


if __name__ == "__main__":
    main()
