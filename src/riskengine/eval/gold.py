"""Gold set: 300 real headlines labelled by hand, blind to model outputs (DECISIONS.md, E3).

uv run python -m riskengine.eval.gold      # writes data/gold/gold_to_label.csv once
"""

import pandas as pd

from riskengine.ingest.news import load_fnspid
from riskengine.paths import DATA

GOLD_DIR = DATA / "gold"
TO_LABEL = GOLD_DIR / "gold_to_label.csv"
GOLD = GOLD_DIR / "gold.csv"
START, END = "2019-01-01", "2020-06-11"
SEED, N_CREDIT, N_RANDOM = 7, 150, 150
CREDIT_WORDS = (
    r"rating|moody|fitch|downgrad|default|bankrupt|chapter 11|covenant|lender|debt|bond|"
    r"credit|loan"
)


def sample_gold() -> pd.DataFrame:
    news = load_fnspid()[0]
    news = news[(news["published_at"] >= START) & (news["published_at"] < END)]
    news = news.drop_duplicates("text")
    credit = news["text"].str.contains(CREDIT_WORDS, case=False, regex=True)
    picked = pd.concat(
        [
            news[credit].sample(N_CREDIT, random_state=SEED),
            news[~credit].sample(N_RANDOM, random_state=SEED),
        ]
    ).sample(frac=1, random_state=SEED)  # shuffle so the two halves are not recognisable
    return pd.DataFrame(
        {
            "doc_id": picked["doc_id"],
            "date": picked["published_at"].dt.date.astype(str),
            "tickers": picked["tickers"].map(" ".join),
            "headline": picked["text"],
            "label": "",
        }
    ).reset_index(drop=True)


def load_gold() -> pd.DataFrame:
    """Frozen labelled gold set: doc_id, headline, label."""
    return pd.read_csv(GOLD, dtype={"doc_id": str})


def main() -> None:
    if TO_LABEL.exists():
        raise SystemExit(f"{TO_LABEL} already exists; the sample is fixed, not re-drawn.")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df = sample_gold()
    df.to_csv(TO_LABEL, index=False)
    print(f"{len(df)} headlines -> {TO_LABEL}")


if __name__ == "__main__":
    main()
