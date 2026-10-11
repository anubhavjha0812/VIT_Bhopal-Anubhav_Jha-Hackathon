"""E18 leak check: which test texts a model has already seen in training (docs/DECISIONS.md).

uv run python -m riskengine.eval.leaks

Exact: equal E7 normalised forms. Near: cosine of word 1-2-gram TF-IDF vectors >= 0.9.
Writes data/outputs/leak_check.json; `seen()` gives the per-row flags used when scoring.
"""

import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from riskengine.eval.datasets import HF, e18_tests, fiqa_split, phrasebank_lines
from riskengine.paths import DATA

OUT = DATA / "outputs" / "leak_check.json"
NEAR = 0.9


def norm(text: str) -> str:
    """The E7 normal form: lower case, URLs removed, non-alphanumerics -> space."""
    text = re.sub(r"https?://\S+", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tfns_train() -> list[str]:
    url = HF.format(repo="twitter-financial-news-sentiment", file="sent_train.csv")
    return pd.read_csv(url)["text"].tolist()


def corpora() -> dict[str, list[str]]:
    """Training texts per model family (E18)."""
    tfns = tfns_train()
    pb75 = phrasebank_lines("75Agree")["text"].tolist()
    pb_all = phrasebank_lines("50Agree")["text"].tolist()  # every PhraseBank sentence
    fiqa = fiqa_split("train")["text"].tolist() + fiqa_split("validation")["text"].tolist()
    return {
        "E5 models (TFNS train)": tfns,
        "E7/E10 models (TFNS + PhraseBank 75Agree + FiQA train/val)": tfns + pb75 + fiqa,
        "FinBERT baseline (PhraseBank)": pb_all,
        "E5 FinBERT (TFNS train + PhraseBank)": tfns + pb_all,
    }


def flags(test: list[str], train: list[str]) -> pd.DataFrame:
    """Per test row: exact match, and the highest TF-IDF cosine to any training text."""
    t, s = [norm(x) for x in test], [norm(x) for x in train]
    seen = set(s)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(t + s)
    a, b = vec.transform(t), vec.transform(s)
    best = np.zeros(len(t))
    for i in range(0, len(t), 1000):  # rows are L2-normalised, so the dot product is the cosine
        best[i : i + 1000] = (a[i : i + 1000] @ b.T).max(axis=1).toarray().ravel()
    exact = np.array([x in seen for x in t])
    return pd.DataFrame({"exact": exact, "cosine": best, "seen": exact | (best >= NEAR)})


def seen(tests: dict[str, pd.DataFrame] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """{corpus: {test set: boolean array of rows that were seen}}."""
    tests = tests or e18_tests()
    out = {}
    for c, train in corpora().items():
        out[c] = {n: flags(t["text"].tolist(), train)["seen"].to_numpy() for n, t in tests.items()}
    return out


def main() -> None:
    tests = e18_tests()
    out = {}
    for c, train in corpora().items():
        out[c] = {}
        for name, df in tests.items():
            f = flags(df["text"].tolist(), train)
            out[c][name] = {
                "rows": len(f),
                "exact": int(f["exact"].sum()),
                "near_only": int((f["seen"] & ~f["exact"]).sum()),
                "clean": int((~f["seen"]).sum()),
                "examples": df.loc[f["seen"] & ~f["exact"], "text"].head(3).tolist(),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for c, r in out.items():
        print(c)
        for name, v in r.items():
            print(f"  {name}: {v['rows']} rows, exact {v['exact']}, near {v['near_only']}, "
                  f"clean {v['clean']}")  # fmt: skip


if __name__ == "__main__":
    main()
