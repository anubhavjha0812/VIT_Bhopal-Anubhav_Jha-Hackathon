"""E27 post-hoc US sanity check: own headline sentiment vs entity sentiment on FiQA test (150).

uv run python -m riskengine.eval.e27_fiqa_check

Not a registered test and not used to choose anything. FiQA gives one target company and a
continuous gold score; we compare Spearman rank correlation and sign agreement. One run.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskengine.nlp.sentiment import EntitySentiment, get_scorer
from riskengine.paths import DATA


def metrics(gold: np.ndarray, score: np.ndarray) -> dict:
    nz = gold != 0
    return {
        "spearman": round(float(spearmanr(gold, score).statistic), 4),
        "sign_accuracy": round(float((np.sign(score[nz]) == np.sign(gold[nz])).mean()), 4),
    }


def main() -> None:
    df = pd.read_csv(DATA / "eval" / "fiqa_test.csv")
    texts, targets = df["sentence"].tolist(), df["target"].astype(str).tolist()
    gold = df["sentiment_score"].to_numpy()
    own = np.array(get_scorer("own").score(texts))
    ent = np.array(EntitySentiment().score_entities(texts, targets))
    out = {
        "rows": len(df),
        "own": metrics(gold, own),
        "entity": metrics(gold, ent),
        "note": "Post-hoc check, not registered. Target ticker/name from FiQA used as entity.",
    }
    (DATA / "outputs" / "e27_fiqa_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
