"""E25 student data: agreement filter, credit drop, Other cut, validation split, soft labels."""

import numpy as np
import pandas as pd

from riskengine.eval.e24_gate import CLASSES
from riskengine.eval.e25_student_data import (
    PCOLS,
    agreed_rows,
    cut_other,
    split_valid,
)


def teacher(top: list[str]) -> pd.DataFrame:
    """One-hot-ish probabilities (0.9 on the chosen class) for ids r0, r1, ..."""
    p = np.full((len(top), len(CLASSES)), 0.1 / (len(CLASSES) - 1))
    for i, c in enumerate(top):
        p[i, CLASSES.index(c)] = 0.9
    return pd.DataFrame(
        p, columns=[f"p_{c}" for c in CLASSES], index=[f"r{i}" for i in range(len(top))]
    )


def test_agreed_rows_keep_only_agreement_and_drop_credit_classes():
    q = teacher(["M&A", "Other", "Default", "Earnings"])
    m = teacher(["M&A", "Earnings", "Default", "Earnings"])
    texts = pd.Series(["a", "b", "c", "d"], index=q.index)
    out = agreed_rows(q, m, texts, "pool")
    assert out["id"].tolist() == ["r0", "r3"]  # r1 disagrees, r2 is a credit class
    assert out["label"].tolist() == ["M&A", "Earnings"]
    assert out["source"].tolist() == ["pool", "pool"]
    assert np.allclose(out[PCOLS].sum(axis=1), 1, atol=1e-3)  # soft label over 9 classes


def frame(labels):
    return pd.DataFrame({"id": [f"r{i:04d}" for i in range(len(labels))], "label": labels})


def test_cut_other_keeps_cap_and_every_other_class_and_is_deterministic():
    df = frame(["Other"] * 50 + ["M&A"] * 5)
    a, b = cut_other(df, cap=10, seed=1), cut_other(df, cap=10, seed=1)
    assert a.equals(b)
    assert (a["label"] == "Other").sum() == 10 and (a["label"] == "M&A").sum() == 5


def test_split_valid_is_stratified_disjoint_and_leaves_tiny_classes_in_train():
    df = frame(["Other"] * 100 + ["M&A"] * 40 + ["Geopolitical"] * 1)
    train, valid = split_valid(df, frac=0.05, seed=3)
    assert not set(train["id"]) & set(valid["id"])
    assert len(train) + len(valid) == len(df)
    assert (valid["label"] == "Other").sum() == 5 and (valid["label"] == "M&A").sum() == 2
    assert "Geopolitical" not in set(valid["label"])  # one row: stays in train


def test_v3_adds_hard_rows_without_duplicates_or_overlap():
    import pandas as pd

    from riskengine.eval.e25_student_data_v3 import add_rows

    cols = ["id", "text", "label", "source"]
    train = pd.DataFrame([["a", "Oil falls", "Macroeconomic", "pool"]], columns=cols)
    valid = pd.DataFrame([["b", "War starts", "Geopolitical", "pool"]], columns=cols)
    extra = pd.DataFrame(
        [["c", f"Title {i}", "Geopolitical", "official_geo"] for i in range(40)]
        + [["d", "oil falls ", "Macroeconomic", "resolved_ai"]],
        columns=cols,
    )
    t, v, added = add_rows(train, valid, extra)
    assert added == 40 and len(t) + len(v) == 42  # the seen text is dropped
    assert not set(t["text"]) & set(v["text"])
