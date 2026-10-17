"""R3 re-test rule: E replaces C only with a gain of 0.02 and no big class dropping 0.05."""

import pandas as pd

from riskengine.eval.e25_retest_score import decide, fixed12, per_class


def pc(f1_by_class: dict, support: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"class": list(f1_by_class), "f1": list(f1_by_class.values()),
                         "support": support})  # fmt: skip


def test_decide_needs_the_gain_and_no_big_class_drop():
    c = pc({"M&A": 0.5, "Other": 0.8})
    assert decide(0.40, 0.43, c, pc({"M&A": 0.6, "Other": 0.8}))["e_replaces_c"]
    assert not decide(0.40, 0.41, c, pc({"M&A": 0.6, "Other": 0.8}))["e_replaces_c"]  # gain < 0.02
    assert not decide(0.40, 0.45, c, pc({"M&A": 0.6, "Other": 0.7}))["e_replaces_c"]  # Other -0.10


def test_decide_ignores_classes_with_fewer_than_ten_rows():
    c = pc({"M&A": 0.5}, support=5)
    assert decide(0.40, 0.43, c, pc({"M&A": 0.0}, support=5))["e_replaces_c"]


def test_fixed12_counts_missing_classes_as_zero_and_per_class_has_all_classes():
    assert fixed12(["Other", "Other"], ["Other", "Other"]) < 0.1  # 11 empty classes score 0
    assert len(per_class(["Other"], ["Other"])) == 12
