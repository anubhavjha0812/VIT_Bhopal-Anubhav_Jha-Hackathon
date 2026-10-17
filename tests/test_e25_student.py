"""E25 student training script: label order, soft labels, class weights input, metrics.
Only the pure helpers are tested; no model is loaded and nothing is downloaded."""

import importlib.util

import numpy as np
import pandas as pd

from riskengine.nlp.events import LEARNED_CLASSES
from riskengine.paths import ROOT

SCRIPT = ROOT / "notebooks" / "e25_student" / "train_student.py"
spec = importlib.util.spec_from_file_location("train_student", SCRIPT)
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def test_class_order_matches_the_engine_and_the_tfn_table_covers_them():
    assert ts.CLASSES == LEARNED_CLASSES
    assert set(ts.TOPIC_TO_EVENT.values()) == set(ts.CLASSES)
    assert len(ts.TOPIC_NAMES) == 20 and set(ts.TOPIC_NAMES) == set(ts.TOPIC_TO_EVENT)


def test_tfn_table_is_the_engine_table():
    from riskengine.nlp.events import TOPIC_TO_EVENT

    assert ts.TOPIC_TO_EVENT == TOPIC_TO_EVENT


def test_soft_matrix_uses_teacher_probabilities_and_one_hot_for_hard_rows():
    cols = [f"p_{c}" for c in ts.CLASSES]
    soft = np.full(len(cols), 0.1 / 8)
    soft[ts.CLASSES.index("M&A")] = 0.9
    df = pd.DataFrame([[*soft, "M&A"], [*[np.nan] * len(cols), "Macroeconomic"]],
                      columns=[*cols, "label"])  # fmt: skip
    p = ts.soft_matrix(df)
    assert np.allclose(p[0], soft) and p[1, ts.CLASSES.index("Macroeconomic")] == 1.0
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)


def test_macro_f1_counts_only_classes_present_in_the_validation_rows():
    y, p = np.array([0, 0, 1]), np.array([0, 0, 1])
    assert ts.macro(y, p) == 1.0  # classes 2..8 have no rows and do not count as zero


def test_per_class_reports_support_precision_and_recall():
    out = ts.per_class(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert out["M&A"] == {"support": 2, "predicted": 1, "precision": 1.0, "recall": 0.5}
    assert out["Geopolitical"]["recall"] == 1.0
    assert out["Other"]["precision"] is None and out["Other"]["recall"] is None
