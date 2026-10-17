"""Candidate E reuses C's rules with a different probability backend. Synthetic texts only."""

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e25_candidate
from riskengine.eval.e24_dev import ORDER, Config, build_predictor
from riskengine.eval.e24_gate import GateError
from riskengine.nlp.events import LEARNED_CLASSES


def model_probabilities(texts):
    names = LEARNED_CLASSES
    p = np.zeros((len(texts), len(names)))
    p[:, names.index("M&A")] = 1.0
    return names, p


def test_E_is_exactly_C_with_supplied_student_probabilities():
    texts = [
        "Moody's cuts Acme credit rating",
        "Wealth Management Buys Fund A, Fund B",
        "Broker upgrades Acme to Buy",
        "What's in store for Acme earnings?",
        "Acme reports quarterly results",
        "Acme launches a phone",
        "Acme to buy Beta",
    ]
    e, sources = build_predictor(Config("E"), model_probabilities)(texts)
    assert (e, sources) == build_predictor(Config("C"), model_probabilities)(texts)
    assert e == [
        "Credit downgrade",
        "Other",
        "Analyst rating",
        "Other",
        "Earnings",
        "Product news",
        "M&A",
    ]
    assert build_predictor(Config("E"), model_probabilities)([]) == ([], [])


def test_E_default_uses_student_factory_and_leaves_selection_order_unchanged(monkeypatch):
    monkeypatch.setattr(e25_candidate, "student_probs", lambda: model_probabilities)
    assert build_predictor(Config("E"))(["Acme to buy Beta"])[0] == ["M&A"]
    assert ORDER == ["A", "B", "C", "D"]


def test_student_label_schema_is_enforced(monkeypatch):
    class InvalidBackend:
        labels = ["Default", "Other"]

        def __init__(self, _):
            pass

    monkeypatch.setattr(e25_candidate, "EvaluationProbs", InvalidBackend)
    with pytest.raises(GateError, match="nine learned"):
        e25_candidate.student_probs()


def test_read_d1_loads_only_two_D1_references(tmp_path, monkeypatch):
    monkeypatch.setattr(e25_candidate, "D1_ROWS", 2)
    raw = pd.DataFrame({"item_id": ["a", "b"], "text": ["one", "two"]})
    input_path = tmp_path / "D1_to_label.csv"
    raw.to_csv(input_path, index=False)
    for version in ("v6", "v7"):
        folder = tmp_path / f"reference_{version}/development"
        folder.mkdir(parents=True)
        pd.DataFrame(
            {"item_id": ["b", "a"], "label": ["Other", "M&A"], "ambiguous": ["N", "Y"]}
        ).to_csv(folder / "D1_labels.csv", index=False)
    frame, refs = e25_candidate.read_d1(input_path, tmp_path)
    assert frame.reference_v6.tolist() == frame.reference_v7.tolist() == ["M&A", "Other"]
    assert set(refs) == {"v6", "v7"}
    with pytest.raises(GateError, match="D1 input only"):
        e25_candidate.read_d1(tmp_path / "D2_to_label.csv", tmp_path)


def test_primary_macro_f1_uses_all_twelve_classes():
    score = e25_candidate.metrics(["Other", "Other"], ["Other", "Other"])
    assert score["macro_f1_12_primary"] == pytest.approx(1 / 12)
    assert score["macro_f1_supported_supplementary"] == 1
    assert score["accuracy"] == 1
