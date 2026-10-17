"""R3 re-test rule: E replaces C only with a gain of 0.02 and no big class dropping 0.05."""

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from riskengine.eval import e25_retest_score as retest
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


@pytest.mark.parametrize("alternate", [False, True])
def test_student_cli_routes_only_e_and_preserves_v2_outputs(
    tmp_path, monkeypatch, capsys, alternate
):
    sample = tmp_path / "r3"
    sample.mkdir()
    ids = [f"fixture{i:03d}" for i in range(400)]
    texts = [f"Fake news fixture number {i}" for i in range(400)]
    pd.DataFrame({"item_id": ids, "text": texts}).to_csv(sample / "to_label.csv", index=False)
    pd.DataFrame({"item_id": ids, "label": ["Other"] * 400}).to_csv(
        sample / "R3_labels_claude.csv", index=False
    )
    monkeypatch.setattr(retest, "SET", sample)
    monkeypatch.setattr(retest, "DATA", tmp_path)
    base = object()
    student = SimpleNamespace(inference_metadata={"method": "fixture-only"})
    monkeypatch.setattr(retest, "engine_probs", lambda: base)
    loaded, routed = [], {}

    def load_student(folder):
        loaded.append(folder)
        return student

    def build(cfg, probs):
        routed[cfg.candidate] = (cfg, probs)

        def predict(items):
            assert items == texts
            return ["Other"] * len(items), ["fixture"] * len(items)

        return predict

    monkeypatch.setattr(retest, "student_probs", load_student)
    monkeypatch.setattr(retest, "build_predictor", build)
    monkeypatch.setattr(retest, "bootstrap_gain", lambda *args: [0.0, 0.0])
    output = tmp_path / "outputs"
    output.mkdir()
    v2_files = [
        output / f"e25_retest_{name}.{ext}"
        for name, ext in (
            ("scores", "json"),
            ("per_class", "csv"),
            ("predictions", "csv"),
        )
    ]
    for path in v2_files:
        path.write_text("existing v2 fixture\n")
    folder = tmp_path / "local-student"
    argv = ["retest"]
    suffix = ""
    if alternate:
        suffix = "_v3"
        argv += ["--student", str(folder), "--output-suffix", suffix]
    monkeypatch.setattr(sys, "argv", argv)
    retest.main()
    assert loaded == ([folder] if alternate else [])
    assert routed["E"][1] is (student if alternate else None)
    assert all(routed[k][1] is base for k in ("A", "C", "D"))
    assert routed["D"][0].threshold == 0.5
    assert all(routed[k][0].temperature is None for k in routed)
    report = json.loads((output / f"e25_retest_scores{suffix}.json").read_text())
    assert report["rows"] == 400 and not report["rule"]["e_replaces_c"]
    assert (output / f"e25_retest_per_class{suffix}.csv").exists()
    preds = pd.read_csv(output / f"e25_retest_predictions{suffix}.csv")
    assert preds["item_id"].tolist() == ids
    assert {"A", "C", "D", "E"}.issubset(preds.columns)
    if alternate:
        assert all(path.read_text() == "existing v2 fixture\n" for path in v2_files)
        assert report["student"] == str(folder)
        assert report["student_inference"] == student.inference_metadata
    else:
        assert "student" not in report
    capsys.readouterr()


@pytest.mark.parametrize("suffix", ["_v3", ""])
def test_student_comparison_refuses_existing_outputs_before_inference(
    tmp_path, monkeypatch, suffix
):
    monkeypatch.setattr(retest, "DATA", tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    path = output / f"e25_retest_scores{suffix}.json"
    path.write_text("preserve\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retest",
            "--student",
            str(tmp_path / "student"),
            "--output-suffix",
            suffix,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        retest.main()
    assert path.read_text() == "preserve\n"


def test_comparison_suffix_cannot_escape_output_folder(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["retest", "--output-suffix", "../../elsewhere"])
    with pytest.raises(SystemExit, match="2"):
        retest.main()
