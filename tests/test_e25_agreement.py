"""E25 agreement report: teacher vs reference, teacher vs teacher, and the speed rule."""

import json

import pandas as pd
import pytest

from riskengine.eval import e25_agreement as ag
from riskengine.eval.e24_gate import CLASSES


def probs(ids, labels):
    rows = [{f"p_{c}": (0.9 if c == lab else 0.1 / 11) for c in CLASSES} for lab in labels]
    return pd.DataFrame(rows, index=pd.Index(ids, name="id")).reset_index()


@pytest.fixture
def folder(tmp_path):
    ids = ["a", "b", "c", "d"]
    ref = pd.DataFrame({"item_id": ids, "label": ["Other", "Other", "Analyst rating", "M&A"],
                        "ambiguous": "N", "rationale": "x"})  # fmt: skip
    ref.to_csv(tmp_path / "ref.csv", index=False)
    probs(ids, ["Other", "Other", "Analyst rating", "Other"]).to_csv(
        tmp_path / "d1_qwen.csv.gz", index=False
    )
    probs(ids, ["Other", "Product news", "Analyst rating", "M&A"]).to_csv(
        tmp_path / "d1_mistral.csv.gz", index=False
    )
    return tmp_path


def test_top_class_reads_the_probability_columns():
    df = probs(["a", "b"], ["Earnings", "Default"]).set_index("id")
    assert ag.top_class(df).tolist() == ["Earnings", "Default"]


def test_report_scores_teachers_and_the_agreed_rows(folder):
    ref = pd.read_csv(folder / "ref.csv").set_index("item_id")["label"]
    qwen = ag.top_class(ag.load_teacher(folder / "d1_qwen.csv.gz"))
    mistral = ag.top_class(ag.load_teacher(folder / "d1_mistral.csv.gz"))
    q = ag.score_vs_reference(qwen, ref)
    assert q["accuracy"] == 0.75 and q["rows"] == 4
    pair = ag.pair_report(qwen, mistral, ref)
    assert pair["agree_rate"] == 0.5 and pair["kept_rows"] == 2  # a and c agree
    assert pair["accuracy_on_kept"] == 1.0
    assert pair["accuracy_on_dropped"] == 0.5  # b and d: qwen right on b only


def test_supported_macro_is_above_fixed_macro_when_classes_are_empty(folder):
    ref = pd.read_csv(folder / "ref.csv").set_index("item_id")["label"]
    qwen = ag.top_class(ag.load_teacher(folder / "d1_qwen.csv.gz"))
    q = ag.score_vs_reference(qwen, ref)
    assert q["macro_f1_supported"] > q["macro_f1_fixed_12"]


def test_load_teacher_rejects_missing_class_columns(tmp_path):
    pd.DataFrame({"id": ["a"], "p_Other": [1.0]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing probability columns"):
        ag.load_teacher(tmp_path / "bad.csv")


def test_speed_rule_keeps_a_fast_second_teacher_and_replaces_a_slow_one():
    fast = {"mistral_nemo": {"projected_pool_hours": 3.0, "rows_per_second": 3.7}}
    slow = {"mistral_nemo": {"projected_pool_hours": 9.5, "rows_per_second": 1.2}}
    assert ag.speed_decision(fast)["decision"] == "keep"
    assert "backup" in ag.speed_decision(slow)["decision"]
    assert "no meta" in ag.speed_decision({})["decision"]


def test_dry_run_report_needs_no_reference(folder):
    (folder / "dry_mistral_nemo_meta.json").write_text(json.dumps({"projected_pool_hours": 2.0}))
    probs(["p1", "p2"], ["Other", "Earnings"]).to_csv(
        folder / "dry_mistral_nemo.csv.gz", index=False
    )
    out = ag.build(folder, "dry")
    assert out["second_teacher_speed"]["decision"] == "keep"
    assert out["teachers"]["mistral_nemo"]["predicted_counts"] == {"Other": 1, "Earnings": 1}
