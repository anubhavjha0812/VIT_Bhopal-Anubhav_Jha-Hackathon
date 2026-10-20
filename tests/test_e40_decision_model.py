"""E40 uses invented rows and a fake SDK only; no Laya import or model run."""

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e40_decision_model as decision


def test_arms_strict_boundaries_and_filter_only():
    a = np.array([True, True, True, False, True])
    q1, q2 = np.array([0.8, 0.5, 0.8, 0.9, 0.8]), np.full(5, 0.8)
    q3 = np.array([0.2, 0.2, 0.5, 0.1, 0.7])
    arms = decision.arm_predictions(a, q1, q2, q3, 0.5)
    assert arms["B"].tolist() == [True, False, True, True, True]
    assert arms["C"].tolist() == [True, False, False, False, False]
    assert not (arms["C"] & ~arms["A"]).any()


def test_threshold_8k_only_and_deterministic_tie():
    rows = pd.DataFrame(
        {
            "dataset": ["8K"] * 4,
            "item_id": list("abcd"),
            "Q1": [0.9, 0.8, 0.3, 0.1],
            "Q2": [0.8, 0.9, 0.3, 0.2],
            "credit_true": [True, True, False, False],
        }
    )
    fitted = decision.choose_threshold(rows)
    assert fitted["threshold"] == 0.3
    assert fitted["F1"] == 1
    assert fitted["E23_rows_used"] == 0
    e23 = rows.assign(dataset="E23", credit_true=~rows.credit_true)
    with pytest.raises(ValueError, match="8K rows only"):
        decision.choose_threshold(pd.concat([rows, e23]))
    with pytest.raises(ValueError):
        decision.choose_threshold(e23)
    assert fitted == decision.choose_threshold(rows)


def test_metric_and_promotion_primary_only():
    primary = {
        "arms": {
            "A": {"TP": 39, "FP": 20, "precision": 39 / 59},
            "C": {"TP": 38, "FP": 10, "precision": 38 / 48},
        }
    }
    manifest = {"offline_after_download": True, "timing": {"noul": {"total_seconds": 10}}}
    result = decision.promotion(primary, manifest)
    assert all(result["checks"].values())
    assert "unseen-set" in result["decision"]
    assert result["secondary_choice_used_for_decision"] is False
    primary["arms"]["C"]["TP"] = 37
    assert decision.promotion(primary, manifest)["decision"] == "not promoted"
    manifest["offline_after_download"] = False
    assert not decision.promotion(primary, manifest)["checks"][
        "Kaggle_latency_reported_and_offline"
    ]


def test_run_missing_raw_does_not_lock(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    monkeypatch.setattr(decision, "LOCK", lock)
    with pytest.raises(FileNotFoundError, match="Kaggle raw"):
        decision.run(tmp_path / "missing.csv", tmp_path / "missing.json")
    assert not lock.exists()


def test_run_lock_refuses_repeat_before_loading_model_inputs(tmp_path, monkeypatch):
    raw, manifest, lock = (tmp_path / name for name in ("raw.csv", "manifest.json", "lock"))
    for path in (raw, manifest, lock):
        path.write_text("test")
    monkeypatch.setattr(decision, "LOCK", lock)
    monkeypatch.setattr(decision, "RESULT", tmp_path / "result.json")
    with pytest.raises(FileExistsError, match="already scored or locked"):
        decision.run(raw, manifest)


def test_script_fixed_questions_and_probability_semantics():
    script = decision.kaggle_contract()
    primary = script.make_questions("Example Corp", "noul")
    secondary = script.make_questions("Example Corp", "choice")
    assert "Example Corp" in primary["Q2"]["instructions"]
    assert "<TICKER NAME>" not in primary["Q2"]["instructions"]
    assert primary["Q1"]["instructions"] == secondary["Q1"]["instructions"]
    assert secondary["Q2"]["criteria"] == {"A": "yes", "B": "no"}
    assert script.extract_probabilities({"type": "noul", "noul": 0.2}, "noul") == (0.2, 0.8)
    assert script.extract_probabilities(
        {"type": "choice", "probabilities": {"A": 0.7, "B": 0.3}}, "choice"
    ) == (0.7, 0.3)
    with pytest.raises(ValueError):
        script.extract_probabilities({"type": "noul", "noul": 2}, "noul")
    assert script.LAYA_VERSION == "0.4.11"
    assert script.MODEL_ID == "convaiinnovations/laya" and script.SUBFOLDER == "typed-decisions"


def test_script_matches_frozen_protocol_questions():
    script = decision.kaggle_contract()
    protocol = (decision.ROOT / "docs/experiments/DECISION_MODEL_BENCH.md").read_text()
    for question in script.QUESTIONS.values():
        assert f'"{question}"' in protocol


def test_script_refuses_missing_target_before_model_load(tmp_path):
    script = decision.kaggle_contract()
    eight, e23, mapping = (tmp_path / name for name in ("eight.csv", "e23.csv", "targets.csv"))
    eight.write_text("item_id,text\na,Invented headline\n")
    e23.write_text("headline_id,headline,target_company\nb,Invented headline,EX\n")
    mapping.write_text("item_id,target_company\na,\n")
    with pytest.raises(ValueError, match="Confirm missing 8-K targets"):
        script.load_inputs(eight, e23, mapping)


def test_laptop_guard():
    if not Path("/kaggle/working").exists():
        with pytest.raises(RuntimeError, match="Kaggle GPU only"):
            decision.kaggle_contract().require_kaggle()


def test_offline_network_guard_blocks_outbound():
    script = decision.kaggle_contract()
    original = socket.socket.connect
    with script.offline_network():
        with pytest.raises(RuntimeError, match="must run offline"):
            socket.socket.connect(SimpleNamespace(family=socket.AF_INET), ("127.0.0.1", 9))
    assert socket.socket.connect is original


def test_validate_raw_binds_text_targets_and_all_forms():
    script = decision.kaggle_contract()
    inputs = [{"dataset": "8K", "item_id": "a", "text": "Invented news", "target_company": "EX"}]
    raw = pd.DataFrame(
        [
            {
                "dataset": "8K",
                "item_id": "a",
                "target_company": "EX",
                "text_sha256": script.text_digest("Invented news"),
                "form": f,
                "question": q,
                "p_yes": 0.8,
                "p_no": 0.2,
                "call_seconds": 0.1,
            }
            for f in script.FORMS
            for q in script.QUESTIONS
        ]
    )
    decision.validate_raw(raw, inputs, script)
    with pytest.raises(ValueError, match="every fixed ID"):
        decision.validate_raw(raw.iloc[:-1], inputs, script)
    with pytest.raises(ValueError, match="text or target"):
        decision.validate_raw(raw.assign(target_company="WRONG"), inputs, script)


def test_form_preserves_pair_order_and_removed_texts():
    rows = pd.DataFrame(
        {
            "headline_id": ["b", "a", "c"],
            "headline": ["B news", "A news", "C news"],
            "target_company": ["B", "A", "C"],
            "reference_fire": [True, False, True],
        }
    )
    table = pd.DataFrame(
        {
            "dataset": ["E23"] * 3,
            "item_id": ["a", "c", "b"],
            "Q1": [0.1, 0.9, 0.9],
            "Q2": [0.1, 0.9, 0.9],
            "Q3": [0.1] * 3,
        }
    )
    result = decision.evaluate_form(table, rows, np.array([True, True, False]), 0.5)
    assert result["arms"]["A"]["TP"] == 1
    assert result["arms"]["C"]["TP"] == 1
    assert result["C_false_removed"] == 1
    assert [r["headline"] for r in result["C_removed_alerts"]] == ["A news"]


def test_manifest_rejects_wrong_checkpoint(tmp_path):
    script = decision.kaggle_contract()
    raw = tmp_path / "raw.csv"
    raw.write_text("fake")
    manifest = {"package_version": script.LAYA_VERSION, "model_id": "wrong/model"}
    with pytest.raises(ValueError, match="model_id"):
        decision.validate_manifest(manifest, raw, tmp_path / "manifest.json", script, {})


def test_result_writer_does_not_overwrite(tmp_path):
    result = tmp_path / "result.json"
    decision.write_json(result, {"test": "fixture"})
    with pytest.raises(FileExistsError):
        decision.write_json(result, {})
    assert json.loads(result.read_text()) == {"test": "fixture"}


def test_one_run_end_to_end_with_fake_probabilities(tmp_path, monkeypatch):
    """Exercise joins, artifact hashes, one threshold and primary-only promotion."""
    script = decision.kaggle_contract()
    eight = pd.DataFrame(
        {
            "item_id": [f"k{i}" for i in range(287)],
            "text": [f"Invented issuer K{i} event" for i in range(287)],
            "final": ["Default"] * 147 + ["Other"] * 140,
        }
    )
    e23 = pd.DataFrame(
        {
            "headline_id": [f"e{i}" for i in range(243)],
            "headline": [f"Invented issuer E{i} event" for i in range(243)],
            "target_company": [f"E{i}" for i in range(243)],
            "rule_class": ["Credit downgrade"] * 243,
            "predicted_fire": [i < 59 for i in range(243)],
            "reference_fire": [i < 39 or i >= 236 for i in range(243)],
        }
    )
    paths = {name: tmp_path / name for name in ("eight.csv", "e23.csv", "targets.csv")}
    eight.to_csv(paths["eight.csv"], index=False)
    e23.to_csv(paths["e23.csv"], index=False)
    eight[["item_id"]].assign(target_company=[f"K{i}" for i in range(287)]).to_csv(
        paths["targets.csv"], index=False
    )
    inputs = script.load_inputs(paths["eight.csv"], paths["e23.csv"], paths["targets.csv"])
    raw_rows = []
    for row in inputs:
        i = int(row["item_id"][1:])
        yes = (i < 147) if row["dataset"] == "8K" else not 39 <= i < 49
        for form in script.FORMS:
            for question in script.QUESTIONS:
                probability = 0.9 if yes and form == "noul" and question != "Q3" else 0.1
                raw_rows.append(
                    {
                        "dataset": row["dataset"],
                        "item_id": row["item_id"],
                        "target_company": row["target_company"],
                        "text_sha256": script.text_digest(row["text"]),
                        "form": form,
                        "question": question,
                        "p_yes": probability,
                        "p_no": 1 - probability,
                        "call_seconds": 0.1,
                    }
                )
    raw_path, manifest_path = tmp_path / "raw.csv", tmp_path / "raw_manifest.json"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    manifest = {
        "package_version": script.LAYA_VERSION,
        "model_id": script.MODEL_ID,
        "subfolder": script.SUBFOLDER,
        "questions_sha256": script.question_digest(),
        "script_sha256": decision.sha256(decision.KAGGLE_SCRIPT),
        "raw_sha256": decision.sha256(raw_path),
        "inputs_sha256": {
            "8K": decision.sha256(paths["eight.csv"]),
            "E23": decision.sha256(paths["e23.csv"]),
            "targets": decision.sha256(paths["targets.csv"]),
        },
        "checkpoint_files_sha256": {
            "model.safetensors": "0" * 64,
            "rl_agent_config.json": "0" * 64,
        },
        "download_utc": "fixture-only",
        "device": "fake GPU; no model loaded",
        "offline_after_download": True,
        "timing": {"noul": {"total_seconds": 53.0}},
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(decision, "EIGHT_K", paths["eight.csv"])
    monkeypatch.setattr(decision, "E23", paths["e23.csv"])
    monkeypatch.setattr(decision, "TARGETS", paths["targets.csv"])
    monkeypatch.setattr(decision, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(decision, "LOCK", tmp_path / "run.lock")
    fit_calls = []
    choose = decision.choose_threshold

    def fit_spy(rows):
        fit_calls.append(rows["dataset"].unique().tolist())
        return choose(rows)

    monkeypatch.setattr(decision, "choose_threshold", fit_spy)
    result = decision.run(raw_path, manifest_path)
    assert fit_calls == [["8K"]]
    assert result["threshold"]["threshold"] == 0.1
    assert result["results"]["noul"]["arms"]["C"]["TP"] == 39
    assert result["results"]["noul"]["C_false_removed"] == 10
    assert result["results"]["choice"]["arms"]["C"]["alerts"] == 0
    assert "unseen-set" in result["decision"]["decision"]
    with pytest.raises(FileExistsError):
        decision.run(raw_path, manifest_path)
    assert len(fit_calls) == 1
