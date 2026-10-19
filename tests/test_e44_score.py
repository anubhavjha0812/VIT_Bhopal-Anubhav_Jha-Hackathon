"""Invented E44 rows only: no project headline, annotation, prediction or model is loaded."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from riskengine.eval import e24_inference, e25_candidate
from riskengine.eval import e44_score as score
from riskengine.eval.e24_gate import CLASSES

NAMES = CLASSES[3:]


def fake_probabilities(texts):
    p = np.full((len(texts), len(NAMES)), 0.01)
    for i, text in enumerate(texts):
        label = "Product news" if "phone" in text else "Other"
        p[i, NAMES.index(label)] = 0.92
    return NAMES, p


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    work = tmp_path / "annotations"
    work.mkdir()
    ids = [f"invented_{i:02}" for i in range(16)]
    labels = ["Product news"] * 4 + ["Earnings"] * 4 + ["Other"] * 8
    sample = pd.DataFrame(
        {"id": ids, "text": ["Acme phone commentary"] * 8 + ["Acme general commentary"] * 8}
    )
    sample.to_csv(work / "sample.csv", index=False)
    pd.DataFrame({"id": ids, "label": labels}).to_csv(work / "labels_final.csv", index=False)
    pd.DataFrame(
        {
            "id": ids,
            "date": [f"2019-01-{17 - i:02}" for i in range(16)],
            "hint": "FORBIDDEN",
            "source": "FORBIDDEN",
        }
    ).to_csv(work / "provenance.csv", index=False)
    for name in ["labels_chatgpt.csv", "labels_claude.csv", "dropped_disagreements.csv"]:
        (work / name).write_text("invented,not parsed\n")
    (tmp_path / "guide.md").write_text("invented guide")
    (tmp_path / "protocol.md").write_text("invented protocol")
    support = {c: labels.count(c) for c in CLASSES}
    record = {
        "kept_rows": 16,
        "support": support,
        "sha256": {
            name: score.sha256(work / name)
            for name in [
                "labels_final.csv",
                "sample.csv",
                "labels_chatgpt.csv",
                "labels_claude.csv",
                "dropped_disagreements.csv",
            ]
        },
    }
    record["sha256"].update(
        guide=score.sha256(tmp_path / "guide.md"), protocol=score.sha256(tmp_path / "protocol.md")
    )
    score.write_json(work / "labels_final_record.json", record)
    score.write_json(
        work / "sampling.json",
        {
            "files_sha256": {
                name: score.sha256(work / name) for name in ["sample.csv", "provenance.csv"]
            }
        },
    )
    model = tmp_path / "models/topic"
    model.mkdir(parents=True)
    (model / "model.onnx").write_text("fake weights, never loaded")
    outputs = tmp_path / "data/outputs"
    outputs.mkdir(parents=True)
    frozen = {
        "candidate": "C",
        "threshold": None,
        "temperature": None,
        "pinned_sha256": {"models/topic/model.onnx": score.sha256(model / "model.onnx")},
        "inference": {"assets_sha256": {"model.onnx": score.sha256(model / "model.onnx")}},
    }
    score.write_json(outputs / "e24_frozen.json", frozen)
    spec = {
        "rows": 16,
        "classes": CLASSES,
        "eligible_classes": CLASSES,
        "secondary_classes": CLASSES[:-1],
        "support": support,
        "candidates": ["A", "C", "C-noPF", "E25-v3"],
        "student_folder": "fake_student",
        "work_folder": "annotations",
        "guide": "guide.md",
        "protocol": "protocol.md",
        "frozen": "data/outputs/e24_frozen.json",
        "labels_sha256": score.sha256(work / "labels_final.csv"),
        "expected_sha256": {
            str(p.relative_to(tmp_path)): score.sha256(p)
            for p in [work / "labels_final_record.json", work / "sampling.json"]
        },
        "runtime": {},
        "bootstrap": {"draws": 40, "seed": 44, "block_length": 4, "percentiles": [2.5, 97.5]},
        "decision": {
            "challengers": ["C-noPF", "E25-v3"],
            "minimum_primary_gain": 0.02,
            "paired_lower_bound_strictly_above": 0,
        },
    }
    plan = tmp_path / score.PLAN
    plan.parent.mkdir(parents=True)
    plan.write_text("# Invented plan\n```json\n" + json.dumps(spec) + "\n```\n")
    (tmp_path / score.PLAN_LOCK).write_text(score.sha256(plan) + " 2026-10-19T00:00:00+00:00\n")
    monkeypatch.setattr(score, "PLAN_SHA256", score.sha256(plan))
    monkeypatch.setattr(score, "LABEL_SHA256", spec["labels_sha256"])
    return tmp_path, spec, frozen


@pytest.mark.parametrize("target", ["plan", "lock"])
def test_plan_hash_refuses_changes(sealed, target):
    root, spec, _ = sealed
    assert score.load_plan(root) == spec
    path = root / (score.PLAN if target == "plan" else score.PLAN_LOCK)
    path.write_text(path.read_text() + "changed")
    with pytest.raises(ValueError, match="mismatch"):
        score.load_plan(root)


@pytest.mark.parametrize(
    "name",
    [
        "labels_final.csv",
        "labels_chatgpt.csv",
        "labels_claude.csv",
        "dropped_disagreements.csv",
        "labels_final_record.json",
        "sampling.json",
        "provenance.csv",
        "model.onnx",
    ],
)
def test_every_frozen_input_hash_refuses_tampering(sealed, name):
    root, spec, _ = sealed
    score.verify_inputs(root, spec)
    path = (
        (root / "models/topic/model.onnx")
        if name == "model.onnx"
        else (root / "annotations" / name)
    )
    path.write_text(path.read_text() + "changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        score.verify_inputs(root, spec)


def test_integrity_check_never_parses_labels(sealed, monkeypatch):
    root, spec, _ = sealed
    monkeypatch.setattr(pd, "read_csv", lambda *_a, **_kw: pytest.fail("CSV parse forbidden"))
    score.verify_inputs(root, spec)


def test_exact_runtime_is_required(sealed, monkeypatch):
    root, spec, _ = sealed
    spec["runtime"] = {"invented_package": "1.0"}
    monkeypatch.setattr(score.importlib.metadata, "version", lambda _: "2.0")
    with pytest.raises(ValueError, match="runtime version mismatch"):
        score.verify_inputs(root, spec, runtime=True)


def test_id_join_is_by_id_and_refuses_missing_extra_duplicate():
    frame = pd.DataFrame({"id": ["b", "a"], "label": ["Other", "Earnings"]})
    assert score.align_ids(frame, ["a", "b"]).label.tolist() == ["Earnings", "Other"]
    for invalid in [["a"], ["a", "c"], ["a", "a"]]:
        with pytest.raises(ValueError, match="ID join"):
            score.align_ids(frame, invalid)
    with pytest.raises(ValueError, match="duplicate IDs"):
        score.align_ids(pd.concat([frame, frame.iloc[:1]]), ["a", "b"])


def test_inference_reads_only_frozen_ids_and_provenance_only_dates(sealed, monkeypatch):
    root, spec, _ = sealed
    real_read = pd.read_csv
    opened = []

    def spy(path, *args, **kwargs):
        opened.append((Path(path).name, kwargs.get("usecols")))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    frame = score.prediction_texts(root, spec)
    order = score.publication_order(root, spec, frame.id.tolist())
    assert frame.id.tolist() == list(reversed(order))
    assert ("labels_final.csv", ["id"]) in opened
    assert ("provenance.csv", ["id", "date"]) in opened
    assert not any(name.startswith("labels_c") for name, _ in opened)


def test_publication_order_rejects_unparseable_dates(sealed):
    root, spec, _ = sealed
    path = root / "annotations/provenance.csv"
    path.write_text("id,date\ninvented_00,not-a-date\n")
    with pytest.raises(ValueError):
        score.publication_order(root, spec, ["invented_00"])


def test_metric_math_matches_sklearn_including_absent_classes():
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 2, 1, 2, 3, 3])
    classes = ["a", "b", "c", "Other", "absent"]
    v = score.metric_vector(y, pred, classes, list(range(5)), [0, 1, 2, 4])
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(5)), zero_division=0)
    assert v[:3] == pytest.approx(
        [
            f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0),
            f1_score(y, pred, labels=[0, 1, 2, 4], average="macro", zero_division=0),
            accuracy_score(y, pred),
        ]
    )
    assert v[3:] == pytest.approx(np.r_[pr, rc, f1])


def test_bootstrap_indices_fixed_contiguous_paired_and_truncated():
    n, length, draws = 8, 3, 5
    expected_starts = np.random.default_rng(44).integers(0, 6, size=(5, 3))
    expected = np.array(
        [
            np.concatenate([np.arange(start, start + length) for start in row])[:n]
            for row in expected_starts
        ]
    )
    actual = score.bootstrap_indices(n, draws, 44, length)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, score.bootstrap_indices(n, draws, 44, length))
    assert not np.array_equal(actual, score.bootstrap_indices(n, draws, 45, length))
    assert actual.shape == (draws, n) and actual.min() >= 0 and actual.max() < n
    assert math.ceil(math.sqrt(822)) == 29


def test_paired_intervals_fixed_classes_and_a_cannot_be_selected(sealed):
    root, spec, _ = sealed
    labels = pd.read_csv(root / "annotations/labels_final.csv")
    predictions = labels[["id"]].copy()
    for candidate in spec["candidates"]:
        predictions[candidate] = labels.label if candidate in ["A", "C-noPF"] else "Other"
    result = score.score_predictions(labels, predictions, labels.id.tolist(), spec)
    assert result["scores"]["C"]["macro_f1_primary"]["difference_vs_C_ci95"] == [0.0, 0.0]
    assert result["scores"]["C-noPF"]["macro_f1_primary"]["difference_vs_C"] > 0.02
    assert result["decision"]["C-noPF"]["qualifies_for_owner_review"] is True
    assert result["decision"]["A"]["qualifies_for_owner_review"] is False
    assert result["selected_model"] is None and result["engine_changed"] is False
    assert result["primary_classes"] == CLASSES
    assert result["scores"]["C"]["per_class"]["Default"]["support"] == 0
    assert result["scores"]["C"]["per_class"]["Default"]["f1"]["ci95"] == [0.0, 0.0]
    again = score.score_predictions(labels, predictions.iloc[::-1], labels.id.tolist(), spec)
    assert result == again


def test_no_product_fallback_preserves_every_other_route_and_cached_probabilities():
    texts = [
        "Moody's downgrades Acme to junk",
        "Acme posts quarterly revenue of $3 billion",
        "Acme unveils a phone",
        "Acme phone commentary",
        "Acme general commentary",
        "Capital Fund buys Acme, Beta, Gamma",
        "Broker upgrades Acme to buy",
        "What to expect before Acme earnings",
    ]
    calls = []

    def backend(items):
        calls.append(list(items))
        return fake_probabilities(items)

    outputs = score.predict_candidates(
        texts, backend, None, {"candidate": "C", "temperature": None, "threshold": None}
    )
    c, routes = outputs["C"]
    ablated, ablated_routes = outputs["C-noPF"]
    assert len(calls) == 1
    assert "model_product_to_other" in routes
    for i, route in enumerate(routes):
        if route == "model_product_to_other":
            assert (c[i], ablated[i], ablated_routes[i]) == ("Other", "Product news", "model")
        else:
            assert (ablated[i], ablated_routes[i]) == (c[i], routes[i])
    assert {
        "credit_rule",
        "earnings_rule",
        "product_rule",
        "model",
        "holdings_rule",
        "analyst_rule",
        "earnings_preview_rule",
    } <= set(routes)


def test_ablation_refuses_unknown_fallback_semantics():
    with pytest.raises(ValueError, match="fallback semantics"):
        score.without_product_fallback(
            ["Other"], ["model_product_to_other"], ["Other"], np.ones((1, 1))
        )


@pytest.mark.parametrize("name", list(score.OUTPUTS))
def test_repeat_prediction_refused_even_for_partial_run(tmp_path, name):
    (tmp_path / name).write_text("partial or previous run")
    with pytest.raises(FileExistsError, match="repeat prediction refused"):
        score.claim_prediction(tmp_path)


def test_fixture_only_prediction_and_scoring_transaction(sealed, monkeypatch):
    root, spec, frozen = sealed

    class FakeBackend:
        inference_metadata = frozen["inference"]

        def __init__(self, *_args):
            pass

        def __call__(self, texts):
            return fake_probabilities(texts)

    monkeypatch.setattr(e24_inference, "EvaluationProbs", FakeBackend)
    monkeypatch.setattr(e24_inference, "inference_metadata", lambda *_: frozen["inference"])
    monkeypatch.setattr(e25_candidate, "student_probs", FakeBackend)
    assert score.run_predict(root)["rows"] == 16
    with pytest.raises(FileExistsError, match="repeat prediction refused"):
        score.run_predict(root)
    assert score.run_score(root)["phase"] == "scored"
    state = score.read_json(root / "data/outputs/e44.run_lock")
    assert state["phase"] == "scored"
    assert state["results_sha256"] == score.sha256(root / "data/outputs/e44_results.json")
    assert state["labels_sha256"] == spec["labels_sha256"]
    with pytest.raises(FileExistsError, match="scoring already claimed"):
        score.run_score(root)


def test_notebook_is_unexecuted_and_cells_compile():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/e44_score/e44_score.ipynb"
    data = json.loads(notebook.read_text())
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(notebook), "exec")


def test_bundle_contains_only_explicit_sealed_inputs_and_new_code(sealed):
    root, spec, _ = sealed
    script = Path(__file__).resolve().parents[1] / "notebooks/e44_score/prepare_bundle.py"
    module_spec = importlib.util.spec_from_file_location("e44_bundle", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    (root / "pyproject.toml").write_text("invented metadata")
    src = root / "src/riskengine"
    src.mkdir(parents=True)
    (src / "invented.py").write_text("# invented code")
    secret = root / "private_notes/secret.txt"
    secret.parent.mkdir()
    secret.write_text("not an upload")
    paths = module.bundle_paths(root, spec)
    assert secret not in paths and (src / "invented.py") in paths
    assert (root / "annotations/labels_chatgpt.csv") in paths
    assert (root / "annotations/provenance.csv") in paths
    assert all(p.is_relative_to(root) for p in paths)
    assert hashlib.sha256(b"fake weights, never loaded").hexdigest() == score.sha256(
        root / "models/topic/model.onnx"
    )
