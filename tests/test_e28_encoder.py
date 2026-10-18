import hashlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e28_encoder_dev as dev
from riskengine.nlp.events import LEARNED_CLASSES
from riskengine.paths import ROOT


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"notebooks/e28_encoder/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temperature_and_loss_preserve_explicit_class_order():
    module = script("train_encoder")
    p = np.full((2, 9), 0.0125)
    p[0, 0], p[1, 8] = 0.9, 0.9
    assert module.nll([module.CLASSES[0], module.CLASSES[8]], p) < 0.11
    hot = module.temperature_scale(p, 2)
    assert np.allclose(hot.sum(1), 1) and np.array_equal(hot.argmax(1), p.argmax(1))
    assert hot.max() < p.max()
    temperature = module.fit_temperature(p, [module.CLASSES[0], module.CLASSES[8]])
    assert temperature > 0
    head = SimpleNamespace(classes_=module.CLASSES[::-1], predict_proba=lambda x: p[:, ::-1])
    assert np.array_equal(module.ordered_probabilities(head, None), p)


def test_setfit_sampling_is_fixed_capped_and_does_not_use_valid():
    module = script("train_setfit")
    frame = pd.DataFrame(
        [
            {"text": f"{label} {i}", "label": label}
            for j, label in enumerate(module.CLASSES)
            for i in range(5 if j == 0 else 90)
        ]
    )
    sampled = module.sample_train(frame)
    assert sampled["label"].value_counts().max() == 64
    assert sampled["label"].eq(module.CLASSES[0]).sum() == 5
    pd.testing.assert_frame_equal(sampled, module.sample_train(frame))
    assert module.SEED == 28 and module.MAX_LENGTH == 64


def test_probability_export_round_trip_reorders_columns_and_checks_texts(tmp_path):
    module = script("train_encoder")
    frame = pd.DataFrame(
        {"id": ["a", "b"], "text": ["invented first", "invented second"], "set": ["D1", "D2"]}
    )
    p = np.full((2, 9), 1 / 9)
    module.save_probabilities(tmp_path, frame, p, {"variant": "G", "encoder": "fixture"})
    path = tmp_path / "probabilities.npz"
    loaded, _ = dev.load_export(path, frame)
    assert np.allclose(loaded, p)
    frame.loc[0, "text"] = "different"
    with pytest.raises(ValueError, match="text/order mismatch"):
        dev.load_export(path, frame)


def test_permuted_export_class_order_is_remapped_not_assumed(tmp_path):
    frame = pd.DataFrame({"id": ["a"], "text": ["fictional headline"], "set": ["D1"]})
    names = list(LEARNED_CLASSES)[::-1]
    p = np.array([[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    path = tmp_path / "p.npz"
    np.savez(
        path,
        ids=["a"],
        sets=["D1"],
        labels=names,
        text_sha256=[hashlib.sha256(frame["text"][0].encode()).hexdigest()],
        probabilities=p,
    )
    meta = {
        "class_order": names,
        "npz_sha256": dev.digest(path),
        "calibrated": True,
        "calibration_split": "valid",
    }
    path.with_suffix(".json").write_text(json.dumps(meta))
    loaded, _ = dev.load_export(path, frame)
    assert np.array_equal(loaded, p[:, ::-1])
    meta["calibration_split"] = "D2"
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="valid only"):
        dev.load_export(path, frame)


def scored(f1, per_class=0.6):
    return {"macro_f1": f1, "per_class_f1": {"Other": per_class}}


def test_d2_margin_class_guard_and_simpler_tie():
    others = {"A": scored(0.4), "C": scored(0.5), "E": scored(0.49)}
    support = {"Other": 20}
    assert dev.d2_decision(others, {"G": scored(0.519)}, support)["chosen_development_only"] == "C"
    assert dev.d2_decision(others, {"G": scored(0.52)}, support)["chosen_development_only"] == "G"
    assert not dev.d2_decision(others, {"G": scored(0.6, 0.5)}, support)["new_candidates_admitted"]
    assert (
        dev.d2_decision(others, {"G_small": scored(0.53), "G_large": scored(0.535)}, support)[
            "chosen_development_only"
        ]
        == "G_small"
    )


def test_candidate_cache_keeps_credit_and_c_rules():
    texts = ["ExampleCo files for Chapter 11 bankruptcy", "Company unveils a new phone"]
    p = np.zeros((2, 9))
    p[:, list(LEARNED_CLASSES).index("Other")] = 1
    from riskengine.eval.e25_candidate import build_candidate

    pred, _ = build_candidate(dev.cached_probs(texts, p))(texts)
    assert pred == ["Default", "Product news"]


def test_e28_rejects_final_paths_before_reading(tmp_path):
    with pytest.raises(ValueError, match="final files are forbidden"):
        dev.evaluate(tmp_path / "final_news.csv", [], out=tmp_path / "report.json")


def test_training_input_rejects_final_sets_without_loading_an_encoder(tmp_path):
    module = script("train_encoder")
    for name in ("train", "valid"):
        pd.DataFrame(
            {
                "id": [str(i) for i in range(9)],
                "text": [f"invented {i}" for i in range(9)],
                "label": module.CLASSES,
            }
        ).to_csv(tmp_path / f"{name}.csv", index=False)
    pd.DataFrame({"id": ["x"], "text": ["fixture"], "set": ["final_news"]}).to_csv(
        tmp_path / "eval_texts.csv", index=False
    )
    module.find = lambda name: tmp_path / name
    with pytest.raises(ValueError, match="finals are forbidden"):
        module.inputs()


def test_end_to_end_report_uses_fake_probabilities_and_no_real_references(tmp_path, monkeypatch):
    module = script("train_encoder")
    frame = pd.DataFrame(
        {
            "id": ["d1", "d2", "r3"],
            "text": [
                "Invented quiet item one",
                "Invented quiet item two",
                "Invented quiet item three",
            ],
            "set": ["D1", "D2", "R3"],
        }
    )
    texts = tmp_path / "eval_texts.csv"
    frame.to_csv(texts, index=False)
    p = np.zeros((3, 9))
    p[:, module.CLASSES.index("Other")] = 1
    folder = tmp_path / "G"
    module.save_probabilities(
        folder, frame, p, {"variant": "G", "encoder": "BAAI/bge-small-en-v1.5"}
    )

    def fake_probs(items):
        matrix = np.zeros((len(items), 9))
        matrix[:, list(LEARNED_CLASSES).index("Other")] = 1
        return list(LEARNED_CLASSES), matrix

    monkeypatch.setattr(dev, "engine_probs", lambda: fake_probs)
    monkeypatch.setattr(dev, "student_probs", lambda: fake_probs)
    reads = []

    def references(part, split, root):
        reads.append(split)
        return ["Other"] * len(part), "fixture-reference"

    monkeypatch.setattr(dev, "read_reference", references)
    monkeypatch.setattr(dev, "bootstrap", lambda *args: {"gain": 0, "ci95": [0, 0]})
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"selection": {"threshold_for_D": 0.5}, "temperature": {"temperature": None}})
    )
    out = tmp_path / "report.json"
    result = dev.evaluate(texts, [folder / "probabilities.npz"], selection_path=selection, out=out)
    assert reads == ["D1", "D2", "R3"] and out.is_file()
    assert result["d2_rule"]["fresh_test_required"]
    assert not result["d2_rule"]["production_changed"]
    assert set(result["d2_existing_comparators"]) == set("ABCDE")
    assert result["sets"]["D2"]["scores"]["G_bge-small-en-v1.5"][
        "macro_f1_12_primary"
    ] == pytest.approx(1 / 12)
    with pytest.raises(FileExistsError):
        dev.evaluate(texts, [], out=out)
