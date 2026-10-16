"""E24 development selection (K3): D1/D2 separation, the choice rule, the threshold grid, and
the freeze refusal once final annotation exists. A fake 9-class model stands in for the engine."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from riskengine.eval import e24_dev
from riskengine.eval.e24_dev import Config, build_predictor, fit_temperature, run_selection, select
from riskengine.eval.e24_gate import GRID, GateError, Paths, section_sha

NAMES = ["Geopolitical", "Macroeconomic", "M&A", "Product news", "Earnings", "Analyst rating",
         "Legal/Regulatory", "Management change", "Other"]  # fmt: skip


def fake_probs(texts):
    """Product news for 'phone', Earnings for 'profit', otherwise Other; 'unsure' lowers the
    top probability to 0.32 and 'hesitant' to 0.68."""
    p = np.full((len(texts), len(NAMES)), 0.01)
    for i, t in enumerate(texts):
        k = NAMES.index(
            "Product news" if "phone" in t else "Earnings" if "profit" in t else "Other"
        )
        top = 0.32 if "unsure" in t else 0.68 if "hesitant" in t else 0.92
        p[i] = (1 - top) / (len(NAMES) - 1)
        p[i, k] = top
    return NAMES, p


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    gold, outputs, backup = tmp_path / "gold", tmp_path / "outputs", tmp_path / "backup"
    for d in (gold, outputs, backup):
        d.mkdir()
    rows = {
        # D1
        "a1": ("Acme unveils a phone", "Product news"),
        "a2": ("Acme posts record profit", "Earnings"),
        "a3": ("Acme chief speaks at a dinner", "Other"),
        # D2
        "b1": ("Beta launches a phone", "Product news"),
        "b2": ("Beta phone sales talk at a conference", "Other"),
        "b3": ("Beta quarterly profit rises", "Earnings"),
        "b4": ("Beta unsure markets drift", "Other"),
        "b5": ("Beta hesitant markets drift", "Other"),
    }
    e3 = pd.DataFrame([(k, v[0]) for k, v in rows.items()], columns=["doc_id", "headline"])
    e3.to_csv(tmp_path / "e3.csv", index=False)
    monkeypatch.setattr(e24_dev, "E3_FILE", tmp_path / "e3.csv")
    split = pd.DataFrame({"doc_id": list(rows), "split": ["D1"] * 3 + ["D2"] * 5})
    split.to_csv(gold / "e3_split.csv", index=False)
    pd.DataFrame({"item_id": ["N000"], "text": ["x"], "date": "d", "companies": "c"}).to_csv(
        gold / "final_news_to_label.csv", index=False
    )
    pd.DataFrame({"item_id": ["T000"], "text": ["y"], "date": "d", "companies": "c"}).to_csv(
        gold / "final_tweets_to_label.csv", index=False
    )
    (gold / "GUIDE.md").write_text("guide\n")
    decisions = tmp_path / "DECISIONS.md"
    decisions.write_text("## E24. x\n- y\n## E4. z\n")
    files = {n: sha(gold / n) for n in ("final_news_to_label.csv", "final_tweets_to_label.csv",
                                        "e3_split.csv")}  # fmt: skip
    (outputs / "e24_final_sample.json").write_text(json.dumps({"files_sha256": files}))
    for n in files:
        (backup / n).write_bytes((gold / n).read_bytes())
    (backup / "MANIFEST.sha256").write_text("".join(f"{v}  {k}\n" for k, v in files.items()))
    p = Paths(gold=gold, outputs=outputs, backup=backup, decisions=decisions, root=tmp_path)
    p.section_sha = section_sha(decisions)

    def write(split_name, which):
        lab = pd.DataFrame({"item_id": which, "label": [rows[i][1] for i in which]})
        lab = lab.assign(ambiguous="N", rationale="r", label_source="ai_agreed")
        path = tmp_path / f"e3_{split_name}_labels.csv"
        lab.to_csv(path, index=False)
        return path

    d1 = write("D1", ["a1", "a2", "a3"])
    d2 = write("D2", ["b1", "b2", "b3", "b4", "b5"])
    return p, d1, d2, rows


# --- predictors ---


def test_predictor_order_and_sources():
    texts = [
        "Moody's downgrades Acme to junk",
        "Acme quarterly profit rises",
        "Acme unveils a phone",
        "Acme phone sales talk",
        "Acme unsure drift",
    ]
    labels, src = build_predictor(Config("D", 0.40), fake_probs)(texts)
    assert labels == ["Credit downgrade", "Earnings", "Product news", "Other", "Other"]
    assert src == ["credit_rule", "earnings_rule", "product_rule", "model_product_to_other",
                   "abstain"]  # fmt: skip
    a_labels, a_src = build_predictor(Config("A"), fake_probs)(texts)
    assert a_labels[3] == "Product news" and set(a_src) <= {"credit_rule", "model"}


@pytest.mark.parametrize(
    "cfg", [Config("D", 0.42), Config("D"), Config("A", 0.5), Config("E")]
)  # fmt: skip
def test_threshold_grid_is_enforced(cfg):
    with pytest.raises(GateError):
        build_predictor(cfg, fake_probs)


# --- temperature: D1 only, only with enough support ---


def test_no_temperature_when_d1_support_is_low(env):
    p, d1, *_ = env
    lab = pd.read_csv(d1, dtype=str)
    out = fit_temperature(lab, ["x"] * len(lab), fake_probs)
    assert out["temperature"] is None and out["classes_with_10_rows"] == 0


def test_temperature_fits_when_five_classes_have_ten_rows():
    classes = ["Other", "Earnings", "Product news", "M&A", "Macroeconomic"]
    lab = pd.DataFrame({"label": [c for c in classes for _ in range(10)],
                        "label_source": "ai_agreed"})  # fmt: skip
    texts = ["phone" if c == "Product news" else "profit" if c == "Earnings" else "x"
             for c in lab["label"]]  # fmt: skip
    out = fit_temperature(lab, texts, fake_probs)
    assert out["temperature"] is not None and out["temperature"] > 0


# --- selection on D2 only ---


def test_selection_is_deterministic_and_uses_d2(env):
    p, d1, d2, rows = env
    texts = [rows[i][0] for i in ["b1", "b2", "b3", "b4", "b5"]]
    lab = pd.read_csv(d2, dtype=str)
    a = select(lab, texts, fake_probs, None)
    b = select(lab, texts, fake_probs, None)
    assert a == b
    assert a["d2_rows_scored"] == 5
    assert a["chosen"] in "ABCD"


def test_changing_d1_labels_does_not_change_the_choice(env, tmp_path):
    p, d1, d2, _ = env
    first = run_selection(p, d1, d2, fake_probs)["selection"]
    # a fresh environment with different D1 labels (temperature stays off: low support)
    for f in (p.frozen, p.backup / p.frozen.name):
        f.chmod(0o644)
        f.unlink()
    lab = pd.read_csv(d1, dtype=str)
    lab["label"] = "Other"
    lab.to_csv(d1, index=False)
    second = run_selection(p, d1, d2, fake_probs)["selection"]
    assert first == second


def test_threshold_choice_ties_go_to_the_lowest_and_boundary_is_flagged(env):
    p, d1, d2, rows = env
    texts = [rows[i][0] for i in ["b1", "b2", "b3", "b4", "b5"]]
    s = select(pd.read_csv(d2, dtype=str), texts, fake_probs, None)
    scores = s["threshold_grid"]
    best = max(scores.values())
    assert s["threshold_for_D"] == min(float(t) for t, v in scores.items() if v == best)
    assert s["threshold_at_grid_boundary"] == (s["threshold_for_D"] in (GRID[0], GRID[-1]))


def test_degradation_rule_blocks_a_candidate():
    """A class with ≥ 10 D2 rows that loses > 0.05 F1 against A makes a candidate ineligible."""
    lab = pd.DataFrame({"item_id": [f"x{i}" for i in range(20)],
                        "label": ["Product news"] * 10 + ["Other"] * 10,
                        "label_source": "ai_agreed"})  # fmt: skip
    texts = ["Acme phone news"] * 10 + ["Acme drift"] * 10  # no product verb: B maps to Other
    s = select(lab, texts, fake_probs, None)
    assert s["allowed"]["A"] and not s["allowed"]["B"] and s["chosen"] == "A"


# --- separation and freeze refusal ---


def test_d1_and_d2_files_cannot_be_swapped(env):
    p, d1, d2, _ = env
    with pytest.raises(GateError, match="do not match"):
        run_selection(p, d2, d1, fake_probs)


def test_selection_never_opens_final_label_files(env, monkeypatch):
    p, d1, d2, _ = env
    opened = []
    real = pd.read_csv

    def spy(path, *a, **k):
        opened.append(str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", spy)
    run_selection(p, d1, d2, fake_probs)
    assert not [f for f in opened if "final" in f and "labels" in f]
    assert not [f for f in opened if "/raw/" in f]


@pytest.mark.parametrize("name", ["final_news_labels.csv", "raw/chatgpt_final_news.csv",
                                  "records/gemini_final_tweets.json"])  # fmt: skip
def test_freeze_refused_when_final_annotation_exists(env, name):
    p, d1, d2, _ = env
    f = p.gold / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x\n")
    with pytest.raises(GateError, match="freeze must come first"):
        run_selection(p, d1, d2, fake_probs)
    assert not p.frozen.exists()


def test_selection_freezes_once_with_its_rules_and_time(env):
    p, d1, d2, _ = env
    rep = run_selection(p, d1, d2, fake_probs)
    frozen = json.loads(p.frozen.read_text())
    assert frozen["candidate"] == rep["selection"]["chosen"]
    assert frozen["frozen_at_utc"].endswith("Z")
    if frozen["candidate"] != "A":
        assert frozen["rules_sha256"] == sha(e24_dev.RULES_FILE)
    with pytest.raises(GateError, match="already frozen"):
        run_selection(p, d1, d2, fake_probs)


def test_freeze_pins_inference_and_refuses_runtime_method_change(env, monkeypatch):
    from riskengine.eval import e24_inference
    from riskengine.eval.e24_gate import check_candidate, freeze_candidate

    p, *_ = env
    model = p.root / "models/topic/model.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture")
    method = {"method": "per_row_test", "batch_size": 1, "onnxruntime_version": "test"}
    monkeypatch.setattr(e24_inference, "inference_metadata", lambda _: method)
    frozen = freeze_candidate(p, "A", None, None, None)
    assert frozen["inference"] == method
    assert check_candidate(p)["inference"] == method
    monkeypatch.setattr(e24_inference, "inference_metadata", lambda _: method | {"batch_size": 64})
    with pytest.raises(GateError, match="method/settings mismatch"):
        check_candidate(p)
