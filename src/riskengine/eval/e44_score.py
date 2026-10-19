"""Locked E44b prediction/scoring for offline Kaggle; never trains or selects a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.e24_dev import Config, build_predictor
from riskengine.paths import ROOT

PLAN = "docs/others/E44_SCORING_PLAN.md"
PLAN_LOCK = "data/outputs/e44_scoring_plan.lock"
PLAN_SHA256 = "8d6f1406db597935ec8387ff7ffb6bc328917bf9031a9497cd8d9e2f74f42193"
LABEL_SHA256 = "722986b2547f7274be4b29b9f9b5cb090942adee08cf179390feae7542d1177d"
OUTPUTS = ("e44_predictions.csv", "e44_results.json", "e44.run_lock")
LIMITS = [
    "AI agreement only; no independent human validation.",
    "78 disagreements were dropped rather than settled, selecting easier agreed rows.",
    "Default has only 28 rows; low-support estimates are unstable.",
    "Enriched sampling is not population prevalence; old news and source mix limit transfer.",
    "Different headlines may describe one event; temporal blocks do not ensure independence.",
    "Project overlap checks do not certify unknown upstream pretraining exposure.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.exists() and relative.startswith("docs/"):
        path = (root / "docs/others" / relative.removeprefix("docs/")).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"not a repository-relative input: {relative}")
    return path


def check_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"hash mismatch or missing file: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_plan(root: Path) -> dict:
    parts = (root / PLAN_LOCK).read_text().strip().split()
    if len(parts) != 2 or parts[0] != PLAN_SHA256:
        raise ValueError("scoring plan lock mismatch")
    date = datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
    if date.utcoffset() is None or date.utcoffset().total_seconds() != 0:
        raise ValueError("plan lock must have a UTC time")
    check_hash(root / PLAN, parts[0])
    blocks = re.findall(r"```json\s*\n(.*?)\n```", (root / PLAN).read_text(), re.S)
    if len(blocks) != 1:
        raise ValueError("expected one executable scoring specification")
    return json.loads(blocks[0])


def verify_inputs(root: Path, spec: dict, *, runtime: bool = False) -> dict:
    """Hash annotation files as bytes; do not parse any label or hidden sampling hint."""
    for relative, expected in spec["expected_sha256"].items():
        check_hash(local(root, relative), expected)
    work = local(root, spec["work_folder"])
    record = read_json(work / "labels_final_record.json")
    if record["sha256"]["labels_final.csv"] != LABEL_SHA256:
        raise ValueError("final-label hash differs from the registered value")
    special = {"guide": spec["guide"], "protocol": spec["protocol"]}
    for name, expected in record["sha256"].items():
        if name in special:
            path = local(root, special[name])
        elif Path(name).name == name:
            path = work / name
        else:
            raise ValueError("unexpected annotation hash path")
        check_hash(path, expected)
    if record["kept_rows"] != spec["rows"] or record["support"] != spec["support"]:
        raise ValueError("registered support or row count changed")
    sampling = read_json(work / "sampling.json")
    for name, expected in sampling["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("unexpected sampling hash path")
        check_hash(work / name, expected)
    frozen = read_json(local(root, spec["frozen"]))
    if any(
        frozen[k] != v
        for k, v in {"candidate": "C", "threshold": None, "temperature": None}.items()
    ):
        raise ValueError("expected the frozen uncalibrated candidate C")
    for relative, expected in frozen["pinned_sha256"].items():
        check_hash(local(root, relative), expected)
    for name, expected in frozen["inference"]["assets_sha256"].items():
        check_hash(local(root, "models/topic/" + name), expected)
    if runtime:
        for name, expected in spec["runtime"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f"runtime version mismatch: {name}; expected {expected}")
    return {"record": record, "frozen": frozen, "sampling": sampling}


def unique_ids(frame: pd.DataFrame, expected_rows: int | None = None) -> None:
    if "id" not in frame or frame.id.eq("").any() or frame.id.duplicated().any():
        raise ValueError("missing, empty or duplicate IDs")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError("row count mismatch")


def align_ids(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    unique_ids(frame)
    if len(ids) != len(set(ids)) or set(frame.id) != set(ids):
        raise ValueError("ID join has missing, extra or duplicate rows")
    return frame.set_index("id").loc[ids].reset_index()


def prediction_texts(root: Path, spec: dict) -> pd.DataFrame:
    work = local(root, spec["work_folder"])
    kept = pd.read_csv(work / "labels_final.csv", usecols=["id"], dtype=str, keep_default_na=False)
    unique_ids(kept, spec["rows"])
    sample = pd.read_csv(work / "sample.csv", dtype=str, keep_default_na=False)
    unique_ids(sample)
    if sample.columns.tolist() != ["id", "text"] or sample.text.str.strip().eq("").any():
        raise ValueError("invalid blind text input")
    selected = sample.loc[sample.id.isin(kept.id)].reset_index(drop=True)
    if len(selected) != len(kept):
        raise ValueError("frozen IDs missing from sample")
    return selected


def publication_order(root: Path, spec: dict, ids: list[str]) -> list[str]:
    # Hash checks above read bytes only. No provenance fields besides id/date are parsed.
    frame = pd.read_csv(
        local(root, spec["work_folder"]) / "provenance.csv",
        usecols=["id", "date"],
        dtype=str,
        keep_default_na=False,
    )
    unique_ids(frame)
    selected = align_ids(frame.loc[frame.id.isin(ids)], ids)
    selected["date"] = pd.to_datetime(selected.date, utc=True, errors="raise")
    if selected.date.isna().any():
        raise ValueError("publication date missing")
    return selected.sort_values(["date", "id"], kind="stable").id.tolist()


def cached_probs(texts: list[str], names: list[str], probabilities: np.ndarray):
    if (
        probabilities.shape != (len(texts), len(names))
        or len(set(names)) != len(names)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("invalid probability matrix")

    def cached(items):
        if list(items) != texts:
            raise ValueError("probabilities requested with different texts/order")
        return list(names), probabilities.copy()

    return cached


def without_product_fallback(labels, sources, names, probabilities):
    """Undo exactly one recorded route, preserving the complete reused C implementation."""
    result, routes = list(labels), list(sources)
    top = np.asarray(names)[probabilities.argmax(axis=1)]
    if not len(labels) == len(sources) == len(top):
        raise ValueError("C output length mismatch")
    for i, source in enumerate(sources):
        if source == "model_product_to_other":
            if labels[i] != "Other" or top[i] != "Product news":
                raise ValueError("unexpected Product-to-Other fallback semantics")
            result[i], routes[i] = str(top[i]), "model"
    return result, routes


def predict_candidates(texts, engine, student, frozen):
    names, p = engine(texts)
    cached = cached_probs(texts, names, p)
    cfg = Config(frozen["candidate"], frozen["threshold"], frozen["temperature"])
    c = build_predictor(cfg, cached)(texts)
    outputs = {
        "A": build_predictor(Config("A"), cached)(texts),
        "C": c,
        "C-noPF": without_product_fallback(*c, names, p),
    }
    if student is not None:
        student_names, student_p = student(texts)
        outputs["E25-v3"] = build_predictor(cfg, cached_probs(texts, student_names, student_p))(
            texts
        )
    return outputs


def bootstrap_indices(n: int, draws: int, seed: int, block_length: int) -> np.ndarray:
    if n < 1 or draws < 1 or not 1 <= block_length <= n:
        raise ValueError("invalid moving-block dimensions")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_length + 1, size=(draws, math.ceil(n / block_length)))
    return (starts[:, :, None] + np.arange(block_length)).reshape(draws, -1)[:, :n]


def metric_vector(y, pred, classes, eligible, secondary) -> np.ndarray:
    if len(y) == 0 or len(y) != len(pred) or not eligible or not secondary:
        raise ValueError("invalid metric dimensions or class lists")
    matrix = np.bincount(y * len(classes) + pred, minlength=len(classes) ** 2).reshape(
        len(classes), len(classes)
    )
    tp, support, predicted = matrix.diagonal(), matrix.sum(axis=1), matrix.sum(axis=0)

    def divide(a, b):
        return np.divide(a, b, out=np.zeros(len(classes), dtype=float), where=b != 0)

    precision, recall = divide(tp, predicted), divide(tp, support)
    f1 = divide(2 * tp, support + predicted)
    return np.r_[
        f1[eligible].mean(), f1[secondary].mean(), tp.sum() / len(y), precision, recall, f1
    ]


def score_predictions(labels, predictions, order, spec) -> dict:
    """Pure metric calculation; callers perform all frozen-input and one-run checks first."""
    classes, candidates = spec["classes"], spec["candidates"]
    labels, predictions = align_ids(labels, order), align_ids(predictions, order)
    if not labels.label.isin(classes).all():
        raise ValueError("invalid reference class")
    support = labels.label.value_counts().reindex(classes, fill_value=0).to_dict()
    if support != spec["support"] or len(labels) != spec["rows"]:
        raise ValueError("actual label support differs from the preregistration")
    eligible = [classes.index(c) for c in spec["eligible_classes"]]
    secondary = [classes.index(c) for c in spec["secondary_classes"]]
    y = np.array([classes.index(c) for c in labels.label])
    boot = spec["bootstrap"]
    indices = bootstrap_indices(len(y), boot["draws"], boot["seed"], boot["block_length"])
    estimates, draws = {}, {}
    for candidate in candidates:
        if not predictions[candidate].isin(classes).all():
            raise ValueError("invalid predicted class")
        p = np.array([classes.index(c) for c in predictions[candidate]])
        estimates[candidate] = metric_vector(y, p, classes, eligible, secondary)
        draws[candidate] = np.array(
            [metric_vector(y[i], p[i], classes, eligible, secondary) for i in indices]
        )

    def interval(values):
        return np.percentile(values, boot["percentiles"]).tolist()

    def entry(candidate, column):
        return {
            "estimate": float(estimates[candidate][column]),
            "ci95": interval(draws[candidate][:, column]),
            "difference_vs_C": float(estimates[candidate][column] - estimates["C"][column]),
            "difference_vs_C_ci95": interval(draws[candidate][:, column] - draws["C"][:, column]),
        }

    scores, decisions = {}, {}
    for candidate in candidates:
        scores[candidate] = {
            name: entry(candidate, i)
            for i, name in enumerate(["macro_f1_primary", "macro_f1_without_other", "accuracy"])
        }
        scores[candidate]["per_class"] = {
            name: {
                "support": int(support[name]),
                **{
                    metric: entry(candidate, 3 + offset * len(classes) + i)
                    for offset, metric in enumerate(["precision", "recall", "f1"])
                },
            }
            for i, name in enumerate(classes)
        }
        primary = scores[candidate]["macro_f1_primary"]
        gain = primary["difference_vs_C"]
        lower = primary["difference_vs_C_ci95"][0]
        rule = spec["decision"]
        allowed = candidate in rule["challengers"]
        decisions[candidate] = {
            "challenger": allowed,
            "gain_at_least_0_02": gain >= rule["minimum_primary_gain"],
            "paired_lower_above_zero": lower > rule["paired_lower_bound_strictly_above"],
            "qualifies_for_owner_review": bool(
                allowed
                and gain >= rule["minimum_primary_gain"]
                and lower > rule["paired_lower_bound_strictly_above"]
            ),
        }
    return {
        "rows": len(y),
        "classes": classes,
        "primary_classes": spec["eligible_classes"],
        "secondary_classes": spec["secondary_classes"],
        "support": support,
        "bootstrap": boot
        | {
            "ordering": "publication date, id",
            "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        },
        "scores": scores,
        "decision": decisions,
        "running_system": "C",
        "engine_changed": False,
        "selected_model": None,
        "limits": LIMITS,
        "deviations": ["Owner dropped 78 disagreements instead of protocol settlement."],
    }


def write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    if not replace:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    else:
        temporary = path.with_name(path.name + ".tmp")
        write_json(temporary, value)
        os.replace(temporary, path)


def claim_prediction(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("E44 run already exists; preserve it, repeat prediction refused")
    lock = output / "e44.run_lock"
    write_json(lock, {"phase": "predicting", "started_utc": datetime.now(UTC).isoformat()})
    return lock


def run_predict(root: Path) -> dict:
    spec = load_plan(root)
    inputs = verify_inputs(root, spec, runtime=True)
    frame = prediction_texts(root, spec)
    lock = claim_prediction(root / "data/outputs")
    from riskengine.eval.e24_inference import (
        EvaluationProbs,
        inference_metadata,
        validate_inference_metadata,
    )
    from riskengine.eval.e25_candidate import student_probs

    validate_inference_metadata(
        inference_metadata(root / "models/topic"), inputs["frozen"]["inference"]
    )
    engine = EvaluationProbs(root / "models/topic")
    student = (
        student_probs(local(root, spec["student_folder"]))
        if "E25-v3" in (spec["candidates"])
        else None
    )
    results = predict_candidates(frame.text.tolist(), engine, student, inputs["frozen"])
    if list(results) != spec["candidates"]:
        raise ValueError("candidate list differs from plan")
    prediction = frame[["id"]].copy()
    for candidate, (labels, sources) in results.items():
        prediction[candidate], prediction[candidate + "_source"] = labels, sources
    verify_inputs(root, spec, runtime=True)
    path = root / "data/outputs/e44_predictions.csv"
    prediction.to_csv(path, index=False, mode="x")
    state = read_json(lock) | {
        "phase": "predicted",
        "predicted_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256(root / PLAN),
        "labels_sha256": spec["labels_sha256"],
        "rows": len(frame),
        "ids_sha256": hashlib.sha256("\n".join(frame.id).encode()).hexdigest(),
        "predictions_sha256": sha256(path),
        "scorer_sha256": sha256(Path(__file__)),
        "runtime": spec["runtime"],
        "engine_inference": engine.inference_metadata,
        "student_inference": student.inference_metadata if student is not None else None,
    }
    write_json(lock, state, replace=True)
    return {"phase": state["phase"], "rows": len(frame), "predictions": str(path)}


def run_score(root: Path) -> dict:
    spec = load_plan(root)
    verify_inputs(root, spec, runtime=True)
    output = root / "data/outputs"
    lock = output / "e44.run_lock"
    state = read_json(lock)
    if state["phase"] != "predicted" or (output / "e44_results.json").exists():
        raise FileExistsError("prediction is incomplete or scoring already claimed")
    if state["plan_sha256"] != sha256(root / PLAN):
        raise ValueError("prediction used a different plan")
    if state["labels_sha256"] != spec["labels_sha256"]:
        raise ValueError("prediction used a different reference")
    check_hash(Path(__file__), state["scorer_sha256"])
    check_hash(output / "e44_predictions.csv", state["predictions_sha256"])
    # An exclusive claim also prevents two score processes from using the same predicted lock.
    claim = output / "e44.run_lock.score_claim"
    with claim.open("x") as stream:
        stream.write(datetime.now(UTC).isoformat() + "\n")
    write_json(lock, state | {"phase": "scoring"}, replace=True)
    frame = prediction_texts(root, spec)
    if state["ids_sha256"] != hashlib.sha256("\n".join(frame.id).encode()).hexdigest():
        raise ValueError("prediction ID order changed")
    labels = pd.read_csv(
        local(root, spec["work_folder"]) / "labels_final.csv",
        usecols=["id", "label"],
        dtype=str,
        keep_default_na=False,
    )
    predictions = pd.read_csv(output / "e44_predictions.csv", dtype=str, keep_default_na=False)
    expected_columns = [
        "id",
        *(column for c in spec["candidates"] for column in (c, c + "_source")),
    ]
    if predictions.columns.tolist() != expected_columns:
        raise ValueError("unexpected prediction schema")
    align_ids(predictions, frame.id.tolist())
    report = score_predictions(
        labels, predictions, publication_order(root, spec, frame.id.tolist()), spec
    )
    verify_inputs(root, spec, runtime=True)
    report.update(
        {
            "experiment": "E44b",
            "completed_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256(root / PLAN),
            "label_sha256": spec["labels_sha256"],
            "prediction_sha256": state["predictions_sha256"],
            "scorer_sha256": sha256(Path(__file__)),
            "candidate_files_sha256": spec["expected_sha256"],
            "inference": {
                "C": state["engine_inference"],
                "E25-v3": state["student_inference"],
            },
        }
    )
    write_json(output / "e44_results.json", report)
    write_json(
        lock,
        state
        | {
            "phase": "scored",
            "results_sha256": sha256(output / "e44_results.json"),
            "completed_utc": report["completed_utc"],
        },
        replace=True,
    )
    return {"phase": "scored", "results": str(output / "e44_results.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict", action="store_true")
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps((run_predict if args.predict else run_score)(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
