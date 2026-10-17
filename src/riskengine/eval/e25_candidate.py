"""Candidate E: unchanged C rules with the v2 student's probabilities. D1 diagnostics only.

uv run python -m riskengine.eval.e25_candidate --d1

No D2/final/credit labels are loaded, and no selection, calibration or training is performed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from riskengine.eval.e24_dev import Config, Predictor, Probs, build_predictor, engine_probs
from riskengine.eval.e24_gate import CLASSES, E24_SECTION_SHA, GateError, section_sha, sha256
from riskengine.eval.e24_inference import EvaluationProbs, save_probability_cache
from riskengine.nlp.events import LEARNED_CLASSES, OWN_TOPIC
from riskengine.paths import DATA, ROOT

TASK = ROOT / "private_notes/annotation_uploads/task1"
STUDENT_MODEL = TASK / "e25_student_output/v2/models/e25-student-bge-small-en-v1.5-int8"
D1_INPUT = TASK / "development/D1_to_label.csv"
D1_ROWS = 153
PREFIX = "e25_candidate_E_"


def student_probs(model_path: Path = STUDENT_MODEL) -> Probs:
    backend = EvaluationProbs(model_path)
    if len(backend.labels) != len(LEARNED_CLASSES) or set(backend.labels) != set(LEARNED_CLASSES):
        raise GateError("student must predict exactly the nine learned event classes")
    return backend


def build_candidate(probs: Probs | None = None, temperature: float | None = None) -> Predictor:
    """Reuse C's exact routing; only the probability model is replaced."""
    return build_predictor(Config("C", temperature=temperature), probs or student_probs())


def read_d1(input_path: Path, reference_root: Path) -> tuple[pd.DataFrame, dict]:
    if input_path.name != "D1_to_label.csv":
        raise GateError("this diagnostic accepts the D1 input only")
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if len(frame) != D1_ROWS or frame["item_id"].duplicated().any():
        raise GateError("D1 must have the expected number of unique rows")
    refs = {}
    for version in ("v6", "v7"):
        path = reference_root / f"reference_{version}/development/D1_labels.csv"
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        if (
            len(lab) != len(frame)
            or lab["item_id"].duplicated().any()
            or set(lab["item_id"]) != set(frame["item_id"])
            or not lab["label"].isin(CLASSES).all()
            or not lab["ambiguous"].isin(["Y", "N"]).all()
        ):
            raise GateError(f"invalid or mismatched {version} D1 labels")
        lab = lab.set_index("item_id").loc[frame["item_id"]].reset_index()
        frame[f"reference_{version}"] = lab["label"].tolist()
        frame[f"ambiguous_{version}"] = lab["ambiguous"].tolist()
        refs[version] = {"path": str(path), "sha256": sha256(path), "source": "AI-derived"}
    return frame, refs


def metrics(y: list[str], pred: list[str]) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    supported = [label for label in CLASSES if label in y]
    return {
        "macro_f1_12_primary": float(
            f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "macro_f1_supported_supplementary": float(
            f1_score(y, pred, labels=supported, average="macro", zero_division=0)
        ),
        "supported_classes": supported,
        "accuracy": float(accuracy_score(y, pred)),
        "correct": int(np.equal(y, pred).sum()),
        "predicted_counts": {label: pred.count(label) for label in CLASSES},
        "per_class": {
            label: {
                "precision": float(pp),
                "recall": float(rr),
                "f1": float(ff),
                "support": int(nn),
            }
            for label, pp, rr, ff, nn in zip(CLASSES, p, r, f, n, strict=True)
        },
        "confusion": {
            "labels": CLASSES,
            "matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        },
    }


def run_d1() -> dict:
    output = DATA / "outputs"
    if list(output.glob(PREFIX + "*")):
        raise GateError("candidate E diagnostic already exists; preserve the recorded run")
    if (output / "e24_frozen.json").exists():
        raise GateError("a frozen candidate already exists; this diagnostic precedes selection")
    if section_sha(ROOT / "docs/DECISIONS.md") != E24_SECTION_SHA:
        raise GateError("locked E24 section changed")
    protected_paths = [
        ROOT / "docs/DECISIONS.md",
        ROOT / "src/riskengine/nlp/events.py",
        ROOT / "src/riskengine/nlp/classifier.py",
        ROOT / "src/riskengine/eval/e24_rules.py",
        ROOT / "src/riskengine/gate.py",
        D1_INPUT,
        *(
            p
            for folder in (OWN_TOPIC, STUDENT_MODEL)
            for p in sorted(folder.iterdir())
            if p.is_file()
        ),
        *(TASK / f"reference_{v}/development/D1_labels.csv" for v in ("v6", "v7")),
    ]
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in protected_paths}
    frame, refs = read_d1(D1_INPUT, TASK)
    texts = frame["text"].tolist()
    probabilities = {}
    inference = {}
    for candidate, factory in (("C", engine_probs), ("E", student_probs)):
        start = perf_counter()
        backend = factory()
        names, p = backend(texts)
        if p.shape != (len(frame), 9) or not np.isfinite(p).all():
            raise GateError("invalid D1 probability matrix")
        metadata = backend.inference_metadata
        save_probability_cache(
            output / f"{PREFIX}{candidate}_probabilities.npz",
            frame["item_id"].tolist(),
            names,
            p,
            metadata,
            input_sha256=sha256(D1_INPUT),
            candidate=candidate,
        )
        probabilities[candidate] = (names, p)
        inference[candidate] = {
            "settings": metadata,
            "load_and_inference_seconds": perf_counter() - start,
        }

        def cached(items, names=names, p=p):
            if list(items) != texts:
                raise GateError("D1 cache requested with different texts or order")
            return names, p.copy()

        pred, sources = build_predictor(Config(candidate), cached)(texts)
        frame[f"{candidate}_prediction"], frame[f"{candidate}_source"] = pred, sources

    scores, changes, per_class, matrices = {}, [], [], []
    effects = {}
    for version in ("v6", "v7"):
        y = frame[f"reference_{version}"].tolist()
        scores[version] = {c: metrics(y, frame[f"{c}_prediction"].tolist()) for c in ("C", "E")}
        counts = {"fixed": 0, "broken": 0, "changed_error": 0}
        for row in frame.to_dict("records"):
            c, e, actual = row["C_prediction"], row["E_prediction"], row[f"reference_{version}"]
            if c == e:
                continue
            effect = "fixed" if e == actual else "broken" if c == actual else "changed_error"
            counts[effect] += 1
            changes.append(
                {
                    "reference": version,
                    "item_id": row["item_id"],
                    "text": row["text"],
                    "reference_label": actual,
                    "C": c,
                    "E": e,
                    "effect": effect,
                    "C_source": row["C_source"],
                    "E_source": row["E_source"],
                }
            )
        effects[version] = counts
        for candidate, score in scores[version].items():
            for label, values in score["per_class"].items():
                per_class.append(
                    {"reference": version, "candidate": candidate, "class": label, **values}
                )
            for label, values in zip(CLASSES, score["confusion"]["matrix"], strict=True):
                matrices.append(
                    {
                        "reference": version,
                        "candidate": candidate,
                        "actual_class": label,
                        **dict(zip(CLASSES, values, strict=True)),
                    }
                )
    for path, wanted in protected.items():
        if sha256(ROOT / path) != wanted:
            raise GateError(f"protected file changed: {path}")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "D1 diagnostic only",
        "D1_rows": len(frame),
        "reference_provenance": "AI-derived, same-resolver v7 repeat review",
        "references": refs,
        "student": str(STUDENT_MODEL.relative_to(ROOT)),
        "student_selection": "seed 43, epoch 3; chosen on training-pool validation, before D1",
        "candidate_E": "unchanged C rules and E13 credit rules, v2 student probabilities",
        "temperature": None,
        "threshold": None,
        "tuning_or_training": False,
        "D2_final_or_E23_labels_opened": False,
        "candidate_selected_or_frozen": False,
        "inference": inference,
        "scores": scores,
        "E_vs_C_changes": effects,
        "protected_files_sha256": protected,
        "source_files_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                Path(__file__),
                ROOT / "src/riskengine/eval/e24_dev.py",
                ROOT / "src/riskengine/eval/e24_inference.py",
            )
        },
    }
    frame.to_csv(output / f"{PREFIX}predictions.csv", index=False)
    pd.DataFrame(
        changes,
        columns=[
            "reference",
            "item_id",
            "text",
            "reference_label",
            "C",
            "E",
            "effect",
            "C_source",
            "E_source",
        ],
    ).to_csv(output / f"{PREFIX}changes_vs_C.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / f"{PREFIX}per_class.csv", index=False)
    pd.DataFrame(matrices).to_csv(output / f"{PREFIX}confusion.csv", index=False)
    (output / f"{PREFIX}D1.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Candidate E: D1 only",
        "",
        "E reuses C's unchanged rules with the v2 student. Production is unchanged.",
        "Both models use the same per-row CPU inference method. No calibration, threshold search,",
        "retraining, D2 selection or final inference occurred. Reference labels are AI-derived.",
        "",
        "| Ref | Candidate | Fixed-12 F1 | Supported F1 (supplementary) | Accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for version, result in scores.items():
        for candidate, score in result.items():
            lines.append(
                f"| {version} | {candidate} | {score['macro_f1_12_primary']:.6f} | "
                f"{score['macro_f1_supported_supplementary']:.6f} | "
                f"{score['accuracy']:.2%} ({score['correct']}/153) |"
            )
    lines += [
        "",
        "| Reference | E fixes vs C | E breaks vs C | Changes one error to another |",
        "|---|---:|---:|---:|",
    ]
    for version, counts in effects.items():
        lines.append(
            f"| {version} | {counts['fixed']} | {counts['broken']} | {counts['changed_error']} |"
        )
    lines += [
        "",
        "Full precision/recall/support and fixed-12 confusion matrices are in the matching CSVs",
        "and JSON. Every changed row is in changes_vs_C.csv; all 153 predictions are preserved.",
        "D1 is a development diagnostic. Low class support and AI labels limit conclusions.",
        "Absent credit classes have no measurable D1 recall. No candidate is selected here.",
        "Production is unchanged. The existing D2 selector/freeze still handles A–D only;",
        "adding E to that procedure requires the separate authorized selection step.",
        "",
    ]
    (output / f"{PREFIX}D1.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1", action="store_true", required=True)
    parser.parse_args()
    result = run_d1()
    print(
        json.dumps(
            {
                "scores": {
                    v: {
                        c: {
                            k: s[k]
                            for k in (
                                "macro_f1_12_primary",
                                "macro_f1_supported_supplementary",
                                "accuracy",
                                "correct",
                            )
                        }
                        for c, s in result["scores"][v].items()
                    }
                    for v in ("v6", "v7")
                },
                "E_vs_C_changes": result["E_vs_C_changes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
