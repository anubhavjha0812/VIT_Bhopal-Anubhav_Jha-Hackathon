"""Score the fixed E40 arms once after the Kaggle raw output is returned.

This evaluator never loads Laya or any other learned model. E23 is design data.
"""

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.e39_calibration import claim_run, sha256, write_json
from riskengine.gate_rules import blocked_reason
from riskengine.paths import DATA

ROOT = DATA.parent
OUT = DATA / "outputs"
RAW = OUT / "e40_laya_raw.csv"
RAW_MANIFEST = OUT / "e40_laya_raw_manifest.json"
RESULT = OUT / "e40_decision_model.json"
LOCK = OUT / "e40_decision_model.run_lock"
E23 = OUT / "e24_e23_gate_v6_rows.csv"
EIGHT_K = OUT / "e25_credit_8k_full_rows.csv"
TARGETS = ROOT / "private_notes/annotation_uploads/task1/e40_inputs/8k_targets.csv"
KAGGLE_SCRIPT = ROOT / "kaggle/e40_laya_baseline.py"
CREDIT = {"Default", "Credit downgrade", "Covenant breach"}


def kaggle_contract():
    """Reuse the self-contained script's input/question contract, without main()."""
    spec = importlib.util.spec_from_file_location("e40_kaggle_contract", KAGGLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_threshold(eight_k: pd.DataFrame) -> dict:
    """A single shared strict cutoff for Q1/Q2; E23 is forbidden here."""
    if eight_k.empty or not eight_k["dataset"].eq("8K").all():
        raise ValueError("Threshold fitting accepts 8K rows only; E23 is forbidden")
    p = eight_k[["Q1", "Q2"]].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid threshold-setting probabilities")
    y = eight_k["credit_true"].to_numpy(dtype=bool)
    scores = p.min(axis=1)
    # Strict > yields every possible split at 0, observed scores and 1.
    # Resolve equal F1 conservatively with the highest threshold, fixed before scoring.
    grid = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best = None
    for threshold in grid:
        predicted = scores > threshold
        tp, fp, fn = (
            int((predicted & y).sum()),
            int((predicted & ~y).sum()),
            int((~predicted & y).sum()),
        )
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        candidate = {"threshold": float(threshold), "F1": f1, "TP": tp, "FP": fp, "FN": fn}
        if best is None or (f1, threshold) > (best["F1"], best["threshold"]):
            best = candidate
    return {
        **best,
        "fitted_on": "8K only",
        "rows": len(eight_k),
        "fit_ids": eight_k["item_id"].tolist(),
        "primary_form": "noul",
        "rule": "Q1 > t AND Q2 > t; maximise credit-label F1; ties use highest t",
        "candidate_count": len(grid),
        "E23_rows_used": 0,
    }


def arm_predictions(
    a: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, threshold: float
) -> dict:
    b = (q1 > threshold) & (q2 > threshold)
    return {"A": np.asarray(a, dtype=bool), "B": b, "C": a & b & (q3 < 0.5)}


def binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    return {
        "alerts": tp + fp,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def strict_bool(values: pd.Series) -> np.ndarray:
    mapped = values.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("Expected explicit true/false decisions")
    return mapped.to_numpy(dtype=bool)


def arm_a(e23: pd.DataFrame) -> np.ndarray:
    """Replay only the two pure rules on stored fire decisions, never the gate/model."""
    old_fire = strict_bool(e23["predicted_fire"])
    blocked = np.array(
        [
            blocked_reason(text, event) is not None
            for text, event in zip(e23["headline"], e23["rule_class"], strict=True)
        ]
    )
    return old_fire & ~blocked


def validate_raw(raw: pd.DataFrame, inputs: list[dict], contract) -> None:
    expected = {
        (r["dataset"], r["item_id"], f, q)
        for r in inputs
        for f in contract.FORMS
        for q in contract.QUESTIONS
    }
    columns = ["dataset", "item_id", "form", "question"]
    keys = list(raw[columns].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Raw output must contain every fixed ID/question/form exactly once")
    lookup = {(r["dataset"], r["item_id"]): r for r in inputs}
    for row in raw.itertuples(index=False):
        source = lookup[(row.dataset, row.item_id)]
        if row.target_company != source[
            "target_company"
        ] or row.text_sha256 != contract.text_digest(source["text"]):
            raise ValueError("Raw text or target differs from the fixed input")
    p = raw[["p_yes", "p_no"]].to_numpy(dtype=float)
    if (
        not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
        or not np.allclose(p.sum(axis=1), 1, atol=0.0002, rtol=0)
    ):
        raise ValueError("Invalid raw yes/no probabilities")
    timing = raw["call_seconds"].to_numpy(dtype=float)
    if not np.isfinite(timing).all() or (timing <= 0).any():
        raise ValueError("Every call must have a positive, finite measured duration")
    group = raw.groupby(["dataset", "item_id", "form"])["call_seconds"]
    if not group.nunique().eq(1).all():
        raise ValueError("Q1/Q2/Q3 must carry the same three-question call time")


def validate_manifest(
    manifest: dict, raw_path: Path, manifest_path: Path, contract, sources: dict
) -> dict:
    expected = {
        "package_version": contract.LAYA_VERSION,
        "model_id": contract.MODEL_ID,
        "subfolder": contract.SUBFOLDER,
        "questions_sha256": contract.question_digest(),
        "script_sha256": sha256(KAGGLE_SCRIPT),
        "raw_sha256": sha256(raw_path),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Kaggle manifest mismatch: {key}")
    if manifest.get("inputs_sha256") != {k: sha256(p) for k, p in sources.items()}:
        raise ValueError("Kaggle input hashes differ from local sources/target map")
    files = manifest.get("checkpoint_files_sha256", {})
    if "model.safetensors" not in files or "rl_agent_config.json" not in files:
        raise ValueError("Manifest must identify the actual downloaded checkpoint files")
    if not manifest.get("download_utc") or not manifest.get("device"):
        raise ValueError("Missing download date or Kaggle GPU identity")
    return {"path": str(manifest_path), "sha256": sha256(manifest_path), **manifest}


def probability_table(raw: pd.DataFrame, form: str) -> pd.DataFrame:
    return (
        raw[raw["form"] == form]
        .pivot(index=["dataset", "item_id"], columns="question", values="p_yes")
        .reset_index()
    )


def evaluate_form(table: pd.DataFrame, e23: pd.DataFrame, a: np.ndarray, threshold: float) -> dict:
    rows = e23.merge(
        table[table["dataset"] == "E23"],
        left_on="headline_id",
        right_on="item_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if rows[["Q1", "Q2", "Q3"]].isna().any().any():
        raise ValueError("Missing E23 question probabilities")
    truth = strict_bool(rows["reference_fire"])
    predictions = arm_predictions(a, *(rows[q].to_numpy() for q in ("Q1", "Q2", "Q3")), threshold)
    arms = {name: binary_metrics(predicted, truth) for name, predicted in predictions.items()}
    removed = rows[a & ~predictions["C"]].copy()
    return {
        "threshold": threshold,
        "arms": arms,
        "C_true_lost": arms["A"]["TP"] - arms["C"]["TP"],
        "C_false_removed": arms["A"]["FP"] - arms["C"]["FP"],
        "C_removed_alerts": removed[
            ["headline_id", "headline", "target_company", "reference_fire", "Q1", "Q2", "Q3"]
        ].to_dict("records"),
    }


def promotion(primary: dict, manifest: dict) -> dict:
    a, c = primary["arms"]["A"], primary["arms"]["C"]
    timing = manifest.get("timing", {}).get("noul", {})
    latency_ok = (
        isinstance(timing.get("total_seconds"), (int, float)) and timing["total_seconds"] > 0
    )
    checks = {
        "keeps_at_least_38_true": c["TP"] >= 38,
        "removes_at_least_10_false": a["FP"] - c["FP"] >= 10,
        "precision_above_A": c["precision"] > a["precision"],
        "Kaggle_latency_reported_and_offline": latency_ok
        and manifest.get("offline_after_download") is True,
    }
    return {
        "checks": checks,
        "decision": "promote to a separately registered unseen-set test only"
        if all(checks.values())
        else "not promoted",
        "live_gate_changed": False,
        "secondary_choice_used_for_decision": False,
    }


def run(raw_path: Path = RAW, manifest_path: Path = RAW_MANIFEST) -> dict:
    # A missing download must not consume the one scoring lock.
    if not raw_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Kaggle raw CSV and its manifest are required; E40 not run")
    if RESULT.exists() or LOCK.exists():
        raise FileExistsError("E40 already scored or locked")
    contract = kaggle_contract()
    inputs = contract.load_inputs(EIGHT_K, E23, TARGETS)
    raw = pd.read_csv(raw_path, dtype={"item_id": str})
    validate_raw(raw, inputs, contract)
    manifest = validate_manifest(
        json.loads(manifest_path.read_text()),
        raw_path,
        manifest_path,
        contract,
        {"8K": EIGHT_K, "E23": E23, "targets": TARGETS},
    )
    e23 = pd.read_csv(E23, dtype={"headline_id": str})
    eight_k = pd.read_csv(EIGHT_K, dtype={"item_id": str})
    a = arm_a(e23)
    baseline = binary_metrics(a, strict_bool(e23["reference_fire"]))
    if (baseline["alerts"], baseline["TP"], baseline["FP"]) != (59, 39, 20):
        raise ValueError("Arm A no longer reproduces fixed E36b counts (59,39,20)")
    primary_table = probability_table(raw, "noul")
    # Only 8K data can reach choose_threshold; E23 is scored after the scalar is frozen.
    fit_rows = eight_k[["item_id", "final"]].merge(
        primary_table[primary_table["dataset"] == "8K"],
        on="item_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    fit_rows["credit_true"] = fit_rows["final"].isin(CREDIT)
    claim_run(RESULT, LOCK)
    threshold = choose_threshold(fit_rows)
    forms = {
        form: evaluate_form(probability_table(raw, form), e23, a, threshold["threshold"])
        for form in contract.FORMS
    }
    out = {
        "experiment": "E40",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": "docs/experiments/DECISION_MODEL_BENCH.md#e40",
        "Kaggle_run": manifest,
        "threshold": threshold,
        "results": forms,
        "decision": promotion(forms["noul"], manifest),
        "secondary_note": "Choice uses the same primary-frozen threshold, reported only; "
        "no second threshold fit and no promotion vote.",
        "limits": [
            "E23 is seen design data, not an unseen accuracy claim.",
            "AI labels from one labeller family; no human check.",
            "Questions written by us; 39 true alerts make the retention rule coarse.",
            "No deployment; adoption needs a separately registered unseen set.",
        ],
    }
    write_json(RESULT, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--manifest", type=Path, default=RAW_MANIFEST)
    args = parser.parse_args()
    result = run(args.raw, args.manifest)
    print(
        json.dumps(
            {
                "results": {f: r["arms"] for f, r in result["results"].items()},
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
