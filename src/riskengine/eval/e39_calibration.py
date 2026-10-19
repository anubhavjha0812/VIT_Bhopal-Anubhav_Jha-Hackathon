"""One offline, cross-fitted calibration benchmark of the two stored D1 caches.

No inference imports: this module reads probabilities, never loads a model.
"""

import argparse
import hashlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, softmax
from sklearn.model_selection import KFold, StratifiedKFold

from riskengine.paths import DATA

ROOT = DATA.parent
OUT = DATA / "outputs"
RESULT = OUT / "e39_calibration.json"
FOLDS_FILE = OUT / "e39_calibration_folds.csv"
LOCK = OUT / "e39_calibration.run_lock"
REFERENCE = ROOT / "private_notes/annotation_uploads/task1/reference_v6/development/D1_labels.csv"
D1_INPUT = ROOT / "private_notes/annotation_uploads/task1/development/D1_to_label.csv"
CACHES = {
    "event_model": OUT / "e24_d1_model_probabilities.npz",
    "student_E": OUT / "e25_candidate_E_E_probabilities.npz",
}
BINARY_FILES = {
    "E23": OUT / "e24_e23_gate_v6_rows.csv",
    "8K": OUT / "e25_credit_8k_full_rows.csv",
}
SEED, N_FOLDS, N_BINS, N_BOOTSTRAP = 39, 5, 10, 1000
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
LOG_T_BOUNDS = (-5.0, 5.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_run(result: Path, lock: Path, *other_outputs: Path) -> None:
    """Keep the exclusive lock even on failure; never silently rescore a run."""
    if any(p.exists() for p in (result, lock, *other_outputs)):
        raise FileExistsError("One scored run only: result, lock or output already exists")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x") as stream:
        json.dump({"started_utc": datetime.now(UTC).isoformat()}, stream)


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2:
        raise ValueError("Expected a nonempty rows-by-classes probability matrix")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Non-finite or out-of-range probability")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Probability rows do not sum to one")
    return p


def reliability(p: np.ndarray, y: np.ndarray, bins: int = N_BINS) -> list[dict]:
    """Equal-mass top-label bins; stable order breaks confidence ties."""
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    order = np.argsort(confidence, kind="stable")
    return [
        {
            "count": len(ix),
            "confidence": float(confidence[ix].mean()),
            "accuracy": float(correct[ix].mean()),
        }
        for ix in np.array_split(order, min(bins, len(order)))
    ]


def ece(p: np.ndarray, y: np.ndarray, bins: int = N_BINS) -> float:
    return float(
        sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in reliability(p, y, bins))
        / len(y)
    )


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    onehot = np.eye(p.shape[1])[y]
    return {
        "ECE": ece(p, y),
        "Brier": float(np.square(p - onehot).sum(axis=1).mean()),
        "NLL": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


def temperature_scale(p: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(np.log(np.clip(p, 1e-12, 1)) / temperature, axis=1)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    log_p = np.log(np.clip(p, 1e-12, 1))

    def training_nll(log_t):
        scaled = softmax(log_p / np.exp(log_t), axis=1)
        return -np.log(np.clip(scaled[np.arange(len(y)), y], 1e-12, 1)).mean()

    fit = minimize_scalar(
        training_nll,
        bounds=LOG_T_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not fit.success:
        raise RuntimeError(f"Temperature optimisation failed: {fit.message}")
    return float(np.exp(fit.x))


def platt_crossfit(scores: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple:
    """Logistic on logit, training-fold fits only; unused when scores are absent."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Platt scaling requires actual probabilities in [0,1]")
    bounded = np.clip(scores, 1e-12, 1 - 1e-12)
    logits = np.log(bounded / (1 - bounded))
    after, audit = np.empty_like(scores), []
    for fold in range(N_FOLDS):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if len(np.unique(y[train])) != 2 or not len(test):
            raise ValueError("Binary cross-fitting requires both labels in every training fold")

        def objective(params, train=train):
            z = params[0] * logits[train] + params[1]
            return float(np.mean(np.logaddexp(0, z) - y[train] * z))

        fitted = minimize(objective, [1.0, 0.0], method="BFGS", options={"gtol": 1e-6})
        if not fitted.success:
            raise RuntimeError(f"Platt optimisation failed: {fitted.message}")
        after[test] = expit(fitted.x[0] * logits[test] + fitted.x[1])
        audit.append(
            {
                "fold": fold,
                "slope": float(fitted.x[0]),
                "intercept": float(fitted.x[1]),
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
            }
        )
    return after, audit


def make_folds(y: np.ndarray) -> tuple[np.ndarray, str]:
    if len(y) < N_FOLDS:
        raise ValueError("Five-fold cross-fitting needs at least five rows")
    counts = np.unique(y, return_counts=True)[1]
    if counts.max() >= N_FOLDS:
        splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        method = "stratified; classes with fewer than five rows cannot enter every fold"
    else:
        splitter = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        method = "KFold fallback: every class has fewer than five rows"
    assignments = np.full(len(y), -1, dtype=int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class.*")
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
            if np.intersect1d(train, test).size:
                raise AssertionError("Training and scoring rows overlap")
            assignments[test] = fold
    if (assignments < 0).any():
        raise AssertionError("Unassigned scoring row")
    return assignments, method


def crossfit(p: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, list]:
    after = np.empty_like(p, dtype=float)
    audit = []
    for fold in range(N_FOLDS):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if not len(train) or not len(test) or np.intersect1d(train, test).size:
            raise ValueError("Invalid cross-fitting split")
        temperature = fit_temperature(p[train], y[train])
        after[test] = temperature_scale(p[test], temperature)
        audit.append(
            {
                "fold": fold,
                "temperature": temperature,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
            }
        )
    if not np.array_equal(p.argmax(axis=1), after.argmax(axis=1)):
        raise AssertionError("Temperature scaling changed an argmax")
    return after, audit


def abstain_table(p: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        keep = p.max(axis=1) >= threshold
        rows.append(
            {
                "threshold": threshold,
                "kept": int(keep.sum()),
                "coverage": float(keep.mean()),
                "accuracy": float((p[keep].argmax(axis=1) == y[keep]).mean())
                if keep.any()
                else None,
            }
        )
    return rows


def bootstrap_ece(before: np.ndarray, after: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    samples = np.empty((N_BOOTSTRAP, 2))
    for i in range(N_BOOTSTRAP):
        ix = rng.integers(0, len(y), len(y))
        samples[i] = ece(before[ix], y[ix]), ece(after[ix], y[ix])
    return {
        "resamples": N_BOOTSTRAP,
        "seed": SEED,
        "before_95": np.quantile(samples[:, 0], [0.025, 0.975]).tolist(),
        "after_95": np.quantile(samples[:, 1], [0.025, 0.975]).tolist(),
        "after_minus_before_95": np.quantile(
            samples[:, 1] - samples[:, 0], [0.025, 0.975]
        ).tolist(),
        "note": "Paired row bootstrap conditional on fixed OOF predictions; no refitting. "
        "It omits temperature-fit and overlapping-training-fold uncertainty.",
    }


def load_d1() -> tuple[dict, np.ndarray, np.ndarray, list[str], dict]:
    """Bind the reference by E24's saved SHA, never select it by a metric."""
    reference = pd.read_csv(REFERENCE, dtype={"item_id": str})
    meta = {
        name: json.loads(path.with_suffix(".json").read_text()) for name, path in CACHES.items()
    }
    if sha256(REFERENCE) != meta["event_model"]["labels_sha256"]:
        raise ValueError("D1 reference does not match the E24 label hash")
    input_sha = sha256(D1_INPUT)
    input_ids = pd.read_csv(D1_INPUT, usecols=["item_id"], dtype=str)["item_id"].to_numpy()
    ids, classes, matrices = None, None, {}
    for name, path in CACHES.items():
        if sha256(path) != meta[name]["cache_sha256"] or meta[name]["input_sha256"] != input_sha:
            raise ValueError(f"Cache or D1 input hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as cache:
            row_ids = cache["item_id"].astype(str)
            row_classes = cache["labels"].astype(str).tolist()
            p = validate_probabilities(cache["probabilities"])
        if len(row_ids) != 153 or len(set(row_ids)) != 153 or len(row_classes) != 9:
            raise ValueError("Expected 153 unique D1 IDs and nine classes")
        if len(set(row_classes)) != 9 or p.shape != (153, 9):
            raise ValueError("Duplicate classes or unexpected probability shape")
        if not np.array_equal(row_ids, input_ids):
            raise ValueError(f"Cache IDs/order differ from D1: {name}")
        if ids is None:
            ids, classes = row_ids, row_classes
        if set(row_classes) != set(classes) or not np.array_equal(row_ids, ids):
            raise ValueError("Cache ID/class mismatch")
        matrices[name] = p[:, [row_classes.index(c) for c in classes]]
    if not np.array_equal(reference["item_id"].to_numpy(), ids):
        raise ValueError("Reference IDs/order differ from caches")
    if set(reference["label"]) - set(classes):
        raise ValueError("Reference contains a class outside the nine cached classes")
    y = np.array([classes.index(label) for label in reference["label"]])
    provenance = {
        "reference": str(REFERENCE.relative_to(ROOT)),
        "reference_sha256": sha256(REFERENCE),
        "reference_selection": "Exact match to E24 cache metadata labels_sha256 (reference_v6)",
        "input_sha256": input_sha,
        "classes": classes,
        "class_counts": reference["label"].value_counts().to_dict(),
        "caches": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "metadata_sha256": sha256(path.with_suffix(".json")),
                "stored_metadata": meta[name],
            }
            for name, path in CACHES.items()
        },
    }
    return matrices, y, ids, classes, provenance


def binary_availability() -> dict:
    """The specified snapshots have severity/hard decisions, no alert probability."""
    expected = {
        "E23": {"impact", "fired", "predicted_fire", "reference_fire"},
        "8K": {"final", "pred_C"},
    }
    result = {}
    for name, path in BINARY_FILES.items():
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if not expected[name].issubset(columns):
            raise ValueError(f"Unexpected {name} snapshot schema; inspect before scoring")
        probability_names = {
            c
            for c in columns
            if any(token in c.lower() for token in ("probab", "confidence", "logit", "score"))
        }
        if probability_names:
            raise ValueError(f"Unreviewed score columns in {name}: {sorted(probability_names)}")
        result[name] = {
            "status": "not stored",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "columns_checked": columns,
            "Platt_scaling": "skipped",
            "binary_reliability": "not stored",
            "reason": "Severity impact and hard fire/class decisions are not probabilities; "
            "no conversion, model run or binary calibration performed.",
        }
    return result


def run() -> dict:
    # Validate everything before consuming the one scored run.
    if any(p.exists() for p in (RESULT, LOCK, FOLDS_FILE)):
        raise FileExistsError("E39 already scored or locked")
    matrices, y, ids, _, provenance = load_d1()
    binary = binary_availability()
    folds, method = make_folds(y)
    claim_run(RESULT, LOCK, FOLDS_FILE)
    results = {}
    for name, before in matrices.items():
        after, audit = crossfit(before, y, folds)
        raw, calibrated = metrics(before, y), metrics(after, y)
        useful = calibrated["ECE"] < raw["ECE"] and calibrated["NLL"] <= raw["NLL"]
        results[name] = {
            "before": raw,
            "after_OOF": calibrated,
            "decision": "useful" if useful else "no gain",
            "abstain_before": abstain_table(before, y),
            "abstain_after_OOF": abstain_table(after, y),
            "reliability_before": reliability(before, y),
            "reliability_after_OOF": reliability(after, y),
            "ECE_bootstrap": bootstrap_ece(before, after, y),
            "fold_audit": audit,
            "OOF_probabilities": after.tolist(),
        }
    pd.DataFrame({"item_id": ids, "label_index": y, "score_fold": folds}).to_csv(
        FOLDS_FILE, index=False, mode="x"
    )
    out = {
        "experiment": "E39",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": "docs/DECISION_MODEL_BENCH.md#e39",
        "seed": SEED,
        "provenance": provenance,
        "fold_method": method,
        "folds": N_FOLDS,
        "fold_file_sha256": sha256(FOLDS_FILE),
        "metrics": results,
        "binary": binary,
        "implementation": {
            "ECE": "10 equal-mass top-label bins, stable confidence sort, weighted absolute gap",
            "Brier": "Mean per-row sum of squared errors across nine classes; not divided by 9",
            "temperature": "softmax(log(clip(p,1e-12,1))/T); one scalar T per training fold",
            "optimiser": "bounded scalar minimisation of training NLL over log(T)",
            "log_T_bounds": list(LOG_T_BOUNDS),
            "no_full_data_fit": True,
        },
        "limits": [
            "153 AI-labelled rows; no human check; rare classes and noisy ECE.",
            "These caches are nine-class event heads, not a learned credit-alert probability.",
            "The requested E24 cache used batch 64; student E used per-row inference. "
            "They are calibrated as stored, not regenerated or claimed to be current hybrid C.",
            "OOF estimates fit calibration only; they are not a new unseen model evaluation.",
            "Nothing is applied to the live engine.",
        ],
    }
    write_json(RESULT, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    out = run()
    print(
        json.dumps(
            {
                k: {"before": v["before"], "after_OOF": v["after_OOF"], "decision": v["decision"]}
                for k, v in out["metrics"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
