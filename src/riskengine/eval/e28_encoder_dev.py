"""E28 G/S development comparison from exported probabilities. Never reads E24 finals.

--texts eval_texts.csv --probabilities <G/S probability NPZ files>
Optional --references points to reference_v7; --selection supplies the earlier E24 D2
selection settings. No candidate or live configuration is changed by this report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.e24_dev import Config, build_predictor, engine_probs
from riskengine.eval.e24_gate import CLASSES, macro_f1
from riskengine.eval.e25_candidate import build_candidate, metrics, student_probs
from riskengine.nlp.events import LEARNED_CLASSES
from riskengine.paths import DATA, ROOT

TASK = ROOT / "private_notes/annotation_uploads/task1"
REFERENCES = TASK / "reference_v7"
SELECTION = DATA / "outputs/e24_dev_selection.json"
OUT = DATA / "outputs/e28_encoder_dev.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_export(path: Path, frame: pd.DataFrame):
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("npz_sha256") != digest(path):
        raise ValueError("probability export hash mismatch")
    if not meta.get("calibrated") or meta.get("calibration_split") != "valid":
        raise ValueError("E28 temperature must already be fitted on valid only")
    with np.load(path, allow_pickle=False) as saved:
        names, p = saved["labels"].tolist(), saved["probabilities"].copy()
        ids, sets, texts = (saved[name].tolist() for name in ("ids", "sets", "text_sha256"))
    if len(names) != len(LEARNED_CLASSES) or set(names) != set(LEARNED_CLASSES):
        raise ValueError("expected exactly the nine learned classes")
    if meta.get("class_order") != names:
        raise ValueError("sidecar and array class order disagree")
    expected = [hashlib.sha256(text.encode()).hexdigest() for text in frame["text"]]
    if ids != frame["id"].tolist() or sets != frame["set"].tolist() or texts != expected:
        raise ValueError("probability ids, sets or text/order mismatch")
    if p.shape != (len(frame), len(names)) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid probability matrix")
    if not np.allclose(p.sum(1), 1):
        raise ValueError("probabilities must sum to one")
    return p[:, [names.index(label) for label in LEARNED_CLASSES]], meta


def cached_probs(texts, matrix):
    index = {}
    for text, row in zip(texts, matrix, strict=True):
        if text in index and not np.allclose(index[text], row):
            raise ValueError("duplicate text has inconsistent cached probabilities")
        index[text] = row

    def get(items):
        return list(LEARNED_CLASSES), np.array([index[text] for text in items]).reshape(
            len(items), len(LEARNED_CLASSES)
        )

    return get


def bootstrap(y, a, b, resamples=1000, seed=0):
    y, a, b = map(np.asarray, (y, a, b))
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(resamples):
        positions = rng.integers(0, len(y), len(y))
        gains.append(macro_f1(y[positions], b[positions]) - macro_f1(y[positions], a[positions]))
    return {
        "gain": macro_f1(y, b) - macro_f1(y, a),
        "ci95": np.percentile(gains, [2.5, 97.5]).tolist(),
        "seed": seed,
        "resamples": resamples,
    }


def d2_decision(existing, alternatives, support):
    """Same class guard, 0.02 entry margin, and 0.01 simpler-candidate tie rule."""
    big = [label for label, count in support.items() if count >= 10]
    base = existing["A"]["per_class_f1"]

    def allowed(row):
        return all(row["per_class_f1"][label] >= base[label] - 0.05 - 1e-12 for label in big)

    eligible = [name for name, row in existing.items() if allowed(row)]
    best = max(existing[name]["macro_f1"] for name in eligible)
    old_choice = next(
        name for name in eligible if existing[name]["macro_f1"] >= best - 0.01 - 1e-12
    )
    admitted = [
        name
        for name, row in alternatives.items()
        if allowed(row) and row["macro_f1"] >= best + 0.02 - 1e-12
    ]
    chosen = old_choice
    if admitted:
        peak = max(alternatives[name]["macro_f1"] for name in admitted)
        chosen = next(
            name for name in admitted if alternatives[name]["macro_f1"] >= peak - 0.01 - 1e-12
        )
    return {
        "chosen_development_only": chosen,
        "best_existing_macro_f1": best,
        "minimum_new_gain": 0.02,
        "simpler_tie_margin": 0.01,
        "new_candidates_admitted": admitted,
        "large_classes": big,
        "fresh_test_required": True,
        "production_changed": False,
    }


def read_reference(frame, split, root):
    path = (
        root / "development" / f"{split}_labels.csv"
        if split != "R3"
        else TASK / "retest_set/R3_labels_claude.csv"
    )
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    id_column = "item_id" if "item_id" in labels else "id"
    if labels[id_column].duplicated().any() or not labels["label"].isin(CLASSES).all():
        raise ValueError("invalid development reference")
    if set(labels[id_column]) != set(frame["id"]):
        raise ValueError("reference ids do not match the requested split")
    return labels.set_index(id_column).loc[frame["id"], "label"].tolist(), digest(path)


def evaluate(
    text_path, probability_paths, references=REFERENCES, selection_path=SELECTION, out=OUT
):
    if out.exists():
        raise FileExistsError("preserve the previous E28 report")
    if "final" in str(text_path).lower():
        raise ValueError("E24 final files are forbidden")
    frame = pd.read_csv(text_path, dtype=str, keep_default_na=False)
    if set(frame) != {"id", "text", "set"} or set(frame["set"]) != {"D1", "D2", "R3"}:
        raise ValueError("need blind D1/D2/R3 texts only")
    if frame.duplicated(["set", "id"]).any():
        raise ValueError("duplicate evaluation ids")
    exports = []
    for path in probability_paths:
        p, meta = load_export(path, frame)
        if meta.get("variant") not in {"G", "S"}:
            raise ValueError("expected G or S variant")
        name = meta["variant"] + "_" + meta["encoder"].split("/")[-1]
        exports.append((name, p, meta))
    if len({name for name, _, _ in exports}) != len(exports):
        raise ValueError("duplicate encoder variant")
    # Fixed simpler-first order; never rank encoders by D1 or R3.
    complexity = [
        "G_bge-small-en-v1.5",
        "S_bge-small-en-v1.5",
        "G_e5-base-v2",
        "S_e5-base-v2",
        "G_multilingual-e5-large",
    ]
    exports.sort(key=lambda row: complexity.index(row[0]))
    previous = json.loads(selection_path.read_text())["selection"]
    temperature = json.loads(selection_path.read_text())["temperature"]["temperature"]
    base, student = engine_probs(), student_probs()
    results, d2_existing, d2_new = {}, {}, {}
    for split in ("D1", "D2", "R3"):
        mask = frame["set"].eq(split).to_numpy()
        part = frame.loc[mask]
        y, ref_sha = read_reference(part, split, references)
        texts = part["text"].tolist()
        names, p = base(texts)

        def base_cache(items, names=names, p=p, texts=texts):
            if list(items) != texts:
                raise ValueError("baseline cache requested with different texts/order")
            return names, p

        c, _ = build_predictor(Config("C", temperature=temperature), base_cache)(texts)
        rows = {"C": metrics(y, c)}
        for name, matrix, _meta in exports:
            pred, _ = build_candidate(cached_probs(texts, matrix[mask]))(texts)
            rows[name] = metrics(y, pred) | {"paired_vs_C": bootstrap(y, c, pred)}
            if split == "D2":
                d2_new[name] = {
                    "macro_f1": rows[name]["macro_f1_12_primary"],
                    "per_class_f1": {
                        label: row["f1"] for label, row in rows[name]["per_class"].items()
                    },
                }
        if split == "D2":
            for key in "ABCD":
                cfg = Config(key, previous["threshold_for_D"] if key == "D" else None, temperature)
                pred, _ = build_predictor(cfg, base_cache)(texts)
                score = metrics(y, pred)
                d2_existing[key] = {
                    "macro_f1": score["macro_f1_12_primary"],
                    "per_class_f1": {label: row["f1"] for label, row in score["per_class"].items()},
                }
            pred, _ = build_candidate(student)(texts)
            score = metrics(y, pred)
            d2_existing["E"] = {
                "macro_f1": score["macro_f1_12_primary"],
                "per_class_f1": {label: row["f1"] for label, row in score["per_class"].items()},
            }
            d2_support = {label: y.count(label) for label in CLASSES}
        results[split] = {"rows": len(y), "reference_sha256": ref_sha, "scores": rows}
    record = {
        "scope": "Exploratory D1/D2/R3 development; AI-derived references; fresh test required.",
        "sets": results,
        "d2_rule": d2_decision(d2_existing, d2_new, d2_support),
        "d2_existing_comparators": d2_existing,
        "input_sha256": digest(text_path),
        "export_metadata": {name: meta for name, _, meta in exports},
        "rules": "Unchanged E13 credit rules and C rule layer via e25_candidate.build_candidate",
        "temperature": "valid-only calibration already applied; no D1/D2 re-fit",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, nargs="+", required=True)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    record = evaluate(args.texts, args.probabilities, args.references, args.selection, args.out)
    print(json.dumps(record["d2_rule"], indent=2))


if __name__ == "__main__":
    main()
