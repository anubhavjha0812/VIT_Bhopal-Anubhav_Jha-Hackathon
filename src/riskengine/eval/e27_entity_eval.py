"""E27: one registered test of headline versus entity-aware sentiment.

python -m riskengine.eval.e27_entity_eval --dry-run
python -m riskengine.eval.e27_entity_eval --score [--summary <training summary.json>]

Dry-run prints split counts only. Scoring uses local int8 exports, per-row E24 inference,
and E18/E20's config-label argmax mapping. It reports the fixed rule, never changes the
engine, and refuses an existing report (including an interrupted scoring attempt).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.e20 import LABELS, macro_f1, paired
from riskengine.eval.e24_inference import EvaluationProbs
from riskengine.nlp.sentiment import OWN_SENTIMENT
from riskengine.paths import DATA, ROOT

SPLITS = DATA / "cache/e27"
EXPORT = ROOT / "private_notes/annotation_uploads/task1/e27_entity_output"
MODEL = EXPORT / "models/e27-entity-bge-small-en-v1.5-int8"
OUT = DATA / "outputs/e27_entity_eval.json"
SEEDS = [42, 43, 44]
BOOTSTRAP_SEED, RESAMPLES = 0, 1000
MIN_GAIN, MAX_DROP = 0.05, 0.02
REQUIRED = {"sno", "headline", "entity", "label", "conflict", "n_entities"}


class _PairTokenizer:
    """Adapt E24's single-row tokenization to the training (entity, headline) pair."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, pairs, **kwargs):
        return self.tokenizer(
            [entity for entity, _ in pairs], [headline for _, headline in pairs], **kwargs
        )


class EntityEvaluationProbs:
    """Reuse E24's exact session, padding, softmax and one-row execution loop."""

    def __init__(self, model_dir: Path):
        self.probs = EvaluationProbs(model_dir)
        self.probs.tokenizer = _PairTokenizer(self.probs.tokenizer)
        self.inference_metadata = self.probs.inference_metadata | {
            "input_order": ["entity", "headline"],
            "encoding": "tokenizer(entity, text_pair=headline)",
        }

    def __call__(self, pairs):
        return self.probs(pairs)


def _dev_best(run: dict) -> tuple[int, float]:
    """Validate the exported epoch using dev macro-F1 only, including first-on-tie."""
    if run.get("seed") not in SEEDS or not run.get("history"):
        raise ValueError("missing E27 seed or dev history")
    history = run["history"]
    for epoch in history:
        value = epoch.get("dev_macro_f1")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid dev_macro_f1 in training record")
        if epoch.get("epoch") not in range(1, 5):
            raise ValueError("invalid epoch in training record")
    best = max(history, key=lambda row: row["dev_macro_f1"])
    key = (best["epoch"], best["dev_macro_f1"])
    recorded = run.get("best", {})
    if (recorded.get("epoch"), recorded.get("dev_macro_f1")) != key:
        raise ValueError("exported epoch is not the dev macro-F1 winner")
    return key


def select_seed(runs: list[dict]) -> dict:
    """Recheck the training selection; test metrics never enter this choice."""
    if [run.get("seed") for run in runs] != SEEDS:
        raise ValueError("expected training runs in seed order 42, 43, 44")
    for run in runs:
        _dev_best(run)
    return max(runs, key=lambda run: run["best"]["dev_macro_f1"])


def load_selection(model_dir: Path, summary_path: Path | None = None) -> dict:
    source_path = model_dir / "source_metrics.json"
    raw = source_path.read_bytes()
    source = json.loads(raw)
    epoch, score = _dev_best(source)
    info = {
        "seed": source["seed"],
        "epoch": epoch,
        "dev_macro_f1": score,
        "criterion": "dev_macro_f1 only; first epoch/seed on a tie",
        "source_metrics_file": str(source_path),
        "source_metrics_sha256": hashlib.sha256(raw).hexdigest(),
        "all_seed_comparison_verified": False,
        "note": "Exported winner only; other seeds cannot be rechecked without summary.json.",
    }
    if summary_path is not None:
        raw = summary_path.read_bytes()
        summary = json.loads(raw)
        winner = select_seed(summary["runs"])
        for record in (source, summary["chosen"]):
            if record["seed"] != winner["seed"] or _dev_best(record) != _dev_best(winner):
                raise ValueError("export/summary does not match the dev-only seed winner")
        info.update(
            {
                "all_seed_comparison_verified": True,
                "summary_file": str(summary_path),
                "summary_sha256": hashlib.sha256(raw).hexdigest(),
                "note": "All three training seeds rechecked using dev metrics only.",
            }
        )
    return info


def read_split(path: Path) -> tuple[pd.DataFrame, str]:
    """Read the physical CSV once, using those same bytes for its provenance hash."""
    raw = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), dtype=str)
    if not REQUIRED.issubset(frame.columns) or frame.empty:
        raise ValueError(f"missing E27 rows or columns: {path}")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError(f"missing E27 values: {path}")
    if not frame["label"].isin(LABELS).all():
        raise ValueError(f"invalid sentiment label: {path}")
    if any(frame[column].str.strip().eq("").any() for column in ("sno", "headline", "entity")):
        raise ValueError(f"empty E27 text or id: {path}")
    conflict = (
        frame["conflict"].str.lower().map({"true": True, "false": False, "1": True, "0": False})
    )
    entities = pd.to_numeric(frame["n_entities"], errors="coerce")
    if conflict.isna().any() or entities.isna().any() or (entities < 1).any():
        raise ValueError(f"invalid conflict/n_entities: {path}")
    if (entities % 1 != 0).any() or (conflict & (entities < 2)).any():
        raise ValueError(f"invalid entity count for conflicting headline: {path}")
    frame["conflict"], frame["n_entities"] = conflict.astype(bool), entities.astype(int)
    return frame, hashlib.sha256(raw).hexdigest()


def counts(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "headlines": int(frame["headline"].nunique()),
        "conflicting_rows": int(frame["conflict"].sum()),
        "conflicting_headlines": int(frame.loc[frame["conflict"], "headline"].nunique()),
        "labels": {label: int(frame["label"].eq(label).sum()) for label in LABELS},
    }


def _predict(probs, inputs) -> list[str]:
    names, matrix = probs(inputs)
    names = [name.lower() for name in names]
    matrix = np.asarray(matrix)
    if len(names) != 3 or set(names) != set(LABELS):
        raise ValueError("expected negative/neutral/positive config labels")
    if matrix.shape != (len(inputs), 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid sentiment probabilities")
    return [names[index] for index in matrix.argmax(axis=1)]


def decide(primary_a: float | None, primary_b: float | None, all_a: float, all_b: float) -> dict:
    gain = None if primary_a is None or primary_b is None else primary_b - primary_a
    secondary = all_b - all_a
    gain_ok = gain is not None and (
        gain >= MIN_GAIN or math.isclose(gain, MIN_GAIN, rel_tol=0, abs_tol=1e-12)
    )
    guard_ok = secondary >= -MAX_DROP or math.isclose(
        secondary, -MAX_DROP, rel_tol=0, abs_tol=1e-12
    )
    switch = gain_ok and guard_ok
    return {
        "primary_gain_b_minus_a": gain,
        "minimum_primary_gain": MIN_GAIN,
        "primary_gain_pass": gain_ok,
        "all_macro_f1_b_minus_a": secondary,
        "maximum_all_macro_f1_drop": MAX_DROP,
        "all_macro_f1_guard_pass": guard_ok,
        "switch_to_b": switch,
        "recommended_engine": "B" if switch else "A",
        "applied_to_engine": False,
    }


def _comparison(y, a, b) -> dict:
    if not len(y):
        return {"rows": 0, "A": None, "B": None, "paired_b_minus_a": None}
    return {
        "rows": len(y),
        "A": {"macro_f1": macro_f1(y, a), "accuracy": float(np.mean(y == a))},
        "B": {"macro_f1": macro_f1(y, b), "accuracy": float(np.mean(y == b))},
        "paired_b_minus_a": paired(y, a, b, n=RESAMPLES, seed=BOOTSTRAP_SEED),
    }


def score_frame(frame: pd.DataFrame, headline_probs, entity_probs) -> dict:
    headlines = frame["headline"].drop_duplicates().tolist()
    by_headline = dict(zip(headlines, _predict(headline_probs, headlines), strict=True))
    a = frame["headline"].map(by_headline).to_numpy()
    pairs = list(zip(frame["entity"], frame["headline"], strict=True))
    b, y = np.asarray(_predict(entity_probs, pairs)), frame["label"].to_numpy()
    mask = frame["conflict"].to_numpy()
    primary, all_rows = _comparison(y[mask], a[mask], b[mask]), _comparison(y, a, b)
    return {
        "counts": counts(frame),
        "labels": LABELS,
        "metrics": {"conflicting": primary, "all": all_rows},
        "paired_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": RESAMPLES,
            "interval": "percentile 95%",
            "sampling_unit": "entity row",
        },
        "rule": decide(
            primary["A"]["macro_f1"] if primary["A"] else None,
            primary["B"]["macro_f1"] if primary["B"] else None,
            all_rows["A"]["macro_f1"],
            all_rows["B"]["macro_f1"],
        ),
        "scope": "SEntFiN Indian business news headlines; US news and tweets are not tested.",
        "label_provenance": "Human-made labels supplied by the SEntFiN dataset authors.",
    }


def score(data_dir: Path, model_dir: Path, out: Path, summary_path: Path | None = None) -> dict:
    if out.exists():
        raise FileExistsError(f"E27 scoring already attempted; preserve the report: {out}")
    selection = load_selection(model_dir, summary_path)
    a, b = EvaluationProbs(OWN_SENTIMENT), EntityEvaluationProbs(model_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the sole report before reading/scoring the test. Failed attempts remain visible.
    with out.open("x", encoding="utf-8") as handle:
        record = {"status": "started", "started_utc": datetime.now(UTC).isoformat()}
        json.dump(record, handle, indent=2)
        handle.flush()
        try:
            frame, digest = read_split(data_dir / "test.csv")
            record.update(score_frame(frame, a, b))
            record.update(
                {
                    "status": "complete",
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "test_file": str(data_dir / "test.csv"),
                    "test_sha256": digest,
                    "selection": selection,
                    "models": {
                        "A": {"folder": str(OWN_SENTIMENT), "inference": a.inference_metadata},
                        "B": {"folder": str(model_dir), "inference": b.inference_metadata},
                    },
                }
            )
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})
            handle.seek(0)
            handle.truncate()
            json.dump(record, handle, indent=2)
            handle.write("\n")
            raise
        handle.seek(0)
        handle.truncate()
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="counts only; never load a model")
    mode.add_argument("--score", action="store_true", help="score the test once; no engine changes")
    parser.add_argument("--data-dir", type=Path, default=SPLITS)
    parser.add_argument("--model", type=Path, default=MODEL, help="exported entity int8 folder")
    parser.add_argument(
        "--summary", type=Path, help="optional training summary for all-seed recheck"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.dry_run:
        report = {
            split: counts(read_split(args.data_dir / f"{split}.csv")[0])
            for split in ("train", "dev", "test")
        }
    else:
        report = score(args.data_dir, args.model, args.out, args.summary)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
