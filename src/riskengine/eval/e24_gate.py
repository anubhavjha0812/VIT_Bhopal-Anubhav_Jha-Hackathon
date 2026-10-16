"""E24 evaluation gate: frozen-protocol checks and the one-time final evaluation (DECISIONS E24).

Fails closed: every check raises `GateError`, and nothing reads the final labels until the
frozen sample, the frozen candidate, the annotation records and the label files all check out.

Development code reads labels only through `load_dev_labels`, which refuses final-set ids, so
final labels cannot reach candidate selection or threshold tuning through this module.

Limitation, stated plainly: the completed-evaluation marker is a local file. Anyone with write
access can delete it (and its read-only backup, after `chmod`) and run again; this code cannot
prevent that. The safeguards make a rerun visible rather than impossible: the marker is written
in two places (outputs and the read-only backup folder), it carries the sha256 of the results, the
run is logged in STEPS.md, and committing the marker right after the run puts it in git history.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from riskengine.paths import DATA, ROOT

CLASSES = [
    "Default", "Credit downgrade", "Covenant breach", "Geopolitical", "Macroeconomic", "M&A",
    "Product news", "Earnings", "Analyst rating", "Legal/Regulatory", "Management change", "Other",
]  # fmt: skip
LABEL_SOURCES = {"ai_agreed", "ai_agreed_human_audited", "human_adjudicated", "excluded"}
LABEL_COLUMNS = ["item_id", "label", "ambiguous", "rationale", "label_source"]
CANDIDATES = {"A", "B", "C", "D"}
GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 ... 0.70, fixed in E24
E24_SECTION_SHA = "bfffa0ad37366c8a"  # first 16 hex of the locked DECISIONS E24 section
BOOT, BOOT_SEED, MAX_WIDTH, TWEET_MARGIN = 1000, 0, 0.20, 0.05
FINAL_SETS = {"final_news": "final_news_to_label.csv", "final_tweets": "final_tweets_to_label.csv"}
# Code and model the predictors are built from; hashed at freeze, re-checked before scoring.
PINNED = [
    "src/riskengine/eval/e24_dev.py",
    "src/riskengine/eval/e24_rules.py",
    "src/riskengine/eval/e24_inference.py",
    "src/riskengine/nlp/events.py",
    "src/riskengine/nlp/classifier.py",
    "models/topic/model.onnx",
]
LOW_SUPPORT = 5
RECORD_FIELDS = [
    "labeller", "set", "model_as_shown_by_interface", "interface", "annotation_date_utc",
    "settings", "prompt_file", "prompt_sha256", "guide_sha256", "input_file_sha256",
    "raw_output_file", "raw_output_sha256",
]  # fmt: skip


class GateError(RuntimeError):
    """A frozen-protocol check failed; nothing is evaluated."""


@dataclass
class Paths:
    gold: Path = DATA / "gold" / "e24"
    outputs: Path = DATA / "outputs"
    backup: Path = ROOT / "private_notes" / "e24_frozen"
    decisions: Path = ROOT / "docs" / "DECISIONS.md"
    e3_split_name: str = "e3_split.csv"
    section_sha: str = E24_SECTION_SHA
    root: Path = ROOT
    extra: dict = field(default_factory=dict)

    @property
    def manifest(self) -> Path:
        return self.outputs / "e24_final_sample.json"

    @property
    def frozen(self) -> Path:
        return self.outputs / "e24_frozen.json"

    @property
    def marker(self) -> Path:
        return self.outputs / "e24_final_done.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _need(path: Path, what: str) -> Path:
    if not Path(path).exists():
        raise GateError(f"missing {what}: {path}")
    return Path(path)


def section_sha(decisions: Path) -> str:
    lines, inside = [], False
    for line in _need(decisions, "DECISIONS").read_text().splitlines(keepends=True):
        if line.startswith("## E24."):
            inside = True
        elif inside and line.startswith("## "):
            break
        if inside:
            lines.append(line)
    if not lines:
        raise GateError("DECISIONS has no E24 section")
    return hashlib.sha256("".join(lines).encode()).hexdigest()[:16]


# --- frozen sample ------------------------------------------------------------------------


def check_frozen(p: Paths) -> dict:
    """The E24 entry, the frozen files, the manifest and the read-only backups all agree."""
    if section_sha(p.decisions) != p.section_sha:
        raise GateError("DECISIONS E24 section changed since it was locked")
    manifest = json.loads(_need(p.manifest, "frozen-sample manifest").read_text())
    files = manifest.get("files_sha256") or {}
    if not files:
        raise GateError("manifest lists no files")
    for name, want in files.items():
        if sha256(_need(p.gold / name, f"frozen file {name}")) != want:
            raise GateError(f"hash mismatch: {name}")
        if sha256(_need(p.backup / name, f"backup of {name}")) != want:
            raise GateError(f"backup hash mismatch: {name}")
    listed = _need(p.backup / "MANIFEST.sha256", "backup manifest").read_text().split("\n")
    for line in filter(None, listed):
        want, name = line.split()
        if sha256(_need(p.backup / name, f"backup {name}")) != want:
            raise GateError(f"backup manifest mismatch: {name}")
    return manifest


# --- labels -------------------------------------------------------------------------------


def validate_labels(df: pd.DataFrame, expected_ids: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise GateError(f"label file lacks columns {missing}")
    df = df.astype({c: str for c in LABEL_COLUMNS}).copy()
    if df["item_id"].duplicated().any():
        raise GateError("duplicate item_id in label file")
    if set(df["item_id"]) != set(map(str, expected_ids)):
        raise GateError("label ids do not match the frozen item ids")
    bad = ~df["label_source"].isin(LABEL_SOURCES)
    if bad.any():
        raise GateError(f"{bad.sum()} rows with an unknown label_source")
    kept = df[df["label_source"] != "excluded"]
    if (~kept["label"].isin(CLASSES)).any():
        raise GateError("label outside the 12 frozen classes")
    if (~kept["ambiguous"].isin({"Y", "N"})).any():
        raise GateError("ambiguous must be Y or N")
    if (kept["rationale"].str.strip().isin({"", "nan"})).any():
        raise GateError("empty rationale")
    return df


def final_ids(p: Paths) -> set[str]:
    ids: set[str] = set()
    for name in FINAL_SETS.values():
        ids |= set(pd.read_csv(_need(p.gold / name, name), dtype=str)["item_id"])
    return ids


def load_dev_labels(path: Path, split: str, p: Paths) -> pd.DataFrame:
    """D1 or D2 labels for development. Refuses anything that touches the final set."""
    if split not in {"D1", "D2"}:
        raise GateError("development uses D1 or D2 only")
    if "final" in Path(path).name:
        raise GateError("final labels cannot be used for development")
    df = pd.read_csv(_need(path, "dev label file"), dtype=str, keep_default_na=False)
    if "item_id" in df and set(df["item_id"]) & final_ids(p):
        raise GateError("final-set items found in a development label file")
    split_df = pd.read_csv(_need(p.gold / p.e3_split_name, "E3 split"), dtype=str)
    want = split_df.loc[split_df["split"] == split, "doc_id"].tolist()
    return validate_labels(df, want)


def parse_utc(value) -> datetime:
    """An ISO 8601 timestamp with a time and an explicit UTC offset of zero, e.g.
    2026-10-12T09:30:00Z. Dates alone, local times and other offsets are rejected."""
    if not isinstance(value, str) or "T" not in value:
        raise GateError(f"not a UTC timestamp with a time: {value!r}")
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise GateError(f"malformed timestamp: {value!r}") from e
    if ts.tzinfo is None or ts.utcoffset().total_seconds() != 0:
        raise GateError(f"timestamp must be in UTC: {value!r}")
    return ts


def final_annotation_files(p: Paths) -> list[Path]:
    """Anything that would mean the final set has been labelled (records, raw outputs, labels)."""
    found = [p.gold / f"{s}_labels.csv" for s in FINAL_SETS]
    found += list((p.gold / "records").glob("*final*")) if (p.gold / "records").exists() else []
    found += list((p.gold / "raw").glob("*final*")) if (p.gold / "raw").exists() else []
    return [f for f in found if f.exists()]


def check_record(record_path: Path, p: Paths, labeller: str, set_name: str) -> dict:
    """An annotation record: every field filled from what happened, hashes that match, a UTC
    annotation time; for the final sets that time must not be before the candidate freeze."""
    rec = json.loads(_need(record_path, "annotation record").read_text())
    empty = [f for f in RECORD_FIELDS if rec.get(f) in (None, "")]
    if empty:
        raise GateError(f"annotation record lacks {empty}")
    if rec["labeller"] != labeller or rec["set"] != set_name:
        raise GateError("annotation record is for another labeller or set")
    checks = {
        "prompt_sha256": p.root / rec["prompt_file"],
        "guide_sha256": p.gold / "GUIDE.md",
        "raw_output_sha256": p.root / rec["raw_output_file"],
    }
    if set_name in FINAL_SETS:
        checks["input_file_sha256"] = p.gold / FINAL_SETS[set_name]
    for key, path in checks.items():
        if sha256(_need(path, key)) != rec[key]:
            raise GateError(f"annotation record {key} does not match {path.name}")
    when = parse_utc(rec["annotation_date_utc"])
    if set_name in FINAL_SETS:
        frozen = json.loads(_need(p.frozen, "frozen candidate (e24_frozen.json)").read_text())
        if when < parse_utc(frozen.get("frozen_at_utc")):
            raise GateError("final set was annotated before the candidate was frozen")
    return rec


# --- frozen candidate ---------------------------------------------------------------------


def freeze_candidate(
    p: Paths, candidate: str, threshold: float | None, temperature: float | None,
    rules_file: Path | None,
) -> dict:  # fmt: skip
    """Write e24_frozen.json once, after selection on D2 and before the final labels exist."""
    if p.frozen.exists():
        raise GateError("a candidate is already frozen")
    if candidate not in CANDIDATES:
        raise GateError(f"unknown candidate {candidate!r}")
    if candidate == "D" and threshold not in GRID:
        raise GateError("candidate D needs a threshold from the fixed grid")
    if candidate != "D" and threshold is not None:
        raise GateError("only candidate D has a threshold")
    if temperature is not None and not temperature > 0:
        raise GateError("temperature must be positive")
    if candidate != "A" and rules_file is None:
        raise GateError("candidates B, C, D need their rules file")
    early = final_annotation_files(p)
    if early:
        names = ", ".join(f.name for f in early)
        raise GateError(f"final annotation already exists ({names}): freeze must come first")
    manifest = check_frozen(p)
    # Synthetic test fixtures have no model; real freezes always pin the evaluation method.
    from riskengine.eval.e24_inference import inference_metadata

    model_dir = p.root / "models/topic"
    inference = inference_metadata(model_dir) if (model_dir / "model.onnx").exists() else None
    frozen = {
        "rule": "docs/DECISIONS.md E24",
        "frozen_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate": candidate,
        "threshold": threshold,
        "temperature": temperature,
        "inference": inference,
        "rules_file": str(rules_file) if rules_file else None,
        "rules_sha256": sha256(rules_file) if rules_file else None,
        "gate_sha256": sha256(Path(__file__)),
        "pinned_sha256": {f: sha256(p.root / f) for f in PINNED if (p.root / f).exists()},
        "guide_sha256": sha256(p.gold / "GUIDE.md"),
        "sample_files_sha256": manifest["files_sha256"],
    }
    p.frozen.write_text(json.dumps(frozen, indent=2))
    _read_only_copy(p.frozen, p.backup)
    return frozen


def check_candidate(p: Paths) -> dict:
    frozen = json.loads(_need(p.frozen, "frozen candidate (e24_frozen.json)").read_text())
    backup = p.backup / p.frozen.name
    if sha256(_need(backup, "backup of e24_frozen.json")) != sha256(p.frozen):
        raise GateError("e24_frozen.json differs from its read-only backup")
    if frozen.get("candidate") not in CANDIDATES:
        raise GateError("frozen candidate is invalid")
    parse_utc(frozen.get("frozen_at_utc"))
    if (frozen["candidate"] == "D") != (frozen.get("threshold") in GRID):
        raise GateError("frozen threshold is inconsistent with the candidate")
    if frozen.get("rules_file") and sha256(Path(frozen["rules_file"])) != frozen["rules_sha256"]:
        raise GateError("rules file changed since the candidate was frozen")
    if frozen.get("guide_sha256") != sha256(p.gold / "GUIDE.md"):
        raise GateError("GUIDE.md changed since the candidate was frozen")
    for f, want in (frozen.get("pinned_sha256") or {}).items():
        if sha256(_need(p.root / f, f"pinned file {f}")) != want:
            raise GateError(f"{f} changed since the candidate was frozen")
    from riskengine.eval.e24_inference import inference_metadata, validate_inference_metadata

    model_dir = p.root / "models/topic"
    if (model_dir / "model.onnx").exists():
        try:
            validate_inference_metadata(frozen.get("inference"), inference_metadata(model_dir))
        except ValueError as exc:
            raise GateError(str(exc)) from exc
    return frozen


# --- metrics (fixed 12 classes in every resample) -----------------------------------------


def macro_f1(y: Sequence[str], pred: Sequence[str]) -> float:
    return float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0))


def paired_bootstrap(y, base, chosen, n: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """Δ = macro-F1(chosen) − macro-F1(base) on the same resampled rows; every resample uses
    the fixed 12-class list, even when a class is absent from it."""
    y, base, chosen = map(np.asarray, (y, base, chosen))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = [macro_f1(y[i], chosen[i]) - macro_f1(y[i], base[i]) for i in idx]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": macro_f1(y, chosen) - macro_f1(y, base), "ci95": [float(lo), float(hi)]}


def verdict(ci: Sequence[float], bound: float) -> str:
    """pass if the lower bound > bound, fail if the upper bound < bound, else inconclusive
    (also when the interval is wider than 0.20). News: bound 0; tweets: bound -0.05."""
    lo, hi = ci
    if hi - lo > MAX_WIDTH:
        return "inconclusive"
    if lo > bound:
        return "pass"
    if hi < bound:
        return "fail"
    return "inconclusive"


# --- one-time final evaluation ------------------------------------------------------------


def _read_only_copy(src: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / src.name
    if dst.exists():
        dst.chmod(stat.S_IRUSR | stat.S_IWUSR)
    dst.write_bytes(src.read_bytes())
    dst.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


Predictor = Callable[[list[str]], tuple[list[str], list[str]]]
Factory = Callable[[dict], Predictor]


def _production_factory(cfg: dict) -> Predictor:
    """Predictors are built only from a frozen configuration (or candidate A)."""
    from riskengine.eval.e24_dev import Config, build_predictor

    return build_predictor(Config(cfg["candidate"], cfg.get("threshold"), cfg.get("temperature")))


def run_final(p: Paths) -> dict:
    """The one-time final evaluation. Takes no predictor: A and the chosen candidate are built
    from the frozen configuration, after every hash is re-checked."""
    return _run_final(p, _production_factory)


def _per_class(y, pred) -> dict:
    from sklearn.metrics import precision_recall_fscore_support

    pr, rc, f, n = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {c: {"precision": float(a), "recall": float(b), "f1": float(x), "support": int(k)}
            for c, a, b, x, k in zip(CLASSES, pr, rc, f, n, strict=True)}  # fmt: skip


def _detail(y, pred, sources) -> dict:
    from sklearn.metrics import confusion_matrix

    src = pd.Series(sources)
    rule = src.str.endswith("_rule")
    return {
        "macro_f1": macro_f1(y, pred),
        "per_class": _per_class(y, pred),
        "confusion": {"labels": CLASSES,
                      "matrix": confusion_matrix(y, pred, labels=CLASSES).tolist()},
        "rule_coverage": {k: float((src == k).mean()) for k in sorted(set(src[rule]))},
        "model_coverage": float(src.isin(["model", "model_product_to_other"]).mean()),
        "abstention_rate": float((src == "abstain").mean()),
        "sources": src.value_counts().to_dict(),
    }  # fmt: skip


def _run_final(p: Paths, factory: Factory) -> dict:
    """Private: tests pass a fake factory; `run_final` always passes the production one."""
    if p.marker.exists() or (p.backup / p.marker.name).exists():
        raise GateError("E24 final evaluation already ran (completed marker present)")
    check_frozen(p)
    frozen = check_candidate(p)
    for set_name in FINAL_SETS:
        for labeller in ("chatgpt", "gemini"):
            check_record(p.gold / "records" / f"{labeller}_{set_name}.json", p, labeller, set_name)
    results = {
        "rule": "docs/DECISIONS.md E24",
        "candidate": frozen["candidate"],
        "threshold": frozen.get("threshold"),
        "temperature": frozen.get("temperature"),
    }
    for set_name, blind_name in FINAL_SETS.items():
        blind = pd.read_csv(p.gold / blind_name, dtype=str)
        lab = validate_labels(
            pd.read_csv(_need(p.gold / f"{set_name}_labels.csv", f"{set_name} labels"),
                        dtype=str, keep_default_na=False),
            blind["item_id"],
        )  # fmt: skip
        df = blind.merge(lab, on="item_id")
        df = df[df["label_source"] != "excluded"].reset_index(drop=True)
        texts, y = df["text"].tolist(), df["label"].tolist()
        check_frozen(p)  # re-checked immediately before prediction and scoring
        frozen_now = check_candidate(p)
        if frozen_now != frozen:
            raise GateError("frozen candidate changed during the evaluation")
        a, a_src = factory({"candidate": "A"})(texts)
        c, c_src = factory(frozen_now)(texts)
        boot = paired_bootstrap(y, a, c)
        bound = 0.0 if set_name == "final_news" else -TWEET_MARGIN
        touched = df["label_source"].isin(["human_adjudicated", "ai_agreed_human_audited"])
        sup = pd.Series(y).value_counts().reindex(CLASSES, fill_value=0)
        idx = touched.to_numpy()
        results[set_name] = {
            "n": len(df),
            "excluded": int((lab["label_source"] == "excluded").sum()),
            "ambiguous": int((df["ambiguous"] == "Y").sum()),
            "label_sources": lab["label_source"].value_counts().to_dict(),
            "support": sup.to_dict(),
            "absent_classes": [k for k, v in sup.items() if v == 0],
            "low_support_classes": [k for k, v in sup.items() if 0 < v < LOW_SUPPORT],
            "A": _detail(y, a, a_src),
            "chosen": _detail(y, c, c_src),
            "paired": boot,
            "verdict": verdict(boot["ci95"], bound),
            "secondary_human_touched": {
                "note": "not in the DECISIONS E24 entry; from approved draft v4; decides nothing",
                "n": int(idx.sum()),
                "macro_f1_A": macro_f1(np.array(y)[idx], np.array(a)[idx]) if idx.any() else None,
                "macro_f1_chosen": macro_f1(np.array(y)[idx], np.array(c)[idx])
                if idx.any()
                else None,
            },  # fmt: skip
        }
    news, tweets = results["final_news"]["verdict"], results["final_tweets"]["verdict"]
    results["switch_engine"] = news == "pass" and tweets == "pass"
    out = p.outputs / "e24_final_results.json"
    out.write_text(json.dumps(results, indent=2))
    marker = {"done": True, "results_sha256": sha256(out), "frozen_sha256": sha256(p.frozen)}
    p.marker.write_text(json.dumps(marker, indent=2))
    for f in (out, p.marker):
        _read_only_copy(f, p.backup)
    return results
