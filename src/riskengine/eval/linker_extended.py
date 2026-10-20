"""E38 offline blind sample and one-shot linking check; never runs a learned model.

--label exports a blind sample; fill gold_entities/rationale from text alone.
--freeze seals completed labels and candidate hashes before --run. No network.
"""

import argparse
import hashlib
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskengine.paths import DATA, ROOT

WORK = ROOT / "private_notes/annotation_uploads/task1/linker_extended"
RESULT = DATA / "outputs/linker_extended.json"
RULES = ROOT / "docs/experiments/LINKER_EXTENDED.md"
CATALOG = DATA / "reference/universe_extended.csv"
PROBES = DATA / "reference/linker_extended_probes.csv"
E23 = DATA / "gold/credit/credit_to_label.csv"
SOURCES = [
    DATA / "cache/history_texts.csv.gz",
    DATA / "cache/headline_scores_fresh.csv.gz",
    DATA / "raw/fnspid/universe_news_named.csv.gz",
    DATA / "raw/fnspid/fresh_news.csv.gz",
]
CANDIDATE_FILES = [
    CATALOG,
    PROBES,
    DATA / "reference/universe.csv",
    E23,
    ROOT / "src/riskengine/ingest/linking.py",
    ROOT / "src/riskengine/ingest/linking_extended.py",
    ROOT / "src/riskengine/eval/linker_extended.py",
    ROOT / "src/riskengine/api.py",
    ROOT / "src/riskengine/config.py",
    ROOT / "tests/test_linker_extended.py",
]
SEED = 22


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rule_hash() -> str:
    return hashlib.sha256(RULES.read_text().split("## Result\n", 1)[0].encode()).hexdigest()


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def write_json(path: Path, obj: dict) -> None:
    with path.open("x") as out:
        out.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def pool() -> pd.DataFrame:
    """Recover only headline provenance. Cached predictions are never read."""
    from riskengine.impact.fresh import load_fresh_news

    allowed = set(pd.read_csv(SOURCES[0], usecols=["text"]).text.map(norm))
    history = pd.read_csv(
        SOURCES[2],
        usecols=["Date", "Article_title", "Stock_symbol"],
        dtype=str,
        keep_default_na=False,
    )
    history["text"] = history.Article_title.str.strip()
    history = history[history.text.map(norm).isin(allowed)].copy()
    history["date"] = pd.to_datetime(history.Date, utc=True, errors="coerce").astype(str)
    history["symbols"] = history.Stock_symbol.map(lambda s: [s] if s else [])
    history["cache"] = "history_texts"
    # This is deliberately the ONLY column read from the model-score cache.
    ids = set(pd.read_csv(SOURCES[1], usecols=["doc_id"], dtype=str).doc_id)
    fresh = load_fresh_news(SOURCES[3])
    fresh = fresh[fresh.doc_id.isin(ids)].copy()
    fresh["date"] = fresh.published_at.astype(str)
    fresh["symbols"] = fresh.tickers
    fresh["cache"] = "headline_scores_fresh:doc_id only"
    return pd.concat(
        [frame[["text", "date", "symbols", "cache"]] for frame in [history, fresh]],
        ignore_index=True,
    )


def draw(frame: pd.DataFrame, catalog: set[str], n: int = 100) -> tuple[pd.DataFrame, dict]:
    """Merge duplicate-title provenance before fixed balanced sampling."""
    merged = {}
    canonical = {"FB": "META", "GOOG": "GOOGL"}
    for row in frame.itertuples():
        key = norm(row.text)
        if not key:
            continue
        item = merged.setdefault(
            key, {"key": key, "text": row.text, "dates": set(), "symbols": set(), "caches": set()}
        )
        # Deterministic representative independent of input row ordering.
        item["text"] = min(item["text"], row.text)
        item["dates"].add(str(row.date))
        item["symbols"].update(canonical.get(s, s) for s in row.symbols if s)
        item["caches"].add(row.cache)
    groups = {True: [], False: []}
    for key in sorted(merged):
        row = merged[key]
        if not row["symbols"]:
            continue
        member = bool(row["symbols"] & catalog)
        row["stratum"] = "inside" if member else "outside"
        groups[member].append(row)
    if any(len(g) < n for g in groups.values()):
        raise ValueError("insufficient provenance pool for fixed strata")
    rng = random.Random(SEED)
    chosen = [*rng.sample(groups[True], n), *rng.sample(groups[False], n)]
    random.Random(SEED).shuffle(chosen)
    rows = []
    for i, row in enumerate(chosen, 1):
        out = dict(row)
        out["id"] = f"L38-{i:03d}"
        for field in ["dates", "symbols", "caches"]:
            out[field] = json.dumps(sorted(out[field]))
        rows.append(out)
    return pd.DataFrame(rows), {"inside": len(groups[True]), "outside": len(groups[False])}


def identities(frame: pd.DataFrame) -> str:
    raw = frame[["id", "text"]].to_json(orient="records", force_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def export() -> dict:
    if (WORK / "sample.csv").exists():
        raise FileExistsError("blind sample already exists")
    catalog = set(pd.read_csv(CATALOG).ticker)
    sample, counts = draw(pool(), catalog)
    WORK.mkdir(parents=True, exist_ok=True)
    blind = sample[["id", "text"]].copy()
    blind["gold_entities"], blind["rationale"] = "", ""
    blind.to_csv(WORK / "sample.csv", index=False, mode="x")
    sample.drop(columns=["text"]).to_csv(WORK / "provenance.csv", index=False, mode="x")
    record = {
        "seed": SEED,
        "rows": len(blind),
        "strata": {"inside": 100, "outside": 100},
        "pool_counts": counts,
        "identity_sha256": identities(blind),
        "blank_sample_sha256": sha(WORK / "sample.csv"),
        "rule_sha256": rule_hash(),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        "catalog_sha256": sha(CATALOG),
        "probes_sha256": sha(PROBES),
        "note": (
            "No linker/model output; source symbols are hidden sampling provenance, not labels."
        ),
    }
    write_json(WORK / "sampling.json", record)
    return {k: record[k] for k in ["seed", "rows", "strata", "pool_counts"]}


def load_labels(path: Path, expected: int = 200) -> tuple[pd.DataFrame, list[set[str]]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) != expected or frame.id.duplicated().any():
        raise ValueError("wrong label count or duplicate IDs")
    labels = []
    for row in frame.itertuples():
        value = json.loads(row.gold_entities)
        if (
            not isinstance(value, list)
            or any(not isinstance(v, str) or not v for v in value)
            or len(value) != len(set(value))
            or not row.rationale.strip()
        ):
            raise ValueError(f"invalid blind reference: {row.id}")
        labels.append(set(value))
    return frame, labels


def freeze() -> dict:
    sampling = json.loads((WORK / "sampling.json").read_text())
    labels, _ = load_labels(WORK / "sample.csv")
    if identities(labels) != sampling["identity_sha256"]:
        raise ValueError("sample identities/text changed")
    if rule_hash() != sampling["rule_sha256"]:
        raise ValueError("rules changed after sampling")
    if sha(CATALOG) != sampling["catalog_sha256"] or sha(PROBES) != sampling["probes_sha256"]:
        raise ValueError("catalog/probes changed after sampling")
    for path, digest in sampling["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"sampling source changed: {path}")
    files = [
        *CANDIDATE_FILES,
        *SOURCES,
        WORK / "sample.csv",
        WORK / "sampling.json",
        WORK / "provenance.csv",
    ]
    record = {
        "kind": "ai_blind_link_reference",
        "labeller": "ChatGPT",
        "model": "not shown",
        "interface": "ChatGPT through Codex CLI (agent session)",
        "utc_time": datetime.now(UTC).isoformat(),
        "rows": len(labels),
        "independent_human_check": False,
        "same_resolver": True,
        "rule_sha256": rule_hash(),
        "files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
    }
    write_json(WORK / "frozen.json", record)
    return {"rows": len(labels), "frozen": True, "kind": record["kind"]}


def metrics(gold: list[set[str]], predictions: list[set[str]]) -> dict:
    if len(gold) != len(predictions) or not gold:
        raise ValueError("nonempty aligned evaluation required")
    precisions, recalls, f1s = [], [], []
    tp = fp = fn = empty_both = exact = 0
    for truth, pred in zip(gold, predictions, strict=True):
        hit = len(truth & pred)
        tp += hit
        fp += len(pred - truth)
        fn += len(truth - pred)
        exact += truth == pred
        if pred:
            precisions.append(hit / len(pred))
        if truth:
            recalls.append(hit / len(truth))
        if pred or truth:
            f1s.append(2 * hit / (len(pred) + len(truth)))
        else:
            empty_both += 1

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "rows": len(gold),
        "precision": mean(precisions),
        "recall": mean(recalls),
        "f1": mean(f1s),
        "wrong_links": fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_rows": len(precisions),
        "recall_rows": len(recalls),
        "f1_rows": len(f1s),
        "empty_both": empty_both,
        "empty_predictions": len(gold) - len(precisions),
        "empty_references": len(gold) - len(recalls),
        "exact_set_accuracy": exact / len(gold),
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def subject_links(linker, text: str) -> set[str]:
    return set(linker.link(text)) - linker.analyst_only(text)


def decision(old: dict, new: dict, differences: int) -> bool:
    return (
        new["precision"] is not None
        and new["precision"] >= 0.95
        and new["recall"] is not None
        and old["recall"] is not None
        and new["recall"] > old["recall"]
        and differences == 0
    )


def score_once(result: Path = RESULT, work: Path = WORK) -> dict:
    """Exclusive reservation prevents concurrent or repeat scoring, including a failed attempt."""
    if result.exists():
        raise FileExistsError("E38 scored result already exists; second run refused")
    frozen = json.loads((work / "frozen.json").read_text())
    if rule_hash() != frozen["rule_sha256"]:
        raise ValueError("frozen rules changed")
    for path, digest in frozen["files_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"frozen input changed: {path}")
    frame, gold = load_labels(work / "sample.csv")
    from riskengine.ingest.linking import Linker
    from riskengine.ingest.linking_extended import ExtendedLinker

    result.parent.mkdir(parents=True, exist_ok=True)
    with result.with_suffix(".run_lock").open("x") as lock:
        lock.write(datetime.now(UTC).isoformat())
    old, extended = Linker.from_csv(), ExtendedLinker.from_csv()
    predictions = {
        name: [subject_links(linker, text) for text in frame.text]
        for name, linker in [("old", old), ("extended", extended)]
    }
    scores = {name: metrics(gold, pred) for name, pred in predictions.items()}
    # Read E23 only after the probe/catalog hashes were frozen; never read its labels.
    credit = pd.read_csv(E23, usecols=["headline_id", "headline", "target_company"], dtype=str)
    if len(credit) != 243:
        raise ValueError("E23 must have 243 original pairs")
    originals = set(old.tickers)
    comparison = []
    for row in credit.itertuples():
        a, b = set(old.link(row.headline)), set(extended.link(row.headline))
        projected = b & originals
        comparison.append(
            {
                "id": row.headline_id,
                "target": row.target_company,
                "old": sorted(a),
                "extended_originals": sorted(projected),
                "added_outside": sorted(b - originals),
                "differs": a != projected,
                "target_differs": (row.target_company in a) != (row.target_company in b),
            }
        )
    differences = sum(row["differs"] or row["target_differs"] for row in comparison)
    probes = pd.read_csv(PROBES, dtype=str)
    if len(probes) != 60 or probes.id.duplicated().any():
        raise ValueError("expected 60 frozen synthetic probes")
    categories = {}
    for row in probes.itertuples():
        expected = set(json.loads(row.expected))
        pred = subject_links(extended, row.text)
        group = categories.setdefault(row.category, {"passed": 0, "total": 0, "failures": []})
        group["total"] += 1
        group["passed"] += pred == expected
        if pred != expected:
            group["failures"].append(
                {"id": row.id, "expected": sorted(expected), "got": sorted(pred)}
            )
    adopt = decision(scores["old"], scores["extended"], differences)
    rows = []
    for i, row in enumerate(frame.itertuples()):
        rows.append(
            {
                "id": row.id,
                "text": row.text,
                "gold": sorted(gold[i]),
                "old_subjects": sorted(predictions["old"][i]),
                "extended_subjects": sorted(predictions["extended"][i]),
                "old_mentions": list(old.link(row.text)),
                "extended_mentions": list(extended.link(row.text)),
            }
        )
    out = {
        "experiment": "E38",
        "utc_time": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "reference_kind": "ai_blind_link_reference",
        "same_resolver": True,
        "independent_human_check": False,
        "set_a": scores,
        "set_a_rows": rows,
        "out_of_catalog_reference_links": sum(len(g - set(extended.tickers)) for g in gold),
        "set_b": {
            "rows": len(credit),
            "original_company_differences": differences,
            "rows_with_outside_additions": sum(bool(r["added_outside"]) for r in comparison),
            "comparisons": comparison,
        },
        "synthetic_probes": categories,
        "adopt_live_default": adopt,
        "decision_rule": "primary recall higher; primary precision >=0.95; E23 original diff=0",
        "rules_text": RULES.read_text().split("## Result\n", 1)[0],
        "frozen": frozen,
        "limits": "Same-AI labels, one small historical company-conditioned sample; coverage only.",
    }
    write_json(result, out)
    return {
        "set_a": scores,
        "set_b_differences": differences,
        "synthetic_probes": categories,
        "adopt_live_default": adopt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--label", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = export() if args.label else freeze() if args.freeze else score_once()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
