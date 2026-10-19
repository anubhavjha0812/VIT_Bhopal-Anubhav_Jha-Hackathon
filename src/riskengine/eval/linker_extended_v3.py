"""E38c one-shot evaluator. Test-folder access happens only inside --run.

Candidate/test/Set-B hashes live in the separate design folder. Final labels must
be settled and hash-frozen by another pass before this evaluator can score them.
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskengine.eval.linker_extended import PROBES, metrics, norm, sha, subject_links, write_json
from riskengine.eval.linker_extended_v2 import (
    adoption,
    judgement_template,
    read_credit,
    set_b_comparisons,
    validate_judgements,
)
from riskengine.ingest.linking import Linker
from riskengine.ingest.linking_extended import EXTENDED_PATH, ExtendedLinker
from riskengine.paths import DATA, ROOT

WORK = ROOT / "private_notes/annotation_uploads/task1/linker_extended_v3"
DESIGN = ROOT / "private_notes/annotation_uploads/task1/linker_extended_v3_design"
RESULT = DATA / "outputs/linker_extended_v3.json"
RULES = ROOT / "docs/LINKER_EXTENDED_V3.md"


def rules_text() -> str:
    return RULES.read_text().split("## Result\n", 1)[0]


def verify_design(design: Path = DESIGN) -> dict:
    manifest = json.loads((design / "candidate_manifest.json").read_text())
    if manifest["rules_text"] != rules_text():
        raise ValueError("fixed E38c rules changed")
    for path, digest in manifest["files_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"sealed candidate/design file changed: {path}")
    record = json.loads((design / "setb3_judgements_record.json").read_text())
    if record["sha256"] != sha(design / "setb3_judgements.csv"):
        raise ValueError("Set B judgement hash changed")
    return manifest


def read_frozen_reference(work: Path) -> tuple[pd.DataFrame, list[set[str]], dict]:
    """The first contact with real v3 test files is made by score_once, after design seal."""
    labels_path = work / "labels_final.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(
            "await labels_final.csv and its freeze record; no score run started"
        )
    record_path = work / "labels_final_record.json"
    record = json.loads(record_path.read_text())
    expected = record.get("labels_final_sha256", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("final label record must provide labels_final_sha256")
    if sha(labels_path) != expected:
        raise ValueError("frozen final label SHA-256 mismatch")
    sample_path = work / "sample.csv"
    sample_hash, record_hash = sha(sample_path), sha(record_path)
    sample = pd.read_csv(sample_path, usecols=["id", "text"], dtype=str, keep_default_na=False)
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    if (
        len(sample) != 200
        or sample.id.duplicated().any()
        or sample.id.tolist() != labels.id.tolist()
    ):
        raise ValueError("expected 200 final labels matching sample IDs and order")
    if sample.text.str.strip().eq("").any() or sample.text.map(norm).duplicated().any():
        raise ValueError("blank/duplicate normalised sample text")
    for folder in ["linker_extended", "linker_extended_v2"]:
        earlier = pd.read_csv(
            ROOT / "private_notes/annotation_uploads/task1" / folder / "sample.csv",
            usecols=["text"],
            dtype=str,
        )
        if set(sample.text.map(norm)) & set(earlier.text.map(norm)):
            raise ValueError("v3 sample is not disjoint from earlier samples")
    catalog = set(pd.read_csv(EXTENDED_PATH, usecols=["ticker"]).ticker)
    gold = []
    for value in labels.gold_entities:
        entities = json.loads(value)
        if not isinstance(entities, list) or any(not isinstance(s, str) for s in entities):
            raise ValueError("labels must contain JSON arrays of strings")
        if len(entities) != len(set(entities)) or any(
            s not in catalog and not (s.startswith("name:") and s[5:].strip() and s == s.lower())
            for s in entities
        ):
            raise ValueError("labels must use catalog tickers or lowercase name: identities")
        gold.append(set(entities))
    if (
        sha(labels_path) != expected
        or sha(sample_path) != sample_hash
        or sha(record_path) != record_hash
    ):
        raise ValueError("reference changed while loading")
    return (
        sample,
        gold,
        {
            "labels_sha256": expected,
            "sample_sha256": sample_hash,
            "record_sha256": record_hash,
            "record": record,
        },
    )


def score_once(result: Path = RESULT, work: Path = WORK, design: Path = DESIGN) -> dict:
    if result.exists():
        raise FileExistsError("E38c result exists; second scored run refused")
    manifest = verify_design(design)
    # No test-folder listing or read occurs before this call, even at module import.
    sample, gold, reference = read_frozen_reference(work)
    old, candidate = Linker.from_csv(), ExtendedLinker.from_csv(candidate=3)
    comparisons = set_b_comparisons(read_credit(), old, candidate)
    judgements = pd.read_csv(design / "setb3_judgements.csv", dtype=str, keep_default_na=False)
    wrong = validate_judgements(judgement_template(comparisons), judgements)
    probes = pd.read_csv(PROBES, dtype=str)
    if len(probes) != 60 or probes.id.duplicated().any():
        raise ValueError("expected 60 unchanged probes")
    result.parent.mkdir(parents=True, exist_ok=True)
    lock_path = result.with_suffix(".run_lock")
    with lock_path.open("x") as lock:
        lock.write(datetime.now(UTC).isoformat())
    try:
        predictions = {
            name: [subject_links(linker, text) for text in sample.text]
            for name, linker in [("old", old), ("candidate_3", candidate)]
        }
        scores = {name: metrics(gold, values) for name, values in predictions.items()}
        rows = []
        for i, row in enumerate(sample.itertuples()):
            rows.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "gold": sorted(gold[i]),
                    "old_subjects": sorted(predictions["old"][i]),
                    "candidate_3_subjects": sorted(predictions["candidate_3"][i]),
                    "old_mentions": list(old.link(row.text)),
                    "candidate_3_mentions": list(candidate.link(row.text)),
                }
            )
        probe_rows, categories = [], {}
        for row in probes.itertuples():
            expected, pred = set(json.loads(row.expected)), subject_links(candidate, row.text)
            passed = expected == pred
            group = categories.setdefault(row.category, {"passed": 0, "total": 0})
            group["passed"] += passed
            group["total"] += 1
            probe_rows.append(
                {
                    "id": row.id,
                    "category": row.category,
                    "text": row.text,
                    "expected": sorted(expected),
                    "predicted": sorted(pred),
                    "passed": passed,
                }
            )
        passed = sum(g["passed"] for g in categories.values())
        choice = adoption(scores["old"], scores["candidate_3"], wrong, passed)
        counts = {
            view: sum(r[f"old_{view}"] != r[f"candidate_{view}"] for r in comparisons)
            for view in ["mentions", "subjects"]
        }
        catalog = set(candidate.tickers)
        out = {
            "experiment": "E38c",
            "candidate": 3,
            "seed": 40,
            "utc_time": datetime.now(UTC).isoformat(),
            "set_a": scores,
            "set_a_rows": rows,
            "reference": reference,
            "reference_kind": "two_ai_labellers_settled_before_run",
            "independent_human_check": False,
            "reference_links": sum(map(len, gold)),
            "out_of_catalog_reference_links": sum(len(g - catalog) for g in gold),
            "set_b": {
                "rows": len(comparisons),
                "difference_rows": counts,
                "wrong_removed_links_unique_pair_ticker": wrong,
                "comparisons": comparisons,
                "judgements": judgements.to_dict("records"),
            },
            "synthetic_probes": categories,
            "probe_rows": probe_rows,
            "probe_passed": passed,
            "decision": choice,
            "manifest": manifest,
            "limits": "Two AI labellers, one small historical sample; coverage only.",
        }
        write_json(result, out)
        return {
            "set_a": scores,
            "set_b_differences": counts,
            "wrong_removals": wrong,
            "probes": f"{passed}/60",
            "decision": choice,
        }
    except BaseException:
        if not result.exists():
            lock_path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(score_once(), indent=2))


if __name__ == "__main__":
    main()
