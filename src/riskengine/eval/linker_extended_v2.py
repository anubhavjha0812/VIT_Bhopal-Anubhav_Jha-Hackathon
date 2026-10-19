"""E38b: fixed second linker candidate, sealed reference and one offline scored run."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskengine.eval.linker_extended import (
    E23,
    PROBES,
    metrics,
    norm,
    sha,
    subject_links,
    write_json,
)
from riskengine.ingest.linking import Linker
from riskengine.ingest.linking_extended import EXTENDED_PATH, ExtendedLinker
from riskengine.paths import DATA, ROOT

WORK = ROOT / "private_notes/annotation_uploads/task1/linker_extended_v2"
RESULT = DATA / "outputs/linker_extended_v2.json"
RULES = ROOT / "docs/LINKER_EXTENDED_V2.md"
LABELS_SHA256 = "f7362bbb07d60c2e4db931276f9379b179878c580361e77945ea309786945ab0"
JUDGEMENT_COLUMNS = [
    "id",
    "view",
    "text",
    "tickers_added",
    "tickers_removed",
    "ticker",
    "change",
    "removed_link_was_subject",
    "reason",
]


def rules_text() -> str:
    return RULES.read_text().split("## Result\n", 1)[0]


def read_credit() -> pd.DataFrame:
    frame = pd.read_csv(E23, usecols=["headline_id", "headline", "target_company"], dtype=str)
    if len(frame) != 243 or frame.headline_id.duplicated().any():
        raise ValueError("expected exactly 243 E23 pairs")
    return frame


def set_b_comparisons(frame: pd.DataFrame, old: Linker, candidate: ExtendedLinker) -> list[dict]:
    originals = set(old.tickers)
    records = []
    for row in frame.itertuples():
        a, b = set(old.link(row.headline)), set(candidate.link(row.headline))
        records.append(
            {
                "id": row.headline_id,
                "text": row.headline,
                "target": row.target_company,
                "old_mentions": sorted(a),
                "candidate_mentions": sorted(b & originals),
                "old_subjects": sorted(subject_links(old, row.headline)),
                "candidate_subjects": sorted(subject_links(candidate, row.headline) & originals),
                "added_outside": sorted(b - originals),
                "target_membership_changed": (row.target_company in a) != (row.target_company in b),
            }
        )
    return records


def judgement_template(records: list[dict]) -> pd.DataFrame:
    """Every changed original-ticker link, in both mention and subject projections."""
    rows = []
    for row in records:
        for view in ["mentions", "subjects"]:
            before, after = set(row[f"old_{view}"]), set(row[f"candidate_{view}"])
            added, removed = sorted(after - before), sorted(before - after)
            for change, tickers in [("removed", removed), ("added", added)]:
                for ticker in tickers:
                    rows.append(
                        {
                            "id": row["id"],
                            "view": view,
                            "text": row["text"],
                            "tickers_added": json.dumps(added),
                            "tickers_removed": json.dumps(removed),
                            "ticker": ticker,
                            "change": change,
                            "removed_link_was_subject": "" if change == "removed" else "NA",
                            "reason": "",
                        }
                    )
    return pd.DataFrame(rows, columns=JUDGEMENT_COLUMNS)


def validate_judgements(expected: pd.DataFrame, actual: pd.DataFrame) -> int:
    fixed = [c for c in JUDGEMENT_COLUMNS if c not in ["removed_link_was_subject", "reason"]]
    if list(actual.columns) != JUDGEMENT_COLUMNS or not actual[fixed].equals(expected[fixed]):
        raise ValueError("Set B judgements do not cover the exact frozen differences")
    if not actual.reason.str.strip().ne("").all():
        raise ValueError("Set B judgement reason is missing")
    removed = actual.change.eq("removed")
    if not actual.loc[removed, "removed_link_was_subject"].isin(["yes", "no"]).all():
        raise ValueError("every removed link needs a yes/no subject judgement")
    if not actual.loc[~removed, "removed_link_was_subject"].eq("NA").all():
        raise ValueError("added links must be marked NA for removal")
    wrong = actual[removed & actual.removed_link_was_subject.eq("yes")]
    return len(wrong[["id", "ticker"]].drop_duplicates())


def adoption(old: dict, new: dict, wrong_removals: int, passed: int) -> dict:
    checks = {
        "precision_at_least_095": new["precision"] is not None and new["precision"] >= 0.95,
        "recall_higher": old["recall"] is not None
        and new["recall"] is not None
        and new["recall"] > old["recall"],
        "no_wrong_original_removal": wrong_removals == 0,
        "at_least_58_probes": passed >= 58,
    }
    return {"checks": checks, "adopt_live": all(checks.values())}


def verify_seal(work: Path = WORK) -> dict:
    manifest = json.loads((work / "run_manifest.json").read_text())
    if manifest["rules_text"] != rules_text():
        raise ValueError("fixed E38b rules changed")
    for name, digest in manifest["files_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"sealed file changed: {name}")
    if sha(work / "labels_final.csv") != LABELS_SHA256:
        raise ValueError("settled label SHA-256 differs from the registered reference")
    record = json.loads((work / "labels_final_record.json").read_text())
    if record["labels_final_sha256"] != LABELS_SHA256:
        raise ValueError("settled reference record differs")
    judged = json.loads((work / "setb_judgements_record.json").read_text())
    if judged["sha256"] != sha(work / "setb_judgements.csv"):
        raise ValueError("Set B judgement SHA-256 differs")
    return manifest


def score_once(result: Path = RESULT, work: Path = WORK) -> dict:
    if result.exists():
        raise FileExistsError("E38b result exists; second scored run refused")
    manifest = verify_seal(work)
    sample = pd.read_csv(
        work / "sample.csv", usecols=["id", "text"], dtype=str, keep_default_na=False
    )
    labels = pd.read_csv(work / "labels_final.csv", dtype=str, keep_default_na=False)
    if (
        len(sample) != 200
        or sample.id.duplicated().any()
        or sample.id.tolist() != labels.id.tolist()
    ):
        raise ValueError("expected the same 200 settled IDs in file order")
    if sample.text.map(norm).duplicated().any():
        raise ValueError("sample contains duplicate normalised headlines")
    catalog = set(pd.read_csv(EXTENDED_PATH, usecols=["ticker"]).ticker)
    gold = []
    for value in labels.gold_entities:
        items = json.loads(value)
        if not isinstance(items, list) or any(not isinstance(s, str) for s in items):
            raise ValueError("reference must contain JSON arrays of strings")
        if len(items) != len(set(items)) or any(
            s not in catalog and not (s.startswith("name:") and s[5:].strip() and s == s.lower())
            for s in items
        ):
            raise ValueError("reference must use catalog tickers or lowercase name: identities")
        gold.append(set(items))

    lock_path = result.with_suffix(".run_lock")
    result.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x") as lock:
        lock.write(datetime.now(UTC).isoformat())
    try:
        old, candidate = Linker.from_csv(), ExtendedLinker.from_csv(candidate=2)
        comparisons = set_b_comparisons(read_credit(), old, candidate)
        judgements = pd.read_csv(work / "setb_judgements.csv", dtype=str, keep_default_na=False)
        wrong = validate_judgements(judgement_template(comparisons), judgements)
        # No prediction on the new sample happens until the complete pre-run judgement is checked.
        predictions = {
            name: [subject_links(linker, text) for text in sample.text]
            for name, linker in [("old", old), ("candidate_2", candidate)]
        }
        scores = {name: metrics(gold, preds) for name, preds in predictions.items()}
        rows = []
        for i, row in enumerate(sample.itertuples()):
            rows.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "gold": sorted(gold[i]),
                    "old_subjects": sorted(predictions["old"][i]),
                    "candidate_2_subjects": sorted(predictions["candidate_2"][i]),
                    "old_mentions": list(old.link(row.text)),
                    "candidate_2_mentions": list(candidate.link(row.text)),
                }
            )
        probes = pd.read_csv(PROBES, dtype=str)
        if len(probes) != 60 or probes.id.duplicated().any():
            raise ValueError("expected exactly 60 unchanged probes")
        categories, probe_rows = {}, []
        for row in probes.itertuples():
            expected = set(json.loads(row.expected))
            pred = subject_links(candidate, row.text)
            passed = pred == expected
            group = categories.setdefault(row.category, {"passed": 0, "total": 0})
            group["total"] += 1
            group["passed"] += passed
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
        choice = adoption(scores["old"], scores["candidate_2"], wrong, passed)
        diff_counts = {
            view: sum(r[f"old_{view}"] != r[f"candidate_{view}"] for r in comparisons)
            for view in ["mentions", "subjects"]
        }
        out = {
            "experiment": "E38b",
            "utc_time": datetime.now(UTC).isoformat(),
            "candidate": 2,
            "reference_kind": "two_ai_labellers_settled_before_candidate_run",
            "independent_human_check": False,
            "set_a": scores,
            "set_a_rows": rows,
            "reference_links": sum(map(len, gold)),
            "out_of_catalog_reference_links": sum(len(g - catalog) for g in gold),
            "set_b": {
                "rows": 243,
                "difference_rows": diff_counts,
                "unique_changed_pairs": judgements.id.nunique(),
                "wrong_removed_links_unique_pair_ticker": wrong,
                "judgement_rows": len(judgements),
                "comparisons": comparisons,
                "judgements": judgements.to_dict("records"),
                "note": (
                    "Both original mention projection and desk-filtered subject projection audited."
                ),
            },
            "synthetic_probes": categories,
            "probe_rows": probe_rows,
            "probe_passed": passed,
            "decision": choice,
            "manifest": manifest,
            "limits": (
                "AI reference, one small historical company-conditioned sample; coverage only."
            ),
        }
        write_json(result, out)
        return {
            "set_a": scores,
            "set_b_differences": diff_counts,
            "wrong_removals": wrong,
            "probes": f"{passed}/60",
            "decision": choice,
        }
    except BaseException:
        # The written protocol allows a fix/retry only when no result JSON was produced.
        if not result.exists():
            lock_path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, action="store_true")
    parser.parse_args()
    print(json.dumps(score_once(), indent=2))


if __name__ == "__main__":
    main()
