"""Fresh unlabelled credit-style sample. Keyword strata only; no gate/model/label reads.

uv run python -m riskengine.eval.credit_heldout --dry-run
uv run python -m riskengine.eval.credit_heldout --build

The 2010–2016 E25 frame and exclusion filter are reused. Stored history is read with an
explicit column allowlist; its event predictions, gate decisions and scores are never loaded.
Reserved source texts are used only by the established exclusion filter, never for strata.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.e25_extra import EXTRA_PATH
from riskengine.eval.e25_pool import (
    END,
    POOL_PATH,
    START,
    assert_clean,
    exclusions,
    frame,
    remove_seen,
    sha_file,
)
from riskengine.eval.leaks import norm
from riskengine.paths import DATA, ROOT

# Fixed before inspecting any candidate row or output. These are sampling cues, not labels.
SEED = 27
TARGETS = {
    "rating_action": 55,
    "default_bankruptcy": 30,
    "covenant_waiver": 15,
    "restructuring_distress": 20,
    "hard_negative_pattern": 30,
}
PRIORITY = ["covenant_waiver", "default_bankruptcy", "restructuring_distress", "rating_action"]
STRATA = {
    "rating_action": (
        r"\b(?:moody['’]?s|fitch|dbrs|standard (?:&|and) poor['’]?s)\b|\bs&p\b"
        r"|\bdowngrad\w*\b|\b(?:credit|debt|bond|issuer)\b.{0,45}\b(?:cuts?|outlook|watch)\b"
        r"|\b(?:cuts?|outlook|watch)\b.{0,45}\b(?:ratings?|credit|debt|bonds?)\b"
        r"|\brating (?:watch|outlook|cuts?)\b"
    ),
    "default_bankruptcy": (
        r"\bdefault\w*\b|\bbankrupt\w*\b|\bchapter\s*11\b|\binsolven\w*\b"
        r"|\bmiss(?:ed|es|ing)?\b.{0,30}\b(?:payment|coupon|interest)\b"
    ),
    "covenant_waiver": r"\bcovenants?\b|\bwaiv(?:er|ers|e|es|ed)\b",
    "restructuring_distress": (
        r"\brestructur\w*\b|\bdistress\w*\b|\bliquidity\b|\bcash (?:crunch|squeeze)\b"
        r"|\bgoing[- ]concern\b|\b(?:debt|credit)\b.{0,35}\b(?:exchange|rescue|emergency)\b"
        r"|\b(?:draws?|drawn|drawdown)\b.{0,35}\b(?:credit|loan|revolver)\b"
    ),
}
NEGATIVE_PATTERNS = {
    "securitization": (
        r"\b(?:cmbs|rmbs|clo|cdo|securiti[sz]\w*|pass[- ]through)\b"
        r"|\bclass(?:es)?\b.{0,40}\b(?:notes|certificates|ratings?)\b"
        r"|\b(?:mortgage|commercial)\b.{0,30}\b(?:trust|securities)\b"
    ),
    "counterparty": (
        r"\b(?:counterpart\w*|trustee|servicer|creditor|lender|loan[- ]loss)\b"
        r".{0,70}\b(?:default\w*|bankrupt\w*|debt|credit|loss\w*)\b"
        r"|\b(?:default\w*|bankrupt\w*)\b.{0,70}\b(?:lender|creditor|exposure)\b"
    ),
    "stock_rating": (
        r"\b(?:upgrad\w*|downgrad\w*|initiat\w*|maintain\w*|reiterat\w*)\b"
        r".{0,70}\b(?:buy|sell|hold|neutral|outperform|underperform|overweight|underweight|"
        r"price target)\b"
        r"|\b(?:price targets?|target prices?)\b.{0,30}\b(?:raise\w*|cut\w*|lower\w*)\b"
    ),
    "wrong_company_pattern": (
        r"\b(?:bank|bancorp|securities|capital)\b.{0,65}\b(?:rescues?|bails? out|financ\w*)\b"
        r"|\b(?:client|customer|borrower)\b.{0,60}\b(?:default\w*|bankrupt\w*|restructur\w*)\b"
    ),
}
RULE = (
    "Unlabelled FNSPID 2010-2016 E25 frame, excluding existing pool/extra and all fixed test "
    "corpora; seed-27 disjoint keyword strata, 120 credit-style cues plus 30 hard-negative cues"
)
HISTORY_COLUMNS = ["published_at", "text", "ticker", "why", "role"]
SAMPLE = ROOT / "private_notes/annotation_uploads/task1/credit_heldout/to_label.csv"
MANIFEST = DATA / "outputs/credit_heldout_manifest.json"


def assignments(texts: pd.Series) -> pd.DataFrame:
    """Priority partitions overlap; names describe sampling patterns, never ground truth."""
    compiled = {k: re.compile(v, re.I) for k, v in STRATA.items()}
    negatives = {k: re.compile(v, re.I) for k, v in NEGATIVE_PATTERNS.items()}
    rows = []
    for text in texts.astype(str):
        negative = next((k for k, rx in negatives.items() if rx.search(text)), "")
        group = (
            "hard_negative_pattern"
            if negative
            else next((k for k in PRIORITY if compiled[k].search(text)), "")
        )
        rows.append({"sampling_stratum": group, "negative_pattern": negative})
    return pd.DataFrame(rows, columns=["sampling_stratum", "negative_pattern"], index=texts.index)


def without_used(f: pd.DataFrame, pool_texts: list[str], extra_texts: list[str]) -> pd.DataFrame:
    used = {norm(t) for t in [*pool_texts, *extra_texts]}
    return f[~f["text"].map(norm).isin(used)].reset_index(drop=True)


def draw(f: pd.DataFrame, targets: dict[str, int] | None = None, seed: int = SEED) -> pd.DataFrame:
    """Fix deficits by other eligible credit strata, retaining the hard-negative target.

    Hard-negative patterns rotate equally while available. All other choices are random,
    never ranked by gate output, model confidence or supposed label correctness.
    """
    targets = TARGETS if targets is None else targets
    f = f.sort_values(["published_at", "text", "tickers"], kind="stable").reset_index(drop=True)
    f = f.loc[~f["text"].map(norm).duplicated()].reset_index(drop=True)
    f = pd.concat([f, assignments(f["text"])], axis=1)
    order = np.random.default_rng(seed).permutation(len(f)).tolist()
    chosen = []
    negative_queues = {
        name: [i for i in order if f.at[i, "negative_pattern"] == name]
        for name in NEGATIVE_PATTERNS
    }
    while len(chosen) < targets.get("hard_negative_pattern", 0):
        added = False
        for queue in negative_queues.values():
            if queue and len(chosen) < targets["hard_negative_pattern"]:
                chosen.append(queue.pop(0))
                added = True
        if not added:
            break
    negative_count = len(chosen)
    for group in PRIORITY:
        chosen.extend(
            [i for i in order if f.at[i, "sampling_stratum"] == group][: targets.get(group, 0)]
        )
    credit_target = sum(n for k, n in targets.items() if k != "hard_negative_pattern")
    chosen_set = set(chosen)
    for i in order:
        if len(chosen) - negative_count >= credit_target:
            break
        if i not in chosen_set and f.at[i, "sampling_stratum"] in PRIORITY:
            chosen.append(i)
            chosen_set.add(i)
    # Independent seeded presentation order hides the strata in the annotation sheet.
    selected = f.iloc[chosen].copy().reset_index(drop=True)
    selected = selected.iloc[np.random.default_rng(seed + 1).permutation(len(selected))]
    return selected.reset_index(drop=True)


def clean_remainder(history: pd.DataFrame, corpus: dict, pool_texts, extra_texts) -> tuple:
    original = frame(history)
    left = without_used(original, pool_texts, extra_texts)
    audit = {"frame_rows": len(original), "after_exact_pool_extra_removal": len(left)}
    combined = dict(corpus) | {"E25 pool": list(pool_texts), "E25 extra": list(extra_texts)}
    passes = []
    for _ in range(6):
        smaller, counts = remove_seen(left, combined)
        passes.append(counts)
        if len(smaller) == len(left):
            audit["exclusion_passes"] = passes
            audit["clean_remainder_rows"] = len(left)
            return left, combined, audit
        left = smaller
        if left.empty:
            raise RuntimeError("no clean unused headlines remain")
    raise RuntimeError("exclusion filter did not stabilize")


def build(write: bool) -> dict:
    if write and (SAMPLE.exists() or MANIFEST.exists()):
        raise FileExistsError("preserve the existing held-out sample and manifest")
    from riskengine.eval.e24 import exclusion_corpus
    from riskengine.history import HISTORY_PATH

    # Do not load history event, impact, sentiment, fired, reason or any label column.
    history = pd.read_csv(HISTORY_PATH, usecols=HISTORY_COLUMNS, dtype={"ticker": str})
    history["published_at"] = pd.to_datetime(history["published_at"], utc=True)
    pool_texts = pd.read_csv(POOL_PATH, usecols=["text"])["text"].astype(str).tolist()
    extra_texts = pd.read_csv(EXTRA_PATH, usecols=["text"])["text"].astype(str).tolist()
    # Stable neutral aliases keep the manifest focused on this repository's exclusion policy.
    corpus = {
        f"fixed_exclusions_{i:02d}": texts
        for i, texts in enumerate(exclusions(exclusion_corpus()).values())
    }
    left, corpus, audit = clean_remainder(history, corpus, pool_texts, extra_texts)
    available = assignments(left["text"])["sampling_stratum"].value_counts().to_dict()
    for _ in range(6):
        picked = draw(left)
        if picked.empty:
            raise RuntimeError("no eligible keyword matches remain")
        clean, _ = remove_seen(picked, corpus)
        if len(clean) == len(picked):
            break
        rejected = set(picked.loc[~picked["text"].isin(clean["text"]), "text"].map(norm))
        left = left[~left["text"].map(norm).isin(rejected)].reset_index(drop=True)
    else:
        raise RuntimeError("sample exclusion filter did not stabilize")
    assert_clean(picked, corpus)
    if set(picked["text"].map(norm)) & {norm(t) for t in [*pool_texts, *extra_texts]}:
        raise RuntimeError("sample overlaps the training pool or extra rows")
    sample = pd.DataFrame(
        {
            "id": [f"CH{k:03d}" for k in range(len(picked))],
            "text": picked["text"],
            "date": pd.to_datetime(picked["published_at"], utc=True).dt.date.astype(str),
            "tickers": picked["tickers"],
        }
    )
    csv_bytes = sample.to_csv(index=False).encode()
    import hashlib

    counts = Counter(picked["sampling_stratum"])
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rule": RULE,
        "seed": SEED,
        "sample_rows": len(sample),
        "target_rows": sum(TARGETS.values()),
        "targets": TARGETS,
        "stratum_counts": {k: counts[k] for k in TARGETS},
        "hard_negative_pattern_counts": picked.loc[
            picked["sampling_stratum"] == "hard_negative_pattern", "negative_pattern"
        ]
        .value_counts()
        .to_dict(),
        "available_keyword_strata": available,
        "window": [START, END],
        "shortfall_vs_target": sum(TARGETS.values()) - len(sample),
        "quota_deficit_policy": (
            "fill scarce credit cues from other credit strata; keep negative target"
        ),
        "strata_are_labels": False,
        "sample_has_labels_or_model_outputs": False,
        "model_or_gate_run": False,
        "labels_read": False,
        "reserved_text_access": "mechanical exclusion only through existing E25 helpers",
        "history_columns_read": HISTORY_COLUMNS,
        "source_files_sha256": {
            str(p.relative_to(ROOT)): sha_file(p) for p in (HISTORY_PATH, POOL_PATH, EXTRA_PATH)
        },
        "script_sha256": sha_file(Path(__file__)),
        "sample_path": str(SAMPLE.relative_to(ROOT)),
        "sample_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "exclusion_audit": audit,
        "sampling_bias": "keyword-enriched challenge sample; not a population prevalence estimate",
        "annotation_status": "unlabelled; obtain AI-derived labels before gate/model evaluation",
    }
    if write:
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE.write_bytes(csv_bytes)
        MANIFEST.write_text(json.dumps(result, indent=2, default=int) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--build", action="store_true")
    args = parser.parse_args()
    result = build(write=args.build)
    print(json.dumps(result, indent=2, default=int))


if __name__ == "__main__":
    main()
