"""E23: a credit-event gold set for the frozen alert gate (docs/DECISIONS.md, E23).

uv run python -m riskengine.eval.credit_gold --export     # snapshot the gate, draw the sample
uv run python -m riskengine.eval.credit_gold --evaluate   # once the labels are reviewed

Export writes the blind file data/gold/credit/credit_to_label.csv (no system output), the
system side data/gold/credit/sample_meta.csv, the stratum sizes and the gate snapshot.
Evaluate reads data/gold/credit/credit_labelled.csv and scores reviewed rows only.
"""

import argparse
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from riskengine.config import load_config
from riskengine.gate import gate
from riskengine.ingest.linking import Linker
from riskengine.paths import DATA, ROOT

GOLD = DATA / "gold" / "credit"
BLIND = GOLD / "credit_to_label.csv"
META = GOLD / "sample_meta.csv"
INFO = GOLD / "sample_info.json"
LABELLED = GOLD / "credit_labelled.csv"
SNAPSHOT = DATA / "outputs" / "e23_gate_snapshot.json"
RESULTS = DATA / "outputs" / "e23_results.json"
E3_FILE = DATA / "gold" / "gold_to_label.csv"
SEED, N_S2, N_S3, N_S4 = 23, 60, 60, 40
BOOT, BOOT_SEED, Z = 1000, 0, 1.959964
MIN_SPLIT = 10
HISTORY_START, HISTORY_END = "2010-01-01", "2020-04-08"
SNAPSHOT_FILES = [
    "src/riskengine/gate.py",
    "src/riskengine/nlp/events.py",
    "src/riskengine/nlp/impact.py",
    "src/riskengine/ingest/linking.py",
    "src/riskengine/ingest/relevance.py",
]

SUBTYPES = ["default", "downgrade", "upgrade", "covenant", "liquidity", "refinancing",
            "affirmation", "counterparty", "securitization", "other"]  # fmt: skip
MATERIALITY = ["low", "medium", "high"]
STATUS = ["unreviewed", "ai_labelled", "reviewed", "adjudicated", "excluded"]
SCORED = {"ai_labelled", "reviewed", "adjudicated"}  # AI labels count (E23 amendment)
LABEL_COLS = ["target_company_match_correct", "credit_event_relevant", "event_subtype",
              "materiality", "existing_gate_should_fire", "annotator_rationale",
              "adjudication_status"]  # fmt: skip
RULE_SUBTYPE = {
    "Default": "default",
    "Credit downgrade": "downgrade",
    "Covenant breach": "covenant",
}

# S3: credit-flavoured words or patterns the E13 rules did not call a credit class
HARD_NEGATIVE = re.compile(
    r"affirm|upgrad|outlook|rating|moody|fitch|\bS&P\b|DBRS|securiti|CMBS|mortgage trust|"
    r"\bclasses\b|\b(?:19|20)\d{2}-[A-Z0-9]{1,6}\b|bankrupt|default|liquidity|refinanc|covenant|"
    r"waiver|\bjunk\b|\bbonds?\b|\bdebt\b|lender|counterpart",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def headline_id(source: str, published_at, text: str, ticker: str) -> str:
    key = f"{source}|{published_at}|{text}|{ticker}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def gate_snapshot() -> dict:
    files = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in SNAPSHOT_FILES}
    return {"rule": "docs/DECISIONS.md E23", "files_sha256": files,
            "impact_trigger": load_config().impact_trigger}  # fmt: skip


def period(ts: pd.Timestamp, source_set: str) -> str:
    if source_set == "replay":
        return "replay window 2020"
    return "2010-2016" if ts.year <= 2016 else "2017 - 8 Apr 2020"


def history_pool(linker: Linker) -> pd.DataFrame:
    from riskengine.history import load_history
    from riskengine.nlp.events import credit_event, names_agency
    from riskengine.nlp.impact import impact_formula

    h = load_history()
    h = h[h["why"].isna() & (h["role"] == "subject")]
    end = pd.Timestamp(HISTORY_END, tz="UTC") + pd.Timedelta(days=1)
    h = h[(h["published_at"] >= HISTORY_START) & (h["published_at"] < end)].copy()
    h["rule_class"] = h["text"].map(credit_event)
    h["event"] = h["rule_class"].fillna("Other")
    h["impact"] = [
        impact_formula(e, s, "news", names_agency(t))
        for e, s, t in zip(h["event"], h["sentiment"], h["text"], strict=True)
    ]
    h["source"] = "news"
    g = gate(h.reset_index(drop=True), load_config().impact_trigger, linker)
    return g.assign(source_set="history")


def replay_pool(linker: Linker) -> pd.DataFrame:
    from riskengine.nlp.events import credit_event
    from riskengine.pipeline import load_signals

    s = load_signals()
    s = s[s["role"] == "subject"] if "role" in s else s
    g = gate(s, load_config().impact_trigger, linker)
    return g.assign(source_set="replay", rule_class=g["text"].map(credit_event))


def build_pool(pool: pd.DataFrame, e3_texts: set[str]) -> pd.DataFrame:
    """Unit = (headline, company). Drops E3 headlines and repeated pairs."""
    df = pd.DataFrame({
        "headline_id": [headline_id(s, p, t, k) for s, p, t, k in
                        zip(pool["source"], pool["published_at"], pool["text"], pool["ticker"],
                            strict=True)],
        "headline": pool["text"].map(lambda t: " ".join(str(t).split())),
        "event_date": pd.to_datetime(pool["published_at"], utc=True).dt.date.astype(str),
        "target_company": pool["ticker"],
        "source": pool["source"],
        "source_set": pool["source_set"],
        "period": [period(pd.Timestamp(p), s) for p, s in
                   zip(pool["published_at"], pool["source_set"], strict=True)],
        "rule_class": pool["rule_class"],
        "impact": pool["impact"],
        "fired": pool["fired"].astype(bool),
        "gate_reason": pool["reason"],
    })  # fmt: skip
    df = df[~df["headline"].map(_norm).isin(e3_texts)]
    return df.drop_duplicates("headline_id").reset_index(drop=True)


def stratify(pool: pd.DataFrame) -> pd.Series:
    rules = pool["rule_class"].notna()
    hard = ~rules & pool["headline"].str.contains(HARD_NEGATIVE)
    return pd.Series(
        np.select([pool["fired"], rules, hard], ["S1", "S2", "S3"], default="S4"), index=pool.index
    )


def draw(pool: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """S1 in full; S2, S3, S4 at random. Returns the sample and the stratum population sizes."""
    pool = pool.assign(stratum=stratify(pool))
    rng = np.random.default_rng(seed)
    parts = []
    for s, n in (("S1", None), ("S2", N_S2), ("S3", N_S3), ("S4", N_S4)):
        g = pool[pool["stratum"] == s]
        if n is not None and len(g) > n:
            g = g.iloc[np.sort(rng.choice(len(g), n, replace=False))]
        parts.append(g)
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sizes = pool["stratum"].value_counts().to_dict()
    return sample, {s: int(sizes.get(s, 0)) for s in ("S1", "S2", "S3", "S4")}


def blind(sample: pd.DataFrame) -> pd.DataFrame:
    """The annotator's file: identifiers and text only, empty label columns."""
    out = sample[["headline_id", "headline", "event_date", "target_company"]].copy()
    for c in LABEL_COLS:
        out[c] = ""
    out["adjudication_status"] = "unreviewed"
    return out


def export() -> dict:
    if BLIND.exists() or LABELLED.exists():
        raise SystemExit(f"{BLIND} already exists: the sample is drawn once (E23)")
    snap = gate_snapshot()
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snap, indent=2))
    linker = Linker.from_csv()
    e3 = set(pd.read_csv(E3_FILE)["headline"].map(_norm)) if E3_FILE.exists() else set()
    pool = build_pool(pd.concat([history_pool(linker), replay_pool(linker)], ignore_index=True),
                      e3)  # fmt: skip
    sample, sizes = draw(pool)
    GOLD.mkdir(parents=True, exist_ok=True)
    blind(sample).to_csv(BLIND, index=False)
    sample.drop(columns=["headline"]).to_csv(META, index=False)
    info = {
        "rule": "docs/DECISIONS.md E23",
        "seed": SEED,
        "pool_pairs": len(pool),
        "stratum_population": sizes,
        "stratum_sampled": sample["stratum"].value_counts().to_dict(),
        "e3_headlines_excluded": len(e3),
        "gate_snapshot": str(SNAPSHOT.relative_to(ROOT)),
        "blind_sha256": hashlib.sha256(BLIND.read_bytes()).hexdigest(),
    }
    INFO.write_text(json.dumps(info, indent=2))
    return info


# --- evaluation ---------------------------------------------------------------------------


def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "share": None, "ci95": None}
    p = k / n
    c = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    h = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return {"k": k, "n": n, "share": round(p, 4), "ci95": [round(c - h, 4), round(c + h, 4)]}


def validate(lab: pd.DataFrame) -> list[str]:
    """Problems in the labelled file (empty list = fine). Only scored rows must be complete."""
    errors = []
    bad = ~lab["adjudication_status"].isin(STATUS)
    if bad.any():
        errors.append(f"{bad.sum()} rows with an unknown adjudication_status")
    s = lab[lab["adjudication_status"].isin(SCORED)]
    for col, ok in (("target_company_match_correct", {"Y", "N"}),
                    ("credit_event_relevant", {"Y", "N"}),
                    ("existing_gate_should_fire", {"Y", "N"}),
                    ("event_subtype", set(SUBTYPES))):  # fmt: skip
        bad = ~s[col].isin(ok)
        if bad.any():
            errors.append(f"{bad.sum()} scored rows with an invalid {col}")
    rel = s[s["credit_event_relevant"] == "Y"]
    bad = ~rel["materiality"].isin(MATERIALITY)
    if bad.any():
        errors.append(f"{bad.sum()} credit-relevant rows without a materiality")
    return errors


def _weighted(df: pd.DataFrame, sizes: dict, pred_strata: set[str], truth: str) -> dict:
    """Precision and recall when a stratum set is 'predicted positive', weighted to the pool."""
    est = {}
    for s, n_pop in sizes.items():
        g = df[df["stratum"] == s]
        est[s] = (g[truth] == "Y").mean() * n_pop if len(g) else 0.0
    tp = sum(est[s] for s in pred_strata)
    pred = sum(sizes[s] for s in pred_strata)
    pos = sum(est.values())
    return {"precision": tp / pred if pred else None, "recall": tp / pos if pos else None}


def _boot(df: pd.DataFrame, sizes: dict, pred_strata: set[str], truth: str) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    draws = {"precision": [], "recall": []}
    groups = {s: g for s, g in df.groupby("stratum")}
    for _ in range(BOOT):
        res = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups.values()])
        w = _weighted(res, sizes, pred_strata, truth)
        for k in draws:
            if w[k] is not None:
                draws[k].append(w[k])
    point = _weighted(df, sizes, pred_strata, truth)
    return {k: {"estimate": point[k],
                "ci95": np.percentile(v, [2.5, 97.5]).tolist() if v else None}
            for k, v in draws.items()}  # fmt: skip


def evaluate(lab: pd.DataFrame, meta: pd.DataFrame, info: dict) -> dict:
    errors = validate(lab)
    if errors:
        raise SystemExit("labelled file not ready: " + "; ".join(errors))
    df = lab.merge(meta, on="headline_id", validate="one_to_one", suffixes=("", "_meta"))
    counts = df["adjudication_status"].value_counts().to_dict()
    df = df[df["adjudication_status"].isin(SCORED)].copy()
    sizes = info["stratum_population"]
    s1 = df[df["stratum"] == "S1"]
    df["rule_subtype"] = df["rule_class"].map(RULE_SUBTYPE).fillna("none")
    fired_fp = s1[s1["existing_gate_should_fire"] == "N"]

    def split(by: str) -> dict:
        out = {}
        for key, g in df.groupby(by):
            g1 = g[g["stratum"] == "S1"]
            if len(g1) >= MIN_SPLIT:
                out[key] = {"gate_precision": wilson(int((g1["existing_gate_should_fire"] == "Y")
                                                         .sum()), len(g1))}  # fmt: skip
            else:
                out[key] = {"gate_fired_rows": len(g1), "note": f"n < {MIN_SPLIT}, not scored"}
        return out

    return {
        "rule": "docs/DECISIONS.md E23",
        "rows_by_status": counts,
        "labelled_by": "ai_labelled = AI tools; reviewed or adjudicated = a person",
        "scored_rows_by_stratum": df["stratum"].value_counts().to_dict(),
        "stratum_population": sizes,
        "gate": {
            "precision_on_fired": wilson(
                int((s1["existing_gate_should_fire"] == "Y").sum()), len(s1)
            ),  # fmt: skip
            "weighted": _boot(df, sizes, {"S1"}, "existing_gate_should_fire"),
        },
        "credit_rules": _boot(df, sizes, {"S1", "S2"}, "credit_event_relevant"),
        "target_match_precision": {
            "all_scored": wilson(int((df["target_company_match_correct"] == "Y").sum()), len(df)),
            "by_stratum": {
                s: wilson(int((g["target_company_match_correct"] == "Y").sum()), len(g))
                for s, g in df.groupby("stratum")
            },  # fmt: skip
            "recall": "not measurable: only linked (headline, company) pairs are sampled",
        },
        "subtype_confusion": pd.crosstab(df["rule_subtype"], df["event_subtype"]).to_dict(),
        "fired_false_positive_by_subtype": {
            st: {
                "false_positives": int((fired_fp["event_subtype"] == st).sum()),
                "fired": int((s1["event_subtype"] == st).sum()),
            }
            for st in SUBTYPES
            if (s1["event_subtype"] == st).any()
        },  # fmt: skip
        "by_source": split("source_set"),
        "by_period": split("period"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()
    if args.export:
        print(json.dumps(export(), indent=2))
    if args.evaluate:
        if not LABELLED.exists():
            raise SystemExit(f"no labels yet: fill {BLIND} and save it as {LABELLED}")
        res = evaluate(pd.read_csv(LABELLED, dtype=str, keep_default_na=False),
                       pd.read_csv(META), json.loads(INFO.read_text()))  # fmt: skip
        RESULTS.write_text(json.dumps(res, indent=2, default=float))
        print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
