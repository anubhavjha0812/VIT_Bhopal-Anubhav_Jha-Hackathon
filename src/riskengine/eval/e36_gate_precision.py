"""One-run check of the fixed rules in docs/E36_GATE_PRECISION.md.

uv run python -m riskengine.eval.e36_gate_precision --run
uv run python -m riskengine.eval.e36_gate_precision --label-audit   # after labelling the sample
"""

import argparse
import json

import pandas as pd

from riskengine.eval.credit_gold import _norm, history_pool, replay_pool
from riskengine.gate_rules import blocked_reason
from riskengine.ingest.linking import Linker
from riskengine.paths import DATA

OUT = DATA / "outputs"
RESULTS = OUT / "e36_gate_precision.json"
E23_ROWS = OUT / "e24_e23_gate_v6_rows.csv"
SEC_8K_ROWS = OUT / "e25_credit_8k_full_rows.csv"
RESULTS_B = OUT / "e36b_gate_precision.json"
AUDIT = OUT / "e36_removal_audit_sample.csv"
CREDIT = ["Default", "Credit downgrade", "Covenant breach"]
AUDIT_SEED, AUDIT_N = 36, 60
MIN_REMOVED_FALSE, MAX_8K_BLOCKED, MIN_AUDIT_SHARE = 20, 2, 0.8


def e23_design() -> dict:
    rows = pd.read_csv(E23_ROWS)
    fired = rows[rows["predicted_fire"]]
    why = [
        blocked_reason(h, e) for h, e in zip(fired["headline"], fired["rule_class"], strict=True)
    ]
    fired = fired.assign(blocked=why)
    removed = fired[fired["blocked"].notna()]
    tp, fp = int((fired["outcome"] == "TP").sum()), int((fired["outcome"] == "FP").sum())
    fn = int((rows["outcome"] == "FN").sum())
    tp_lost, fp_removed = (
        int((removed["outcome"] == "TP").sum()),
        int((removed["outcome"] == "FP").sum()),
    )
    tp_new, fp_new = tp - tp_lost, fp - fp_removed
    return {
        "note": "design data: the 44 false alerts were seen before the rules were written",
        "before": {"alerts": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                   "precision": tp / (tp + fp), "recall": tp / (tp + fn)},
        "after": {"alerts": tp_new + fp_new, "TP": tp_new, "FP": fp_new, "FN": fn + tp_lost,
                  "precision": tp_new / (tp_new + fp_new),
                  "recall": tp_new / (tp_new + fn + tp_lost)},
        "false_removed": fp_removed,
        "true_removed": tp_lost,
        "false_removed_by_rule": removed[removed["outcome"] == "FP"]["blocked"]
        .value_counts().to_dict(),
        "false_removed_by_old_reason": removed[removed["outcome"] == "FP"]["false_alert_reason"]
        .value_counts().to_dict(),
        "true_removed_rows": removed[removed["outcome"] == "TP"]["headline"].tolist(),
        "removed_rows": removed[["headline", "outcome", "blocked"]].to_dict("records"),
    }  # fmt: skip


def sec_8k_guard() -> dict:
    rows = pd.read_csv(SEC_8K_ROWS)
    credit = rows[rows["final"].isin(CREDIT)]
    why = [blocked_reason(t, e) for t, e in zip(credit["text"], credit["final"], strict=True)]
    blocked = credit.assign(blocked=why).dropna(subset=["blocked"])
    return {
        "credit_true_rows": len(credit),
        "blocked": len(blocked),
        "recall_change_at_most": len(blocked) / len(credit),
        "blocked_rows": blocked[["text", "final", "blocked"]].to_dict("records"),
    }


def pool_removals(e23_texts: set[str]) -> tuple[dict, pd.DataFrame]:
    """Alerts of the old gate in history and replay that the rules remove."""
    linker = Linker.from_csv()
    frames = []
    for name, pool in (("history", history_pool(linker)), ("replay", replay_pool(linker))):
        fired = pool[pool["fired"]].copy()
        fired["blocked"] = [
            blocked_reason(t, e) for t, e in zip(fired["text"], fired["event"], strict=True)
        ]
        fired["set"] = name
        frames.append(fired)
    fired = pd.concat(frames, ignore_index=True)
    removed = fired.dropna(subset=["blocked"])
    summary = {
        s: {
            "old_alerts": int((fired["set"] == s).sum()),
            "removed": int((removed["set"] == s).sum()),
        }
        for s in ("history", "replay")
    }
    replay_removed = removed[removed["set"] == "replay"]
    summary["replay_removed_rows"] = replay_removed[["ticker", "text", "blocked"]].to_dict(
        "records"
    )
    outside = removed[~removed["text"].map(_norm).isin(e23_texts)]
    outside = outside.drop_duplicates(subset=["text"])
    summary["removed_outside_e23_unique_texts"] = len(outside)
    return summary, outside


def run() -> None:
    if RESULTS.exists():
        raise SystemExit("E36 already scored; the rules were fixed before the run (one run).")
    design = e23_design()
    e23_texts = {_norm(h) for h in pd.read_csv(E23_ROWS)["headline"]}
    pools, outside = pool_removals(e23_texts)
    sample = outside.sample(n=min(AUDIT_N, len(outside)), random_state=AUDIT_SEED)
    sample = sample.sort_values("published_at")
    sample[["set", "ticker", "published_at", "text"]].assign(label="", rationale="").to_csv(
        AUDIT, index=False
    )
    ok = {
        "e23_false_removed_at_least_20": design["false_removed"] >= MIN_REMOVED_FALSE,
        "e23_no_true_removed": design["true_removed"] == 0,
    }
    out = {
        "rules": "docs/E36_GATE_PRECISION.md",
        "e23_design_data": design,
        "sec_8k_guard": sec_8k_guard(),
        "pools": pools,
        "audit_sample_file": str(AUDIT.relative_to(DATA.parent)),
        "audit_sample_size": len(sample),
        "checks_before_audit": ok,
    }
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print(
        json.dumps({k: v for k, v in out.items() if k not in ("pools",)}, indent=1, default=str)[
            :3500
        ]
    )
    print({k: v for k, v in pools.items() if k != "replay_removed_rows"})


def run_b() -> None:
    """E36b: the same checks after the outlook-revised-to-negative amendment (no fresh audit)."""
    if RESULTS_B.exists():
        raise SystemExit("E36b already scored")
    e23_texts = {_norm(h) for h in pd.read_csv(E23_ROWS)["headline"]}
    pools, outside = pool_removals(e23_texts)
    design, guard = e23_design(), sec_8k_guard()
    out = {
        "rules": "docs/E36_GATE_PRECISION.md, section E36b",
        "note": "amended after the E36 run; E23 is design data and no unseen alert set exists",
        "e23_design_data": design,
        "sec_8k_guard": guard,
        "pools": pools,
        "decision": {
            "e23_false_removed_at_least_20": design["false_removed"] >= MIN_REMOVED_FALSE,
            "e23_at_most_1_true_removed": design["true_removed"] <= 1,
            "sec_8k_at_most_2_blocked": guard["blocked"] <= MAX_8K_BLOCKED,
            "replay_window_removed": pools["replay"]["removed"],
        },
    }
    RESULTS_B.write_text(json.dumps(out, indent=2, default=str))
    keep = ("before", "after", "false_removed", "true_removed", "true_removed_rows")
    print(json.dumps({k: design[k] for k in keep}, indent=1))
    print(json.dumps({"8k_blocked": guard["blocked"], "decision": out["decision"]}, indent=1))
    print({k: v for k, v in pools.items() if k != "replay_removed_rows"})


def label_audit() -> None:
    res = json.loads(RESULTS.read_text())
    if "audit" in res:
        raise SystemExit("audit already stored")
    lab = pd.read_csv(AUDIT)
    lab = lab[lab["label"].isin(["Y", "N"])]
    n, k = len(lab), int((lab["label"] == "Y").sum())
    res["audit"] = {
        "labeller": "AI, blind to the rule that removed the alert",
        "n": n,
        "correctly_removed": k,
        "share": k / n,
        "wrongly_removed": lab[lab["label"] == "N"][["text", "rationale"]].to_dict("records"),
    }
    d, g, a = res["e23_design_data"], res["sec_8k_guard"], res["audit"]
    res["decision"] = {
        "e23_false_removed_at_least_20": d["false_removed"] >= MIN_REMOVED_FALSE,
        "e23_no_true_removed": d["true_removed"] == 0,
        "sec_8k_at_most_2_blocked": g["blocked"] <= MAX_8K_BLOCKED,
        "audit_share_at_least_80_percent": a["share"] >= MIN_AUDIT_SHARE,
        "replay_real_alerts_kept": "check by hand against replay_removed_rows",
    }
    RESULTS.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({"audit": {k: v for k, v in a.items() if k != "wrongly_removed"},
                      "decision": res["decision"]}, indent=1))  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--label-audit", action="store_true")
    ap.add_argument("--run-b", action="store_true")
    args = ap.parse_args()
    if args.run:
        run()
    elif args.run_b:
        run_b()
    elif args.label_audit:
        label_audit()
