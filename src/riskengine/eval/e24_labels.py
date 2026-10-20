"""E24 label merging: two AI labellers, human adjudication and audit (docs/DECISIONS.md, E24).

Steps:
1. `adjudication_sheet(chatgpt, gemini, ids)` lists every row a person must label: each
   disagreement, each row either labeller marked ambiguous or labelled invalidly, and the audit
   rows. The sheet shows the AI labels only for rows the person must judge anyway.
2. The person fills `human_label` (one of the 12 classes, or `EXCLUDE`) and `human_rationale`.
3. `merge_labels(chatgpt, gemini, human, ids)` returns one row per item with `chatgpt_label`,
   `chatgpt_ambiguous`, `gemini_label`, `gemini_ambiguous`, `audit_selected`, `human_label`,
   `label`, `label_source` (plus `ambiguous` and `rationale` for the evaluation gate), and
   `validate_merged` checks every consistency rule.

Audit rule: among the rows both labellers gave the same valid label with neither marking it
ambiguous ("agreement rows"), ceil(0.20 × n) are drawn without replacement by numpy's
default_rng(24) from the ids sorted as strings. Rounding up means at least 20% are audited
(exactly 20% when n is a multiple of 5). The person's label replaces the AI label on them.

Audit metrics, both over the audited agreement rows: the human correction rate (the person gave a
different class) and the human exclusion rate (the person excluded the row), each with a Wilson
95% interval. Cohen's kappa uses rows where both labellers gave a valid label and neither marked
it ambiguous; the other rows are counted by reason. Excluded rows stay in this report and are
left out of the primary score by the gate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from riskengine.eval.e24_gate import CLASSES, GateError

SEED, AUDIT_SHARE, Z = 24, 0.20, 1.959964
EXCLUDE = "EXCLUDE"
AI_COLUMNS = ["item_id", "label", "ambiguous", "rationale"]
HUMAN_COLUMNS = ["item_id", "human_label", "human_rationale"]
MERGED_COLUMNS = [
    "item_id", "chatgpt_label", "chatgpt_ambiguous", "gemini_label", "gemini_ambiguous",
    "audit_selected", "human_label", "label", "label_source", "ambiguous", "rationale",
    "needs_human",
]  # fmt: skip


def read_ai(df: pd.DataFrame, ids, who: str) -> pd.DataFrame:
    """One labeller's output: the exact frozen ids, once each. Invalid labels are kept (they
    send the row to a person) but flagged."""
    missing = [c for c in AI_COLUMNS if c not in df.columns]
    if missing:
        raise GateError(f"{who} output lacks columns {missing}")
    df = df[AI_COLUMNS].astype(str).apply(lambda s: s.str.strip())
    if df["item_id"].duplicated().any():
        raise GateError(f"{who} output has duplicate item_id")
    if set(df["item_id"]) != set(map(str, ids)):
        raise GateError(f"{who} output ids do not match the frozen item ids")
    df["valid"] = df["label"].isin(CLASSES) & df["ambiguous"].isin({"Y", "N"})
    return df.set_index("item_id")


def audit_rows(agreement_ids, seed: int = SEED) -> list[str]:
    ids = sorted(map(str, agreement_ids))
    k = math.ceil(AUDIT_SHARE * len(ids))
    if k == 0:
        return []
    picked = np.random.default_rng(seed).choice(len(ids), size=k, replace=False)
    return sorted(ids[i] for i in picked)


def _combine(chatgpt: pd.DataFrame, gemini: pd.DataFrame, ids, seed: int) -> pd.DataFrame:
    a, b = read_ai(chatgpt, ids, "chatgpt"), read_ai(gemini, ids, "gemini")
    df = pd.DataFrame(index=sorted(map(str, ids)))
    df.index.name = "item_id"
    df["chatgpt_label"], df["chatgpt_ambiguous"] = a["label"], a["ambiguous"]
    df["gemini_label"], df["gemini_ambiguous"] = b["label"], b["ambiguous"]
    df["chatgpt_rationale"] = a["rationale"]
    valid = a["valid"].reindex(df.index) & b["valid"].reindex(df.index)
    agree = valid & (df["chatgpt_label"] == df["gemini_label"])
    clear = (df["chatgpt_ambiguous"] == "N") & (df["gemini_ambiguous"] == "N")
    df["agreement"] = agree & clear
    df["audit_selected"] = df.index.isin(audit_rows(df.index[df["agreement"]], seed))
    df["reason"] = np.select(
        [~valid, ~agree, ~clear, df["audit_selected"]],
        ["invalid AI label", "disagreement", "ambiguous", "audit"],
        default="",
    )
    df["needs_human"] = df["reason"] != ""
    return df


def adjudication_sheet(chatgpt, gemini, ids, seed: int = SEED) -> pd.DataFrame:
    """Rows a person must label, with empty human columns."""
    df = _combine(chatgpt, gemini, ids, seed)
    sheet = df[df["needs_human"]].reset_index()
    cols = ["item_id", "reason", "chatgpt_label", "chatgpt_ambiguous", "gemini_label",
            "gemini_ambiguous"]  # fmt: skip
    return sheet[cols].assign(human_label="", human_rationale="")


AI_SOURCES = {  # docs/experiments/E25_LABELLER_AMENDMENT.md: when an AI, not a person, adjudicates
    "human_adjudicated": "ai_adjudicated",
    "ai_agreed_human_audited": "ai_agreed_ai_audited",
}


def merge_labels(
    chatgpt, gemini, human: pd.DataFrame, ids, seed: int = SEED, adjudicator: str = "human"
) -> pd.DataFrame:
    """`adjudicator="ai"` marks adjudicated and audited rows as AI-made, never as a person's."""
    df = _combine(chatgpt, gemini, ids, seed)
    missing = [c for c in HUMAN_COLUMNS if c not in human.columns]
    if missing:
        raise GateError(f"human sheet lacks columns {missing}")
    h = human[HUMAN_COLUMNS].astype(str).apply(lambda s: s.str.strip())
    if h["item_id"].duplicated().any():
        raise GateError("human sheet has duplicate item_id")
    need = set(df.index[df["needs_human"]])
    if set(h["item_id"]) != need:
        raise GateError("human sheet must cover exactly the rows that need a person")
    h = h.set_index("item_id")
    ok = h["human_label"].isin([*CLASSES, EXCLUDE])
    if not ok.all():
        raise GateError("human_label must be one of the 12 classes or EXCLUDE")
    if (h["human_rationale"].isin({"", "nan"})).any():
        raise GateError("every human label needs a rationale")
    df["human_label"] = h["human_label"].reindex(df.index).fillna("")
    adjudicated = df["needs_human"] & ~df["audit_selected"]
    excluded = df["human_label"] == EXCLUDE
    names = ["human_adjudicated", "ai_agreed_human_audited"]
    if adjudicator == "ai":
        names = [AI_SOURCES[n] for n in names]
    df["label_source"] = np.select(
        [excluded, adjudicated, df["audit_selected"]],
        ["excluded", *names],
        default="ai_agreed",
    )
    df["label"] = np.where(df["needs_human"], df["human_label"], df["chatgpt_label"])
    df.loc[excluded, "label"] = ""
    either = (df["chatgpt_ambiguous"] == "Y") | (df["gemini_ambiguous"] == "Y")
    df["ambiguous"] = np.where(either, "Y", "N")
    hr = h["human_rationale"].reindex(df.index)
    df["rationale"] = hr.where(df["needs_human"], df["chatgpt_rationale"]).fillna("")
    out = df.reset_index()[MERGED_COLUMNS]
    validate_merged(out)
    return out


def validate_merged(df: pd.DataFrame) -> None:
    """Every label-source rule of E24, row by row."""
    missing = [c for c in MERGED_COLUMNS if c not in df.columns]
    if missing:
        raise GateError(f"merged labels lack columns {missing}")
    if df["item_id"].duplicated().any():
        raise GateError("duplicate item_id in merged labels")
    src, lab, hum = df["label_source"], df["label"], df["human_label"]
    audit = _flag(df["audit_selected"])
    agreement = (
        df["chatgpt_label"].isin(CLASSES) & (df["chatgpt_label"] == df["gemini_label"])
        & (df["chatgpt_ambiguous"] == "N") & (df["gemini_ambiguous"] == "N")
    )  # fmt: skip
    rules = {
        "audit rows must be agreement rows": audit & ~agreement,
        "ai_agreed rows must be two equal, clear AI labels, not audited": (src == "ai_agreed")
        & ~((df["chatgpt_label"] == df["gemini_label"]) & (lab == df["chatgpt_label"])
            & (df["chatgpt_ambiguous"] == "N") & (df["gemini_ambiguous"] == "N") & ~audit),
        "audited rows must use the adjudicator's label": src.isin(
            ["ai_agreed_human_audited", "ai_agreed_ai_audited"]
        )
        & ~(audit & (lab == hum) & hum.isin(CLASSES)),
        "adjudicated rows must use the adjudicator's label": src.isin(
            ["human_adjudicated", "ai_adjudicated"]
        )
        & ~(~audit & (lab == hum) & hum.isin(CLASSES)),
        "excluded rows must be the person's EXCLUDE": (src == "excluded") & ~(hum == EXCLUDE),
        "unknown label_source": ~src.isin(
            {"ai_agreed", "ai_agreed_human_audited", "human_adjudicated", "excluded",
             "ai_adjudicated", "ai_agreed_ai_audited"}  # fmt: skip
        ),
    }  # fmt: skip
    for msg, bad in rules.items():
        if bad.any():
            raise GateError(f"{msg} ({int(bad.sum())} rows)")


def _flag(col: pd.Series) -> pd.Series:
    """A True/False column that survives a CSV round trip ("False" must not read as True)."""
    if col.dtype == bool:
        return col
    m = col.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if m.isna().any():
        raise GateError("audit_selected must be True or False")
    return m.astype(bool)


def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p = k / n
    c = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    h = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return {"k": k, "n": n, "rate": p, "ci95": [c - h, c + h]}


def audit_metrics(merged: pd.DataFrame) -> dict:
    """Audit and agreement figures, each with its denominator stated."""
    validate_merged(merged)
    audit = _flag(merged["audit_selected"])
    audited = merged[audit]
    excl = audited["label_source"].eq("excluded")
    corrected = ~excl & (audited["human_label"] != audited["chatgpt_label"])
    valid = merged["chatgpt_label"].isin(CLASSES) & merged["gemini_label"].isin(CLASSES)
    clear = merged["chatgpt_ambiguous"].eq("N") & merged["gemini_ambiguous"].eq("N")
    use = valid & clear
    agreement = use & (merged["chatgpt_label"] == merged["gemini_label"])
    kappa = None
    if use.sum() >= 2:
        a, b = merged.loc[use, "chatgpt_label"], merged.loc[use, "gemini_label"]
        kappa = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else None
    n = len(audited)
    return {
        "rows": len(merged),
        "label_sources": merged["label_source"].value_counts().to_dict(),
        "excluded_rows_all": int(merged["label_source"].eq("excluded").sum()),
        "invalid_ai_rows": int((~valid).sum()),
        "audit": {
            "eligible_agreement_rows": int(agreement.sum()),
            "audited_rows": n,
            "denominator": "audited agreement rows",
            "human_correction_rate": wilson(int(corrected.sum()), n),
            "human_exclusion_rate": wilson(int(excl.sum()), n),
        },
        "kappa": {
            "value": kappa,
            "denominator": "rows where both AI labels are valid and neither is ambiguous",
            "rows_used": int(use.sum()),
            "rows_left_out": {
                "invalid AI label": int((~valid).sum()),
                "ambiguous (valid labels)": int((valid & ~clear).sum()),
            },
        },
    }
