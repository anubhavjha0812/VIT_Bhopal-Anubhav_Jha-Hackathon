"""How well do the relevance rules keep headlines that are about the company? (DECISIONS E17)

uv run python -m riskengine.eval.relevance

Reads the blind sample (data/relevance/to_label.csv, sample_meta.csv) and the model labels
(labels.csv from notebooks/relevance_labels), applies the frozen rules, and writes
data/outputs/relevance_results.json.
"""

import json
import math

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.ingest.relevance import BANKS, reason
from riskengine.paths import DATA

DIR = DATA / "relevance"
OUT = DATA / "outputs" / "relevance_results.json"
Z = 1.959964


def wilson(k: int, n: int) -> dict:
    """Share k/n with a 95% Wilson interval."""
    if n == 0:
        return {"k": 0, "n": 0, "share": None, "ci95": None}
    p = k / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return {"k": k, "n": n, "share": round(p, 4), "ci95": [round(centre - half, 4),
                                                              round(centre + half, 4)]}  # fmt: skip


def scored(linker: Linker | None = None) -> pd.DataFrame:
    """One row per sampled pair: set, period, bank, label (Y/N), kept by the rules, reason."""
    linker = linker or Linker.from_csv()
    meta = pd.read_csv(DIR / "sample_meta.csv")
    labels = pd.read_csv(DIR / "labels.csv")
    df = meta.merge(labels, on="id", validate="one_to_one")
    df["why"] = [
        reason(linker, t, tk, by_symbol=bool(b))
        for t, tk, b in zip(df["Article_title"], df["ticker"], df["by_symbol"], strict=True)
    ]
    df["kept"] = df["why"].isna()
    year = pd.to_datetime(df["Date"], utc=True).dt.year
    df["period"] = year.map(lambda y: "2010-2016" if y <= 2016 else "2017-2020")
    df["bank"] = df["ticker"].isin(BANKS).map({True: "JPM+BAC", False: "other 13"})
    df["yes"] = df["label"].eq("Y")
    return df


def summary(df: pd.DataFrame) -> dict:
    """Precision before and after the rules and the rules' recall, for one group of S1 pairs.
    Measured on pairs the labellers agree on; `unsure` pairs give the bounds (all N .. all Y)."""
    sure = df[df["label"] != "unsure"]
    kept = sure[sure["kept"]]
    after = df[df["kept"]]
    unsure_kept = int((after["label"] == "unsure").sum())
    yes_kept = int((after["label"] == "Y").sum())
    return {
        "precision_before": wilson(int(sure["yes"].sum()), len(sure)),
        "precision_after": wilson(int(kept["yes"].sum()), len(kept)),
        "recall": wilson(int(kept["yes"].sum()), int(sure["yes"].sum())),
        "unsure": {"all": int((df["label"] == "unsure").sum()), "kept": unsure_kept},
        "precision_after_bounds": [
            round(yes_kept / len(after), 4) if len(after) else None,
            round((yes_kept + unsure_kept) / len(after), 4) if len(after) else None,
        ],
    }


def human_check(df: pd.DataFrame) -> dict:
    """The hand audit (E17): agreement with the models on 30 random agreed pairs, and the
    precision when the human label settles the pairs the models disagreed on."""
    path = DIR / "audit_to_label_completed.csv"
    if not path.exists():
        return {}
    human = pd.read_csv(path)[["id", "about_company_Y_or_N"]]
    human["human"] = human["about_company_Y_or_N"].astype(str).str.strip().str.upper()
    pick = pd.read_csv(DIR / "audit_meta.csv").rename(columns={"why": "pick"})
    h = df.merge(human[["id", "human"]], on="id").merge(pick, on="id")
    audit = h[h["pick"] == "audit"]
    settled = df.merge(human[["id", "human"]], on="id", how="left")
    settled["yes"] = settled["label"].where(settled["label"] != "unsure", settled["human"]).eq("Y")
    s1 = settled[settled["set"] == "S1"]
    kept = s1[s1["kept"]]
    return {
        "audit_agreement": wilson(int((audit["human"] == audit["label"]).sum()), len(audit)),
        "audit_disagreements": audit.loc[audit["human"] != audit["label"], "id"].tolist(),
        "unsure_settled_by_human": h.loc[h["pick"] == "unsure", "human"].value_counts().to_dict(),
        "S1_with_human_settled": {
            "precision_before": wilson(int(s1["yes"].sum()), len(s1)),
            "precision_after": wilson(int(kept["yes"].sum()), len(kept)),
            "recall": wilson(int(kept["yes"].sum()), int(s1["yes"].sum())),
        },
        "S2_with_human_settled": wilson(
            int(settled[settled["set"] == "S2"]["yes"].sum()), int((settled["set"] == "S2").sum())
        ),
    }


def results(df: pd.DataFrame) -> dict:
    s1, s2 = df[df["set"] == "S1"], df[df["set"] == "S2"]
    agree = df[[c for c in ("qwen3_8b", "olmo2_7b", "phi4_mini") if c in df]]
    return {
        "S1_name_linked": {
            "all": summary(s1),
            **{f"period {k}": summary(g) for k, g in s1.groupby("period")},
            **{k: summary(g) for k, g in s1.groupby("bank")},
            "dropped_by_rule": s1["why"].value_counts().to_dict(),
        },
        "S2_own_symbol": {
            "precision": summary(s2)["precision_after"],
            "unsure": int((s2["label"] == "unsure").sum()),
            "precision_bounds": summary(s2)["precision_after_bounds"],
            "dropped_by_rule": s2["why"].value_counts().to_dict(),
        },
        "labellers": {
            **json.loads((DIR / "labellers.json").read_text()),
            "agree_rate_of_used": round(float(df["unanimous"].mean()), 4),
            "label_counts": df["label"].value_counts().to_dict(),
            "y_share": {c: round(float((agree[c] == "Y").mean()), 4) for c in agree},
        },
        "human": human_check(df),
    }


def main() -> None:
    res = results(scored())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
