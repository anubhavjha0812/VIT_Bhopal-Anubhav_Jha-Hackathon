"""E24 development: candidate predictors, optional temperature on D1, selection on D2, freeze
(docs/DECISIONS.md, E24).

uv run python -m riskengine.eval.e24_dev --select      # needs D1 and D2 labels; freezes once

Labels are read only through `e24_gate.load_dev_labels`, which refuses final-set items. This
module never opens a final label file or a raw final annotation output.

Candidates (E24): A = the current engine (E13 credit rules override the 9-class model);
B = A + product rule + narrow Product-news mapping (a model "Product news" without a product
event becomes Other); C = B + earnings rule; D = C + one global abstention threshold on the
model's (optionally temperature-scaled) top probability. The D1-only amendment in
docs/E24_AMENDMENT.md adds holdings/analyst routing to B–D. Order: credit → holdings →
analyst → earnings preview/results → product → model; abstention applies only to model predictions.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from riskengine.eval import e24_rules
from riskengine.eval.e24_gate import (
    CLASSES,
    GRID,
    GateError,
    Paths,
    freeze_candidate,
    load_dev_labels,
    macro_f1,
)
from riskengine.nlp.events import credit_event
from riskengine.paths import DATA

ORDER = ["A", "B", "C", "D"]  # simplest first
PREDICTOR_CANDIDATES = [*ORDER, "E"]  # E is D1-only; selection/freeze remain a separate step.
TIE, DROP, MIN_ROWS, MIN_CLASSES = 0.01, 0.05, 10, 5
RULES_FILE = Path(e24_rules.__file__)
E3_FILE = DATA / "gold" / "gold_to_label.csv"
SELECTION = DATA / "outputs" / "e24_dev_selection.json"

Probs = Callable[[Sequence[str]], tuple[list[str], np.ndarray]]
Predictor = Callable[[Sequence[str]], tuple[list[str], list[str]]]


def engine_probs() -> Probs:
    """The unchanged 9-class model, evaluated one row at a time with pinned CPU settings."""
    from riskengine.eval.e24_inference import EvaluationProbs

    return EvaluationProbs()


def scale(p: np.ndarray, temperature: float | None) -> np.ndarray:
    if temperature is None:
        return p
    z = np.log(np.clip(p, 1e-12, 1.0)) / temperature
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Config:
    candidate: str
    threshold: float | None = None
    temperature: float | None = None


def build_predictor(cfg: Config, probs: Probs | None = None) -> Predictor:
    """texts -> (labels, sources), with the amendment rules enabled only for B–D."""
    if cfg.candidate not in PREDICTOR_CANDIDATES:
        raise GateError(f"unknown candidate {cfg.candidate!r}")
    if (cfg.candidate == "D") != (cfg.threshold is not None):
        raise GateError("only candidate D has a threshold, and D needs one")
    if cfg.threshold is not None and cfg.threshold not in GRID:
        raise GateError("threshold must come from the fixed grid")
    if cfg.candidate == "E":
        from riskengine.eval.e25_candidate import build_candidate

        return build_candidate(probs=probs, temperature=cfg.temperature)
    probs = probs or engine_probs()

    def predict(texts: Sequence[str]) -> tuple[list[str], list[str]]:
        names, p = probs(texts)
        p = scale(p, cfg.temperature)
        top = p.argmax(axis=1)
        labels, sources = [], []
        for text, i, row in zip(texts, top, p, strict=True):
            model_label, conf = names[i], float(row[i])
            if credit := credit_event(text):
                labels.append(credit), sources.append("credit_rule")
            elif cfg.candidate in "BCD" and e24_rules.holdings_rule(text):
                labels.append("Other"), sources.append("holdings_rule")
            elif cfg.candidate in "BCD" and (analyst := e24_rules.analyst_rule(text)):
                labels.append(analyst), sources.append("analyst_rule")
            elif cfg.candidate in "CD" and e24_rules.earnings_preview_rule(text):
                labels.append("Other"), sources.append("earnings_preview_rule")
            elif cfg.candidate in "CD" and e24_rules.earnings_rule(text):
                labels.append("Earnings"), sources.append("earnings_rule")
            elif cfg.candidate in "BCD" and e24_rules.product_rule(text):
                labels.append("Product news"), sources.append("product_rule")
            elif cfg.candidate == "D" and conf < cfg.threshold:
                labels.append("Other"), sources.append("abstain")
            elif cfg.candidate in "BCD" and model_label == "Product news":
                labels.append("Other"), sources.append("model_product_to_other")
            else:
                labels.append(model_label), sources.append("model")
        return labels, sources

    return predict


def support(labels: pd.DataFrame) -> pd.Series:
    kept = labels[labels["label_source"] != "excluded"]
    return kept["label"].value_counts().reindex(CLASSES, fill_value=0)


def fit_temperature(d1: pd.DataFrame, texts: Sequence[str], probs: Probs) -> dict:
    """Temperature on D1 only, only if ≥ 5 classes have ≥ 10 labelled D1 rows (E24)."""
    sup = support(d1)
    eligible = int((sup >= MIN_ROWS).sum())
    out = {"d1_support": sup.to_dict(), "classes_with_10_rows": eligible}
    if eligible < MIN_CLASSES:
        return out | {"temperature": None, "reason": "D1 support too low: raw probabilities"}
    names, p = probs(texts)
    keep = np.array([lab in names for lab in d1["label"]]) & (d1["label_source"] != "excluded")
    y = np.array([names.index(lab) for lab in d1.loc[keep, "label"]])
    logp = np.log(np.clip(p[keep.to_numpy()], 1e-12, 1.0))

    def nll(t: float) -> float:
        z = logp / t
        z = z - z.max(axis=1, keepdims=True)
        return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean())

    t = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
    return out | {"temperature": round(t, 4), "rows_used": int(len(y))}


def per_class_f1(y, pred) -> dict:
    vals = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    return dict(zip(CLASSES, map(float, vals), strict=True))


E_MARGIN = 0.02  # docs/E25_AMENDMENT.md: E must beat the best of A-D by this much


def select(
    d2: pd.DataFrame, d2_texts: Sequence[str], probs: Probs, temperature,
    e_probs: Probs | None = None,
) -> dict:  # fmt: skip
    """Score A, B, C and D (each grid threshold) on D2 only; apply the E24 choice rule.
    With e_probs (the student), E is also scored and needs a clear margin (E25 amendment)."""
    kept = d2["label_source"] != "excluded"
    y = d2.loc[kept, "label"].tolist()
    texts = [t for t, k in zip(d2_texts, kept, strict=True) if k]
    big = [c for c, n in support(d2).items() if n >= MIN_ROWS]

    def score(cfg: Config) -> dict:
        pred, _ = build_predictor(cfg, probs)(texts)
        return {"macro_f1": macro_f1(y, pred), "per_class_f1": per_class_f1(y, pred)}

    rows = {c: score(Config(c, temperature=temperature)) for c in "ABC"}
    grid = {t: score(Config("D", t, temperature)) for t in GRID}
    best_t = max(GRID, key=lambda t: (grid[t]["macro_f1"], -t))  # ties -> lowest threshold
    rows["D"] = grid[best_t]
    base = rows["A"]["per_class_f1"]
    allowed = {c: all(r["per_class_f1"][k] >= base[k] - DROP for k in big) for c, r in rows.items()}
    best = max(r["macro_f1"] for c, r in rows.items() if allowed[c])
    chosen = next(c for c in ORDER if allowed[c] and rows[c]["macro_f1"] >= best - TIE)
    if e_probs is not None:
        pred, _ = build_predictor(Config("E", temperature=None), e_probs)(texts)
        rows["E"] = {"macro_f1": macro_f1(y, pred), "per_class_f1": per_class_f1(y, pred)}
        allowed["E"] = all(rows["E"]["per_class_f1"][k] >= base[k] - DROP for k in big)
        if allowed["E"] and rows["E"]["macro_f1"] >= best + E_MARGIN:
            chosen = "E"
    return {
        "d2_rows_scored": len(y),
        "d2_support": support(d2).to_dict(),
        "classes_with_10_rows": big,
        "scores": rows,
        "threshold_grid": {str(t): r["macro_f1"] for t, r in grid.items()},
        "threshold_for_D": best_t,
        "threshold_at_grid_boundary": best_t in (GRID[0], GRID[-1]),
        "allowed": allowed,
        "chosen": chosen,
    }


def e3_texts(ids: Sequence[str]) -> list[str]:
    e3 = pd.read_csv(E3_FILE, dtype=str).set_index("doc_id")["headline"]
    return [e3[i] for i in ids]


def run_selection(
    p: Paths, d1_path, d2_path, probs: Probs | None = None, e_probs: Probs | None = None
) -> dict:  # fmt: skip
    """D1 → optional temperature; D2 → candidate and threshold; then freeze once."""
    d1 = load_dev_labels(d1_path, "D1", p)
    d2 = load_dev_labels(d2_path, "D2", p)
    probs = probs or engine_probs()
    temp = fit_temperature(d1, e3_texts(d1["item_id"]), probs)
    sel = select(d2, e3_texts(d2["item_id"]), probs, temp["temperature"], e_probs)
    c = sel["chosen"]
    threshold = sel["threshold_for_D"] if c == "D" else None
    frozen = freeze_candidate(
        p, c, threshold, temp["temperature"], RULES_FILE if c != "A" else None
    )
    report = {"rule": "docs/DECISIONS.md E24", "temperature": temp, "selection": sel,
              "frozen": frozen}  # fmt: skip
    (p.outputs / SELECTION.name).write_text(json.dumps(report, indent=2, default=float))
    return report


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--select", action="store_true")
    a.add_argument("--with-e", action="store_true", help="also score E (docs/E25_AMENDMENT.md)")
    a.add_argument("--d1", default=str(DATA / "gold/e24/e3_D1_labels.csv"))
    a.add_argument("--d2", default=str(DATA / "gold/e24/e3_D2_labels.csv"))
    args = a.parse_args()
    if args.select:
        e_probs = None
        if args.with_e:
            from riskengine.eval.e25_candidate import student_probs

            e_probs = student_probs()
        report = run_selection(Paths(), args.d1, args.d2, e_probs=e_probs)
        print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
