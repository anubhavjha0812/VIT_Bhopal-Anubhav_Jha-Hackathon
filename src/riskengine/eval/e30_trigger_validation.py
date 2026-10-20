"""One-run, local E30 check of the fixed rules in docs/experiments/E30_TRIGGER_VALIDATION.md.

uv run python -m riskengine.eval.e30_trigger_validation --run
"""

import argparse
import json

import numpy as np
import pandas as pd

from riskengine.impact.features import build_dataset
from riskengine.impact.train_v2 import FIRST_YEAR, LAST_FIT_YEAR, features, formula, fresh_dataset
from riskengine.impact.train_v3 import fit_predict
from riskengine.paths import DATA

RESULTS = DATA / "outputs" / "e30_trigger_validation.json"
CHOICE = "d_clipped_no_vol"
N_BOOTSTRAP, SEED = 1000, 0
PRIMARY_LABEL = "abs_car01_ge_0.05"
ADOPT = "Adopt v3 as an extra non-credit trigger."
KEEP = "Do not adopt v3 as an extra non-credit trigger; keep the credit and market-wide gates."
DECISION_RULE = (
    "Adopt v3 as an extra non-credit trigger only if, on BOTH F years, precision at P99 "
    "beats volatility only and the 95% interval of the difference excludes 0."
)


def predict_sets(fit_rows: pd.DataFrame, sets: dict, cols: list[str]) -> tuple[dict, dict]:
    """Fit once; predict fit/test rows together, with quantiles taken only from fit rows."""
    frames = [fit_rows, *sets.values()]
    move = fit_predict(CHOICE, fit_rows, pd.concat(frames, ignore_index=True), cols)
    chunks = np.split(np.asarray(move), np.cumsum([len(t) for t in frames])[:-1])
    thresholds = {f"p{p}": float(np.percentile(chunks[0], p)) for p in (95, 99)}
    return thresholds, dict(zip(sets, chunks[1:], strict=True))


def top_k(values: np.ndarray, k: int) -> np.ndarray:
    """Break equal-volatility ties in input row order and fire exactly k rows."""
    fired = np.zeros(len(values), dtype=bool)
    fired[np.argsort(-np.asarray(values), kind="stable")[:k]] = True
    return fired


def make_triggers(rows: pd.DataFrame, move: np.ndarray, thresholds: dict) -> dict:
    rng = np.random.default_rng(SEED)
    triggers = {}
    for level in (95, 99):
        fired = np.asarray(move) >= thresholds[f"p{level}"]
        k = int(fired.sum())
        triggers[f"v3_p{level}"] = fired
        triggers[f"volatility_p{level}"] = top_k(rows["vol20"].to_numpy(), k)
        random = np.zeros(len(rows), dtype=bool)
        random[rng.choice(len(rows), size=k, replace=False)] = True
        triggers[f"random_p{level}"] = random
    triggers["v3_p99_corroborated"] = triggers["v3_p99"] & (rows["n_prev_24h"].to_numpy() >= 1)
    impact = formula(rows)
    triggers |= {f"formula_ge_{cutoff}": impact >= cutoff for cutoff in (7, 8)}
    return triggers


def metrics(fired: np.ndarray, positive: np.ndarray) -> dict:
    fired, positive = np.asarray(fired, dtype=bool), np.asarray(positive, dtype=bool)
    n, k, shocks = len(positive), int(fired.sum()), int(positive.sum())
    tp = int((fired & positive).sum())
    base = shocks / n if n else None
    precision = tp / k if k else None
    return {
        "n": n,
        "fired": k,
        "true_positives": tp,
        "positives": shocks,
        "precision": precision,
        "recall": tp / shocks if shocks else None,
        "lift": precision / base if precision is not None and base else None,
        "base_rate": base,
    }


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0
    )


def _interval(values: np.ndarray) -> tuple[list, int]:
    finite = values[np.isfinite(values)]
    ci = np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None]
    return ci, len(finite)


def cluster_bootstrap(
    rows: pd.DataFrame,
    triggers: dict,
    labels: dict,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Resample whole stock-days, with the same draw for every trigger and label.

    Masks stay fixed: this estimates uncertainty for the scored triggers, without
    refitting, re-ranking volatility or re-drawing random triggers in a resample.
    Undefined ratios are omitted from percentile intervals; valid counts are stored.
    """
    codes, clusters = pd.factorize(pd.MultiIndex.from_frame(rows[["ticker", "event_day"]]))
    n_clusters, nt, nl = len(clusters), len(triggers), len(labels)
    if not n_clusters:
        raise ValueError("E30 test sets must contain at least one stock-day")
    fires = np.column_stack(list(triggers.values())).astype(int)
    positives = np.column_stack(list(labels.values())).astype(int)
    tp = (fires[:, :, None] * positives[:, None, :]).reshape(len(rows), nt * nl)
    counts = np.column_stack([fires, positives, tp])
    grouped = np.zeros((n_clusters, counts.shape[1]), dtype=np.int64)
    np.add.at(grouped, codes, counts)
    precision = np.empty((n_resamples, nt, nl))
    recall = np.empty_like(precision)
    rng = np.random.default_rng(seed)
    for b in range(n_resamples):
        total = grouped[rng.integers(n_clusters, size=n_clusters)].sum(axis=0)
        true = total[nt + nl :].reshape(nt, nl)
        precision[b] = _divide(true, total[:nt, None])
        recall[b] = _divide(true, total[None, nt : nt + nl])
    intervals = {}
    for i, trigger in enumerate(triggers):
        intervals[trigger] = {}
        for j, label in enumerate(labels):
            p_ci, p_valid = _interval(precision[:, i, j])
            r_ci, r_valid = _interval(recall[:, i, j])
            intervals[trigger][label] = {
                "precision_ci95": p_ci,
                "recall_ci95": r_ci,
                "precision_valid_resamples": p_valid,
                "recall_valid_resamples": r_valid,
            }
    names = list(triggers)
    difference = precision[:, names.index("v3_p99")] - precision[:, names.index("volatility_p99")]
    differences = {}
    for j, label in enumerate(labels):
        ci, valid = _interval(difference[:, j])
        differences[label] = {"ci95": ci, "valid_resamples": valid}
    return {"n_clusters": n_clusters, "intervals": intervals, "differences": differences}


def evaluate(rows: pd.DataFrame, move: np.ndarray, thresholds: dict) -> dict:
    triggers = make_triggers(rows, move, thresholds)
    labels = {
        PRIMARY_LABEL: rows["abs_car01"].to_numpy() >= 0.05,
        "z_ge_3": rows["z"].to_numpy() >= 3,
    }
    boot = cluster_bootstrap(rows, triggers, labels)
    table = {}
    for name, fired in triggers.items():
        table[name] = {}
        for label, positive in labels.items():
            row = metrics(fired, positive) | boot["intervals"][name][label]
            if name.startswith("random_"):
                row["expected_precision"] = row["base_rate"]
            table[name][label] = row
    differences = boot["differences"]
    for label in labels:
        a, b = table["v3_p99"][label]["precision"], table["volatility_p99"][label]["precision"]
        differences[label]["precision_difference"] = (
            a - b if a is not None and b is not None else None
        )
    return {
        "n": len(rows),
        "n_stock_days": boot["n_clusters"],
        "triggers": table,
        "v3_p99_minus_volatility_precision": differences,
    }


def decision(sets: dict) -> dict:
    checks = {}
    for year in (2019, 2020):
        name = f"F_fresh_{year}"
        row = sets[name]["v3_p99_minus_volatility_precision"][PRIMARY_LABEL]
        delta, lower = row["precision_difference"], row["ci95"][0]
        checks[name] = bool(delta is not None and lower is not None and delta > 0 and lower > 0)
    adopt = all(checks.values())
    return {
        "rule": DECISION_RULE,
        "primary_year_pass": checks,
        "adopt": adopt,
        "line": ADOPT if adopt else KEEP,
    }


def run() -> dict:
    if RESULTS.exists():
        raise FileExistsError(f"E30 is run once; result already exists: {RESULTS}")
    print("Building local seen-stock frame...", flush=True)
    df = build_dataset()
    cols = features(df)
    fit_rows = df[df["year"].between(FIRST_YEAR, LAST_FIT_YEAR)]
    print(
        f"Fit rows ({FIRST_YEAR}-{LAST_FIT_YEAR}): {len(fit_rows)}; building fresh frame...",
        flush=True,
    )
    fresh = fresh_dataset(sorted(set(df["publisher"]) - {"other"}))
    sets = {f"F_fresh_{y}": fresh[fresh["year"] == y] for y in (2019, 2020)}
    sets |= {f"S_seen_{y}": df[df["year"] == y] for y in (2019, 2020)}
    thresholds, predictions = predict_sets(fit_rows, sets, cols)
    out = {
        "chosen": CHOICE,
        "fit_years": [FIRST_YEAR, LAST_FIT_YEAR],
        "n_fit": len(fit_rows),
        "thresholds": thresholds,
        "threshold_source": "expected move on fit rows only",
        "labels": {PRIMARY_LABEL: "abs_car01 >= 0.05 (primary)", "z_ge_3": "z >= 3 (report only)"},
        "bootstrap": {
            "cluster": ["ticker", "event_day"],
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
            "interval": "95% percentile, paired across triggers",
            "undefined_ratios": "omitted from intervals; valid resample counts stored",
        },
        "random": {
            "seed": SEED,
            "draw": "without replacement, exact matched fire count",
            "expected_precision": "label base rate",
        },
        "volatility_ties": "stable input row order; exact matched fire count",
        "sets": {},
    }
    for name, rows in sets.items():
        out["sets"][name] = evaluate(rows, predictions[name], thresholds)
        print(name, json.dumps(out["sets"][name]["v3_p99_minus_volatility_precision"]), flush=True)
    out["decision"] = decision(out["sets"])
    payload = json.dumps(out, indent=2, allow_nan=False) + "\n"
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("x") as f:
        f.write(payload)
    print(out["decision"]["line"], flush=True)
    print(RESULTS, flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run once and save the fixed E30 check")
    args = parser.parse_args()
    if args.run:
        run()
