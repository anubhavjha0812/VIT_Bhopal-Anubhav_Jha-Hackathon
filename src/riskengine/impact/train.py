"""Fit, choose and test the learned impact model (rules: docs/DECISIONS.md, E6).

uv run python -m riskengine.impact.train
"""

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from riskengine.impact.features import CATEGORICAL, NUMERIC, build_dataset
from riskengine.nlp.impact import impact_formula
from riskengine.paths import DATA, ROOT

MODEL_PATH = DATA / "models" / "impact_model.joblib"
RESULTS_JSON = DATA / "outputs" / "impact_results.json"
RESULTS_MD = ROOT / "docs" / "others" / "IMPACT.md"
SPLITS = {
    "train": (2009, 2016),
    "valid": (2017, 2018),
    "test_2019": (2019, 2019),
    "test_2020": (2020, 2020),
}


def features(df: pd.DataFrame) -> list[str]:
    return [*CATEGORICAL, *NUMERIC, *sorted(c for c in df if c.startswith("p_"))]


def candidates(cols: list[str]) -> dict[str, Pipeline]:
    num = [c for c in cols if c not in CATEGORICAL]
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), num),
        ]
    )
    return {
        "ridge": Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))]),
        "gbm": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=0,
                    ),  # fmt: skip
                ),
            ]
        ),
    }


def to_scale(pred: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Predictions -> impact 1..10 using the fixed decile edges."""
    return 1 + np.searchsorted(edges, pred, side="right")


def spearman_ci(x, y, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boots = [spearmanr(x[i], y[i]).statistic for i in rng.integers(0, len(x), (n_boot, len(x)))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


def report(name: str, impact, abs_car) -> dict:
    impact, abs_car = np.asarray(impact), np.asarray(abs_car)
    rho, lo, hi = spearman_ci(impact, abs_car)
    return {
        "model": name,
        "n": int(len(impact)),
        "spearman": rho,
        "ci95": [lo, hi],
        "mean_abs_car_pct_impact_ge8": float(abs_car[impact >= 8].mean() * 100)
        if (impact >= 8).any()
        else None,
        "mean_abs_car_pct_impact_le3": float(abs_car[impact <= 3].mean() * 100)
        if (impact <= 3).any()
        else None,
    }


def split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    lo, hi = SPLITS[name]
    return df[df["year"].between(lo, hi)]


def main() -> None:
    df = build_dataset()
    cols = features(df)
    parts = {k: split(df, k) for k in SPLITS}
    print({k: len(v) for k, v in parts.items()})

    # 1. choose on validation (Spearman of raw predictions vs |CAR|)
    valid_rho = {}
    for name, pipe in candidates(cols).items():
        pipe.fit(parts["train"][cols], parts["train"]["y"])
        pred = pipe.predict(parts["valid"][cols])
        valid_rho[name] = float(spearmanr(pred, parts["valid"]["abs_car01"]).statistic)
    formula_valid = [
        impact_formula(e, s, "news")
        for e, s in zip(parts["valid"]["event"], parts["valid"]["sentiment"], strict=True)
    ]
    valid_rho["formula"] = float(spearmanr(formula_valid, parts["valid"]["abs_car01"]).statistic)
    chosen = max((k for k in valid_rho if k != "formula"), key=valid_rho.get)
    print("validation Spearman:", valid_rho, "-> chosen:", chosen)

    # 2. refit every candidate on 2009-2018; decile edges from the refit model's predictions
    fit_rows = pd.concat([parts["train"], parts["valid"]])
    fitted = {}
    for name, pipe in candidates(cols).items():
        pipe.fit(fit_rows[cols], fit_rows["y"])
        edges = np.quantile(pipe.predict(fit_rows[cols]), np.linspace(0.1, 0.9, 9))
        fitted[name] = (pipe, edges)

    # 3. test once per year: the formula baseline and both candidates (chosen one flagged)
    results = {"validation_spearman": valid_rho, "chosen": chosen, "splits": {}}
    for test in ("test_2019", "test_2020"):
        t = parts[test]
        rows = [
            report(
                "formula (baseline)",
                [
                    impact_formula(e, s, "news")
                    for e, s in zip(t["event"], t["sentiment"], strict=True)
                ],
                t["abs_car01"],
            )
        ]
        for name, (pipe, edges) in fitted.items():
            label = f"{name} (chosen)" if name == chosen else name
            rows.append(report(label, to_scale(pipe.predict(t[cols]), edges), t["abs_car01"]))
        results["splits"][test] = rows

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipe, edges = fitted[chosen]
    joblib.dump({"pipeline": pipe, "edges": edges, "features": cols, "kind": chosen}, MODEL_PATH)
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    RESULTS_MD.write_text(to_markdown(results, {k: len(v) for k, v in parts.items()}))
    print(RESULTS_MD.read_text())


def to_markdown(results: dict, sizes: dict) -> str:
    lines = [
        "# Impact score: learned vs fixed formula",
        "",
        "Generated by `uv run python -m riskengine.impact.train`. Rules: "
        "[DECISIONS.md](DECISIONS.md) E6. Target: |CAR[0,+1]| (market-model abnormal return).",
        "",
        f"Rows: {sizes}. Validation Spearman: "
        + ", ".join(f"{k} {v:.3f}" for k, v in results["validation_spearman"].items())
        + f" → chosen **{results['chosen']}**.",
        "",
    ]
    for test, rows in results["splits"].items():
        df = pd.DataFrame(rows)
        df["ci95"] = df["ci95"].map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
        lines += [f"## {test}", "", df.round(3).to_markdown(index=False), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
