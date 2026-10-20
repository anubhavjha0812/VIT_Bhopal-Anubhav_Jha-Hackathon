"""Impact v2: volatility-scaled target, novelty, rolling validation, fresh test (DECISIONS E8).

uv run python -m riskengine.impact.train_v2
(needs the fresh set first: uv run python -m riskengine.impact.fresh)
"""

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskengine.impact import fresh
from riskengine.impact.features import CATEGORICAL, NUMERIC_V2, build_dataset
from riskengine.impact.scores import load_scores
from riskengine.impact.train import candidates, report, to_scale
from riskengine.ingest.prices import load_prices
from riskengine.nlp.impact import impact_formula
from riskengine.paths import DATA, ROOT

MODEL_PATH = DATA / "models" / "impact_model_v2.joblib"
RESULTS_JSON = DATA / "outputs" / "impact_v2_results.json"
RESULTS_MD = ROOT / "docs" / "others" / "IMPACT_V2.md"
FIRST_YEAR, FOLDS, LAST_FIT_YEAR = 2009, (2015, 2016, 2017, 2018), 2018
SQRT2 = np.sqrt(2)


class Ensemble:
    """Average of ridge and gradient boosting on the log(1 + z) scale."""

    def __init__(self, cols: list[str]):
        c = candidates(cols)
        self.members = [c["ridge"], c["gbm"]]

    def fit(self, x, y):
        for m in self.members:
            m.fit(x, y)
        return self

    def predict(self, x):
        return np.mean([m.predict(x) for m in self.members], axis=0)


def features(df: pd.DataFrame) -> list[str]:
    return [*CATEGORICAL, *NUMERIC_V2, *sorted(c for c in df if c.startswith("p_"))]


def models(cols: list[str]) -> dict:
    return {**candidates(cols), "ensemble": Ensemble(cols)}


def expected_move(pred_log_z: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """log(1 + z) prediction -> expected |CAR| = z * sqrt(2) * vol20."""
    return np.clip(np.expm1(pred_log_z), 0, None) * SQRT2 * df["vol20"].to_numpy()


def formula(df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [impact_formula(e, s, "news") for e, s in zip(df["event"], df["sentiment"], strict=True)]
    )


def rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def rolling_validation(df: pd.DataFrame, cols: list[str]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for year in FOLDS:
        tr = df[df["year"].between(FIRST_YEAR, year - 1)]
        va = df[df["year"] == year]
        for name, model in models(cols).items():
            model.fit(tr[cols], np.log1p(tr["z"]))
            move = expected_move(model.predict(va[cols]), va)
            scores.setdefault(name, []).append(rho(move, va["abs_car01"]))
        scores.setdefault("volatility only", []).append(rho(va["vol20"], va["abs_car01"]))
        scores.setdefault("formula", []).append(rho(formula(va), va["abs_car01"]))
        print(year, {k: round(v[-1], 3) for k, v in scores.items()}, flush=True)
    return scores


def fresh_dataset(top_publishers: list[str]) -> pd.DataFrame:
    df = build_dataset(
        news=fresh.load_fresh_news(),
        scores=load_scores(fresh.SCORES_PATH),
        prices=load_prices(fresh.PRICES_PATH),
        publisher_src=fresh.NEWS_PATH,
        top_publishers=top_publishers,
    )
    day = df["published_at"].dt.strftime("%Y-%m-%d")
    return df[(day >= fresh.TEST_START) & (day < fresh.TEST_END)]


def test_rows(t: pd.DataFrame, fitted: dict, cols: list[str], chosen: str, vol_edges) -> list:
    rows = []
    for name, f, raw in [
        ("formula (baseline)", formula(t), formula(t)),
        ("volatility only (baseline)", to_scale(t["vol20"].to_numpy(), vol_edges), t["vol20"]),
    ]:
        rows.append({**report(name, f, t["abs_car01"]), "spearman_vs_z": rho(raw, t["z"])})
    for name, (model, edges) in fitted.items():
        move = expected_move(model.predict(t[cols]), t)
        label = f"{name} (chosen)" if name == chosen else name
        row = report(label, to_scale(move, edges), t["abs_car01"])
        rows.append({**row, "spearman_vs_z": rho(move, t["z"])})
    return rows


def main() -> None:
    df = build_dataset()
    cols = features(df)
    top = sorted(set(df["publisher"]) - {"other"})

    # 1. choose by mean Spearman over the rolling validation years
    val = rolling_validation(df, cols)
    mean_val = {k: float(np.mean(v)) for k, v in val.items()}
    chosen = max(models(cols), key=mean_val.get)
    engine_default = mean_val[chosen] > mean_val["volatility only"]
    print("mean validation Spearman:", mean_val, "-> chosen:", chosen, flush=True)

    # 2. refit on 2009-2018; decile edges of the expected move give the 1-10 scale
    fit_rows = df[df["year"].between(FIRST_YEAR, LAST_FIT_YEAR)]
    fitted = {}
    for name, model in models(cols).items():
        model.fit(fit_rows[cols], np.log1p(fit_rows["z"]))
        move = expected_move(model.predict(fit_rows[cols]), fit_rows)
        fitted[name] = (model, np.quantile(move, np.linspace(0.1, 0.9, 9)))
    vol_edges = np.quantile(fit_rows["vol20"], np.linspace(0.1, 0.9, 9))

    # 3. test once: F (fresh stocks, primary) and S (our stocks, years already seen in v1)
    tests = {}
    fresh_df = fresh_dataset(top)
    for year in (2019, 2020):
        tests[f"F_fresh_{year}"] = fresh_df[fresh_df["year"] == year]
    for year in (2019, 2020):
        tests[f"S_seen_{year}"] = df[df["year"] == year]
    results = {
        "rolling_validation": val,
        "mean_validation_spearman": mean_val,
        "chosen": chosen,
        "engine_default": bool(engine_default),
        "splits": {k: test_rows(t, fitted, cols, chosen, vol_edges) for k, t in tests.items()},
    }
    sizes = {"fit_2009_2018": len(fit_rows), **{k: len(t) for k, t in tests.items()}}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model, edges = fitted[chosen]
    joblib.dump(
        {"model": model, "edges": edges, "features": cols, "kind": chosen, "top_publishers": top},
        MODEL_PATH,
    )
    RESULTS_JSON.write_text(json.dumps({"sizes": sizes, **results}, indent=2))
    RESULTS_MD.write_text(to_markdown(results, sizes))
    print(RESULTS_MD.read_text())


def to_markdown(results: dict, sizes: dict) -> str:
    val = pd.DataFrame(results["rolling_validation"], index=FOLDS).T
    val["mean"] = val.mean(axis=1)
    lines = [
        "# Impact score v2: learned vs volatility-only",
        "",
        "Generated by `uv run python -m riskengine.impact.train_v2`. Rules: "
        "[DECISIONS.md](DECISIONS.md) E8. Target: |CAR[0,+1]|. Impact = expected move "
        "(predicted move in normal two-day moves × the stock's volatility), as 1–10 deciles.",
        "",
        f"Rows: {sizes}.",
        "",
        "## Rolling validation (Spearman vs |CAR|, train 2009..Y−1, validate Y)",
        "",
        val.round(3).to_markdown(),
        "",
        f"Chosen: **{results['chosen']}**. Engine default: "
        f"{'v2' if results['engine_default'] else 'unchanged'}.",
        "",
        "**F** = fresh S&P 100 stocks never used for fitting or choosing (primary test). "
        "**S** = our 15 stocks; these years were already looked at in v1.",
        "",
    ]
    for test, rows in results["splits"].items():
        df = pd.DataFrame(rows)
        df["ci95"] = df["ci95"].map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
        lines += [f"## {test}", "", df.round(3).to_markdown(index=False), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
