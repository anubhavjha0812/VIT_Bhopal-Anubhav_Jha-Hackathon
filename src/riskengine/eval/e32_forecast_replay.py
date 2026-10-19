"""E32: one offline chronological replay; rules in docs/E32_FORECAST_REPLAY.md.

uv run python -m riskengine.eval.e32_forecast_replay --run

Observation dates have no release vintages. The fixed availability buffer is an
assumption; this does not establish a fully vintage-correct real-time backtest.
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.module_b import outlook
from riskengine.module_b.backtest import FEATURES, moves
from riskengine.module_b.book import BOOK_PATH, load_book
from riskengine.module_b.revalue import revalue
from riskengine.module_b.shock import BOOK_DATE, TRADING_DAYS, Market
from riskengine.module_b.sources import MARKET_PATH
from riskengine.paths import DATA

ROOT = DATA.parent
RULES = ROOT / "docs/E32_FORECAST_REPLAY.md"
OUTPUT = DATA / "outputs/e32_forecast_replay.json"
TARGETS = [*FEATURES, "d_ust"]
HORIZONS = (60, 20)
AVAILABILITY_SESSIONS = 10
NEIGHBOURS = 3
MEAN_WEIGHT, STATE_WEIGHT = 0.75, 0.25
BLOCK_MONTHS, RESAMPLES, SEED = 6, 1000, 0
PASS_RATIO = 0.85
CANDIDATES = ("historical_mean", "no_change", "fixed_shock", "clean_analog",
              "state_model", "combination")  # fmt: skip
MARKET_COLUMNS = ["baa", "aaa", "vix", "spx", "usdinr", "ust10y"]


def prepare_market(m: pd.DataFrame) -> pd.DataFrame:
    """Forward fill only; dates are observations, not verified publication dates."""
    if not isinstance(m.index, pd.DatetimeIndex) or m.index.has_duplicates:
        raise ValueError("Market dates must be unique timestamps")
    m = m.sort_index()[MARKET_COLUMNS].ffill()
    if not np.isfinite(m.to_numpy()).all() or (m[["spx", "usdinr"]] <= 0).any().any():
        raise ValueError("Market data must be finite, with positive equity and FX levels")
    return m


def state_at(m: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.Series, pd.Timestamp]:
    j = m.index.get_loc(origin) - AVAILABILITY_SESSIONS
    if j < outlook.MOMENTUM:
        raise ValueError("Not enough earlier observations for the buffered E15 state")
    state = pd.Series({"baa": m["baa"].iloc[j], "vix": m["vix"].iloc[j],
                       "baa_20d": m["baa"].iloc[j] - m["baa"].iloc[j - outlook.MOMENTUM]})
    return state, m.index[j]


def target_history(m: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """E15 month-end design, with configurable H and explicit assumed availability."""
    rows = []
    for origin in outlook.month_ends(m.loc[outlook.TRAIN_FROM:]):
        i = m.index.get_loc(origin)
        ready = i + horizon + AVAILABILITY_SESSIONS
        if i < outlook.MOMENTUM + AVAILABILITY_SESSIONS or ready >= len(m):
            continue
        state, observed = state_at(m, origin)
        rows.append({"origin": origin, "window_end": m.index[i + horizon],
                     "available_on": m.index[ready], "state_observed": observed,
                     **state.to_dict(), **moves(m, origin, horizon)})
    columns = ["origin", "window_end", "available_on", "state_observed",
               *outlook.STATE, *TARGETS]
    return pd.DataFrame(rows, columns=columns).set_index("origin")


def completed_training(hist: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    train = outlook.training(hist, origin)
    return train[(train.index < origin) & (train["available_on"] <= origin)]


def combine(mean: pd.Series, model: pd.Series) -> pd.Series:
    return MEAN_WEIGHT * mean + STATE_WEIGHT * model


def fixed_shock() -> pd.Series:
    shock = pd.Series(0.0, index=TARGETS)
    shock["spx_ret"], shock["d_ust"] = math.log(0.9), 2.0
    return shock


@dataclass
class OriginForecast:
    vectors: pd.DataFrame
    audit: dict


def forecast_at(
    m: pd.DataFrame, origin: pd.Timestamp, horizon: int, hist: pd.DataFrame | None = None
) -> OriginForecast:
    """Forecast without needing the origin's subsequent target or any later row."""
    hist = target_history(m.loc[:origin], horizon) if hist is None else hist
    train = completed_training(hist, origin)
    if len(train) < NEIGHBOURS:
        raise ValueError(f"{origin.date()}: fewer than {NEIGHBOURS} completed training targets")
    state, cutoff = state_at(m, origin)
    model, mean = outlook.forecast(train, state, TARGETS)
    scales = train[outlook.STATE].std().replace(0.0, 1.0)
    centre = train[outlook.STATE].mean()
    standard = (train[outlook.STATE] - centre) / scales
    current = (state[outlook.STATE] - centre) / scales
    distance = ((standard - current) ** 2).sum(axis=1)
    # The index is chronological; stable sort breaks distance ties by earlier origin.
    near = distance.sort_values(kind="stable").index[:NEIGHBOURS]
    vectors = pd.DataFrame({
        "historical_mean": mean, "no_change": pd.Series(0.0, index=TARGETS),
        "fixed_shock": fixed_shock(), "clean_analog": train.loc[near, TARGETS].mean(),
        "state_model": model, "combination": combine(mean, model),
    }).T.loc[list(CANDIDATES), TARGETS]
    sigma = np.log(m.loc[:cutoff, "spx"]).diff().std() * math.sqrt(TRADING_DAYS)
    if not np.isfinite(vectors.to_numpy()).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Nonfinite forecast or invalid earlier-data equity volatility")
    audit = {
        "training_count": len(train), "latest_training_origin": str(train.index.max().date()),
        "latest_training_end": str(train["window_end"].max().date()),
        "latest_training_available": str(train["available_on"].max().date()),
        "observation_cutoff": str(cutoff.date()), "sigma_annual": float(sigma),
        "neighbours": [str(day.date()) for day in near],
        "state": {name: float(state[name]) for name in outlook.STATE},
        "training_origin_violations": int((train.index >= origin).sum()),
        "target_end_violations": int((train["window_end"] > origin).sum()),
        "availability_violations": int((train["available_on"] > origin).sum()),
        "state_date_violations": int((train["state_observed"] > train.index).sum()),
    }
    return OriginForecast(vectors, audit)


def eligible_origins(m: pd.DataFrame) -> pd.DatetimeIndex:
    ends = outlook.month_ends(m.loc[outlook.FIRST_ORIGIN:])
    return ends[m.index.get_indexer(ends) + max(HORIZONS) < len(m)]


def book_base(m: pd.DataFrame) -> Market:
    """Fixed 2019 valuation constants; deliberately separate from predictor inputs."""
    base = m.loc[:BOOK_DATE].iloc[-1]
    return Market("fixed 2019 book", float(base["aaa"]), float(base["baa"]), 0.0, 0.0,
                  0.0, float(base["usdinr"]), 0.0, float(base["ust10y"]), 0.0)


def value_vector(book: pd.DataFrame, base: Market, vector: pd.Series, sigma: float) -> tuple:
    """Use the unchanged E22 pricing path; no shock-history helper reads later data."""
    mk = Market("E32", base.aaa0, base.baa0, float(vector["d_aaa"]), float(vector["d_baa"]),
                -float(vector["spx_ret"]) / sigma, base.fx0, float(vector["fx_ret"]),
                base.rate0, float(vector["d_ust"]))
    result = revalue(book, None, 0, mk, rates="moved")
    return result.loss_usd, float(result.cet1_after)


def replay(m: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    m = prepare_market(m)
    origins, base, rows = eligible_origins(m), book_base(m), []
    if len(origins) < BLOCK_MONTHS:
        raise ValueError("Too few eligible origins for the registered monthly bootstrap")
    for horizon in HORIZONS:
        hist = target_history(m, horizon)
        for n, origin in enumerate(origins, 1):
            predicted = forecast_at(m, origin, horizon, hist)
            truncated = forecast_at(m.loc[:origin], origin, horizon)
            pd.testing.assert_frame_equal(predicted.vectors, truncated.vectors, check_exact=True)
            if predicted.audit != truncated.audit:
                raise AssertionError(f"Future-data invariance failed at {origin}")
            audit = predicted.audit
            if any(audit[key] for key in audit if key.endswith("violations")):
                raise AssertionError(f"Training timing violation at {origin}")
            actual = pd.Series(moves(m, origin, horizon))[TARGETS]
            realised_loss, realised_cet1 = value_vector(book, base, actual, audit["sigma_annual"])
            for candidate, vector in predicted.vectors.iterrows():
                loss, cet1 = value_vector(book, base, vector, audit["sigma_annual"])
                rows.append({
                    "origin": origin, "horizon": horizon, "candidate": candidate,
                    "window_end": m.index[m.index.get_loc(origin) + horizon],
                    **{key: value for key, value in audit.items() if key != "state"},
                    **{f"state_{key}": value for key, value in audit["state"].items()},
                    **{f"forecast_{key}": float(vector[key]) for key in TARGETS},
                    **{f"realised_{key}": float(actual[key]) for key in TARGETS},
                    "loss_usd": loss, "realised_loss_usd": realised_loss,
                    "loss_error_usd_m": abs(loss - realised_loss) / 1e6,
                    "cet1": cet1, "realised_cet1": realised_cet1,
                    "cet1_error_pp": 100 * abs(cet1 - realised_cet1),
                    "future_truncation_identical": True,
                })
            if n % 25 == 0 or n == len(origins):
                message = f"E32 H={horizon}: {n}/{len(origins)} origins; timing checks pass"
                print(message, flush=True)
    return pd.DataFrame(rows)


def bootstrap_indices(n: int) -> np.ndarray:
    if n < BLOCK_MONTHS:
        raise ValueError("Bootstrap needs at least one complete six-month block")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n - BLOCK_MONTHS + 1,
                          size=(RESAMPLES, math.ceil(n / BLOCK_MONTHS)))
    return (starts[:, :, None] + np.arange(BLOCK_MONTHS)).reshape(RESAMPLES, -1)[:, :n]


def paired_interval(difference: np.ndarray, indices: np.ndarray) -> list[float]:
    draws = np.asarray(difference)[indices].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def dm(difference: np.ndarray, horizon: int) -> dict:
    lag = math.ceil(horizon / 21)
    if np.all(difference == 0):
        return {"statistic": 0.0, "p_two_sided": 1.0, "newey_west_lag": lag}
    try:
        with np.errstate(divide="raise", invalid="raise"):
            statistic, p = outlook.diebold_mariano(difference, lag=lag)
        if not np.isfinite(statistic) or not np.isfinite(p):
            raise ValueError("Nonfinite DM result")
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return {"statistic": None, "p_two_sided": None, "newey_west_lag": lag,
                "note": "Degenerate Newey-West variance"}
    return {"statistic": float(statistic), "p_two_sided": float(p), "newey_west_lag": lag}


def basic_metrics(frame: pd.DataFrame, mean_mae: float) -> dict:
    mae = float(frame["loss_error_usd_m"].mean())
    return {
        "origins": len(frame), "loss_mae_usd_m": mae, "ratio_to_mean": mae / mean_mae,
        "cet1_mae_pp": float(frame["cet1_error_pp"].mean()),
        "target_mae": {key: float((frame[f"forecast_{key}"] - frame[f"realised_{key}"])
                                  .abs().mean()) for key in TARGETS},
    }


def summarise(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    results, years = {}, []
    for horizon, frame in predictions.groupby("horizon", sort=False):
        losses = frame.pivot(index="origin", columns="candidate", values="loss_error_usd_m")
        if losses.isna().any().any() or set(losses.columns) != set(CANDIDATES):
            raise ValueError("Candidates must have identical scored origins")
        indices = bootstrap_indices(len(losses))
        benchmark = losses["historical_mean"].mean()
        metrics = {}
        for candidate in CANDIDATES:
            subset = frame[frame["candidate"] == candidate]
            difference = (losses[candidate] - losses["historical_mean"]).to_numpy()
            metrics[candidate] = {
                **basic_metrics(subset, benchmark),
                "paired_difference_usd_m": float(difference.mean()),
                "paired_difference_95ci_usd_m": paired_interval(difference, indices),
                "diebold_mariano": dm(difference, int(horizon)),
            }
        for year, yearly in frame.groupby(frame["origin"].dt.year):
            mean_mae = yearly.loc[yearly["candidate"] == "historical_mean",
                                  "loss_error_usd_m"].mean()
            for candidate in CANDIDATES:
                subset = yearly[yearly["candidate"] == candidate]
                met = basic_metrics(subset, mean_mae)
                years.append({"year": int(year), "horizon": int(horizon),
                              "candidate": candidate,
                              **{k: v for k, v in met.items() if k != "target_mae"},
                              **{f"mae_{k}": v for k, v in met["target_mae"].items()}})
        checks = {key: int(frame[key].sum()) for key in frame if key.endswith("violations")}
        results[str(horizon)] = {
            "origins": len(losses), "first_origin": str(losses.index.min().date()),
            "last_origin": str(losses.index.max().date()),
            "last_target_end": str(frame["window_end"].max().date()),
            "training_count_range": [int(frame["training_count"].min()),
                                     int(frame["training_count"].max())],
            "truncation_checks_passed": int(frame.loc[frame["candidate"] == "historical_mean",
                                                      "future_truncation_identical"].sum()),
            "timing_violations": checks, "candidates": metrics,
        }
    return results, pd.DataFrame(years)


def combination_decision(primary: dict) -> str:
    combo = primary["candidates"]["combination"]
    improved = (combo["ratio_to_mean"] <= PASS_RATIO
                and combo["paired_difference_95ci_usd_m"][1] < 0)
    return "gain over the historical mean" if improved else "no gain over the historical mean"


def old_analog_diagnostic(clean_mae: float) -> dict:
    result = {"clean_monthly_analog_mae_usd_m": clean_mae,
              "warning": "Different origin selection, samples, availability and pricing; "
                         "these gaps do not isolate the causal effect of leakage."}
    for name, path in (("E14", "module_b_crisis_backtest.csv"),
                       ("E22", "e22_crisis_backtest.csv")):
        old = pd.read_csv(DATA / "outputs" / path)
        mae = float(old["predicted_loss_error"].mean() / 1e6)
        result[name] = {"file": path, "crises": len(old), "analog_mae_usd_m": mae,
                        "mean_mae_usd_m": float(old["naive_loss_error"].mean() / 1e6),
                        "clean_minus_old_mae_usd_m": clean_mae - mae,
                        "rates": "fixed" if name == "E14" else "moved"}
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path = OUTPUT) -> dict:
    predictions_path = output.with_name(f"{output.stem}_predictions.csv")
    years_path = output.with_name(f"{output.stem}_per_year.csv")
    # Refuse before reading data or computing anything, including when a sidecar exists.
    for path in (output, predictions_path, years_path):
        if path.exists():
            raise FileExistsError(f"Refusing to rerun or overwrite E32: {path}")
    rules_text = RULES.read_text().split("\n## Results\n", 1)[0]
    m = pd.read_csv(MARKET_PATH, parse_dates=["date"]).set_index("date")
    predictions = replay(m, load_book())
    results, years = summarise(predictions)
    result = {
        "experiment": "E32", "created_utc": datetime.now(UTC).isoformat(),
        "rules": "docs/E32_FORECAST_REPLAY.md",
        "pre_run_rules_sha256": hashlib.sha256(rules_text.encode()).hexdigest(),
        "pre_run_rules_text": rules_text,
        "availability": {"buffer_trading_sessions": AVAILABILITY_SESSIONS,
                         "release_calendar_verified": False, "historical_vintages": False,
                         "scope": "Chronological replay under assumed availability"},
        "pricing": {"book_date": BOOK_DATE, "rates": "moved", "obligor": None,
                    "sigma": "Expanding daily equity volatility through observation cutoff"},
        "combination_weights": {"historical_mean": MEAN_WEIGHT, "state_model": STATE_WEIGHT},
        "bootstrap": {"block_months": BLOCK_MONTHS, "resamples": RESAMPLES, "seed": SEED,
                      "method": "Paired non-circular moving blocks, percentile 95% interval",
                      "difference": "Candidate minus mean absolute loss error, USD millions"},
        "horizons": results, "decision": combination_decision(results["60"]),
        "decision_rule": "60-day combination ratio <= 0.85 and paired 95% CI wholly below 0",
        "engine_changed": False, "development_only": True,
        "old_vs_clean_analog": old_analog_diagnostic(
            results["60"]["candidates"]["clean_analog"]["loss_mae_usd_m"]),
    }
    inputs = [MARKET_PATH, BOOK_PATH, ROOT / "docs/DECISIONS.md",
              DATA / "outputs/module_b_crisis_backtest.csv",
              DATA / "outputs/e22_crisis_backtest.csv", Path(__file__),
              ROOT / "tests/test_e32_forecast_replay.py",
              *sorted((ROOT / "src/riskengine/module_b").glob("*.py"))]
    result["sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    output.parent.mkdir(parents=True, exist_ok=True)
    export = predictions.copy()
    export["neighbours"] = export["neighbours"].map(lambda values: ";".join(values))
    export.to_csv(predictions_path, index=False, mode="x")
    years.to_csv(years_path, index=False, mode="x")
    result["artifacts"] = {str(path.relative_to(ROOT)): sha256(path)
                           for path in (predictions_path, years_path)}
    # The result marker is written last. Once it exists, --run always refuses.
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with output.open("x") as handle:
        handle.write(encoded)
    print(f"Saved {output}\n{result['decision']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run once using local data only")
    args = parser.parse_args()
    if not args.run:
        parser.error("Specify --run; this evaluator never fetches data")
    try:
        run()
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
