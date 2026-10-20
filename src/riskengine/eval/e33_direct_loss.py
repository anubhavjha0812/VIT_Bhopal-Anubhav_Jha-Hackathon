"""E33: one offline direct-loss median/quantile experiment, without engine changes.

uv run python -m riskengine.eval.e33_direct_loss --run

Rules: docs/experiments/E33_DIRECT_LOSS.md. All old E32 results are read-only controls.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from riskengine.eval import e32_forecast_replay as e32
from riskengine.module_b import outlook
from riskengine.module_b.book import load_book
from riskengine.module_b.shock import TRADING_DAYS

RULES = e32.ROOT / "docs/experiments/E33_DIRECT_LOSS.md"
OUTPUT = e32.DATA / "outputs/e33_direct_loss.json"
ALPHA, QUANTILE = 0.05, 0.5
LOSS, CET1 = "loss_usd_m", "cet1_pp"
OUTCOMES = [LOSS, CET1]
CANDIDATES = ("e32_mean", "e32_no_change", "direct_median", "direct_quantile")
CONTROLS = ("e32_mean", "e32_no_change", "direct_median")


def equity_scale(m: pd.DataFrame, observation_cutoff: pd.Timestamp) -> float:
    """Same earlier-data annual equity scale used by E32."""
    sigma = np.log(m.loc[:observation_cutoff, "spx"]).diff().std() * math.sqrt(TRADING_DAYS)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid earlier-data equity volatility")
    return float(sigma)


def make_pricer(book: pd.DataFrame, base):
    """Cache by the complete pricing inputs; the book and valuation base are fixed."""
    @cache
    def price(vector: tuple[float, ...], sigma: float) -> tuple[float, float]:
        return e32.value_vector(book, base, pd.Series(vector, index=e32.TARGETS), sigma)

    return price


def loss_history(m: pd.DataFrame, horizon: int, price) -> pd.DataFrame:
    """Completed historical targets valued with their own origin's earlier-data scale."""
    hist = e32.target_history(m, horizon).copy()
    losses, capital, scales = [], [], []
    for _, row in hist.iterrows():
        sigma = equity_scale(m, row["state_observed"])
        vector = tuple(float(row[target]) for target in e32.TARGETS)
        loss, cet1 = price(vector, sigma)
        losses.append(loss / 1e6)
        capital.append(100 * cet1)
        scales.append(sigma)
    hist[LOSS], hist[CET1], hist["sigma_annual"] = losses, capital, scales
    return hist


@dataclass
class DirectForecast:
    values: pd.DataFrame
    audit: dict


def fit_heads(train: pd.DataFrame, state: pd.Series) -> tuple[pd.DataFrame, dict]:
    """One fixed regularised median regression per scalar target; no parameter search."""
    if len(train) < 3:
        raise ValueError("At least three eligible training targets are required")
    centre = train[outlook.STATE].mean()
    scale = train[outlook.STATE].std(ddof=0).replace(0.0, 1.0)
    x = ((train[outlook.STATE] - centre) / scale).to_numpy(dtype=float)
    now = ((state[outlook.STATE] - centre) / scale).to_numpy(dtype=float).reshape(1, -1)
    if not np.isfinite(x).all() or not np.isfinite(now).all():
        raise ValueError("Nonfinite training state")
    forecasts, targets = {}, {}
    for target in OUTCOMES:
        y = train[target].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite training outcome")
        median = float(np.median(y))
        iqr = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))
        y_scale = iqr if iqr > 0 else 1.0
        model = QuantileRegressor(quantile=QUANTILE, alpha=ALPHA, solver="highs")
        model.fit(x, (y - median) / y_scale)
        prediction = float(model.predict(now)[0] * y_scale + median)
        forecasts[target] = {"direct_median": median, "direct_quantile": prediction}
        targets[target] = {
            "training_median": median, "training_iqr": iqr, "scaling_denominator": y_scale,
            "coefficients_scaled": [float(value) for value in model.coef_],
            "intercept_scaled": float(model.intercept_), "iterations": int(model.n_iter_),
        }
    values = pd.DataFrame(forecasts).loc[["direct_median", "direct_quantile"], OUTCOMES]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite direct forecast")
    audit = {"feature_centre": centre.to_dict(), "feature_scale": scale.to_dict(),
             "target_models": targets}
    return values, audit


def forecast_at(m: pd.DataFrame, hist: pd.DataFrame, origin: pd.Timestamp) -> DirectForecast:
    train = e32.completed_training(hist, origin)
    state, cutoff = e32.state_at(m, origin)
    values, audit = fit_heads(train, state)
    audit.update({
        "training_count": len(train), "latest_training_origin": str(train.index.max().date()),
        "latest_training_end": str(train["window_end"].max().date()),
        "latest_training_available": str(train["available_on"].max().date()),
        "observation_cutoff": str(cutoff.date()),
        "training_origin_violations": int((train.index >= origin).sum()),
        "target_end_violations": int((train["window_end"] > origin).sum()),
        "availability_violations": int((train["available_on"] > origin).sum()),
        "state_date_violations": int((train["state_observed"] > train.index).sum()),
    })
    if any(audit[key] for key in audit if key.endswith("violations")):
        raise AssertionError("Direct model training has a timing violation")
    return DirectForecast(values, audit)


def load_frozen_controls() -> tuple[pd.DataFrame, dict]:
    frozen = json.loads(e32.OUTPUT.read_text())
    csv_path = e32.OUTPUT.with_name("e32_forecast_replay_predictions.csv")
    for name, expected in frozen["artifacts"].items():
        if e32.sha256(e32.ROOT / name) != expected:
            raise ValueError(f"Frozen E32 artifact changed: {name}")
    for path in (e32.MARKET_PATH, e32.BOOK_PATH, Path(e32.__file__)):
        name = str(path.relative_to(e32.ROOT))
        if e32.sha256(path) != frozen["sha256"][name]:
            raise ValueError(f"Frozen E32 input changed: {name}")
    frame = pd.read_csv(csv_path, parse_dates=["origin", "window_end"])
    frame = frame[frame["candidate"].isin(["historical_mean", "no_change"])].copy()
    frame["candidate"] = frame["candidate"].map({
        "historical_mean": "e32_mean", "no_change": "e32_no_change"})
    return frame, frozen


def replay(m: pd.DataFrame, book: pd.DataFrame, controls: pd.DataFrame):
    m = e32.prepare_market(m)
    price = make_pricer(book, e32.book_base(m))
    origins, records, audits, histories = e32.eligible_origins(m), [], [], []
    for horizon in e32.HORIZONS:
        hist = loss_history(m, horizon, price)
        histories.append(hist.reset_index().assign(horizon=horizon))
        control = controls[controls["horizon"] == horizon]
        expected = control[control["candidate"] == "e32_mean"].sort_values("origin")
        if not pd.DatetimeIndex(expected["origin"]).equals(origins):
            raise ValueError("E33 origins differ from the frozen E32 controls")
        for n, origin in enumerate(origins, 1):
            predicted = forecast_at(m, hist, origin)
            prefix = m.loc[:origin]
            truncated = forecast_at(prefix, loss_history(prefix, horizon, price), origin)
            pd.testing.assert_frame_equal(predicted.values, truncated.values, check_exact=True)
            if predicted.audit != truncated.audit:
                raise AssertionError(f"Direct future-removal audit failed at {origin}")
            audit = {"origin": str(origin.date()), "horizon": horizon,
                     "future_truncation_identical": True, **predicted.audit}
            audits.append(audit)
            prior = control[control["origin"] == origin].set_index("candidate")
            sigma = equity_scale(m, pd.Timestamp(audit["observation_cutoff"]))
            actual_moves = e32.moves(m, origin, horizon)
            actual_vector = tuple(float(actual_moves[target]) for target in e32.TARGETS)
            actual_loss, actual_cet1 = price(actual_vector, sigma)
            reference = prior.loc["e32_mean"]
            loss_delta = abs(actual_loss - reference["realised_loss_usd"])
            cet1_delta = 100 * abs(actual_cet1 - reference["realised_cet1"])
            if loss_delta > 0.01 or cet1_delta > 1e-10:
                raise AssertionError("Direct experiment changed the frozen E32 realised outcome")
            values = {candidate: row.to_dict() for candidate, row in predicted.values.iterrows()}
            for candidate, row in prior.iterrows():
                values[candidate] = {LOSS: float(row["loss_usd"]) / 1e6,
                                     CET1: 100 * float(row["cet1"])}
            for candidate in CANDIDATES:
                predicted_loss, predicted_cet1 = values[candidate][LOSS], values[candidate][CET1]
                records.append({
                    "origin": origin, "horizon": horizon, "candidate": candidate,
                    "window_end": m.index[m.index.get_loc(origin) + horizon],
                    LOSS: predicted_loss, CET1: predicted_cet1,
                    "realised_loss_usd_m": actual_loss / 1e6,
                    "realised_cet1_pp": 100 * actual_cet1,
                    "loss_error_usd_m": abs(predicted_loss - actual_loss / 1e6),
                    "cet1_error_pp": abs(predicted_cet1 - 100 * actual_cet1),
                    "training_count": audit["training_count"],
                    "latest_training_available": audit["latest_training_available"],
                    "observation_cutoff": audit["observation_cutoff"],
                    "future_truncation_identical": True,
                    "e32_actual_loss_difference_usd": float(loss_delta),
                    "e32_actual_cet1_difference_pp": float(cet1_delta),
                })
            if n % 25 == 0 or n == len(origins):
                print(f"E33 H={horizon}: {n}/{len(origins)} origins; audits pass", flush=True)
    return pd.DataFrame(records), pd.concat(histories, ignore_index=True), audits


def metrics(frame: pd.DataFrame, mean_mae: float) -> dict:
    loss_mae = float(frame["loss_error_usd_m"].mean())
    return {"origins": len(frame), "loss_mae_usd_m": loss_mae,
            "ratio_to_e32_mean": loss_mae / mean_mae,
            "cet1_mae_pp": float(frame["cet1_error_pp"].mean())}


def summarise(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    results, years = {}, []
    for horizon, frame in predictions.groupby("horizon", sort=False):
        errors = frame.pivot(index="origin", columns="candidate", values="loss_error_usd_m")
        if errors.isna().any().any() or set(errors) != set(CANDIDATES):
            raise ValueError("All E33 candidates must share identical origins")
        indices = e32.bootstrap_indices(len(errors))
        mean_mae = float(errors["e32_mean"].mean())
        candidates = {}
        for candidate in CANDIDATES:
            part = frame[frame["candidate"] == candidate]
            paired = {}
            for control in CONTROLS:
                difference = (errors[candidate] - errors[control]).to_numpy()
                paired[control] = {
                    "difference_usd_m": float(difference.mean()),
                    "difference_95ci_usd_m": e32.paired_interval(difference, indices),
                    "diebold_mariano": e32.dm(difference, int(horizon)),
                }
            candidates[candidate] = {**metrics(part, mean_mae), "paired": paired}
        for year, yearly in frame.groupby(frame["origin"].dt.year):
            baseline = yearly.loc[yearly["candidate"] == "e32_mean", "loss_error_usd_m"].mean()
            for candidate in CANDIDATES:
                years.append({"year": int(year), "horizon": int(horizon),
                              "candidate": candidate,
                              **metrics(yearly[yearly["candidate"] == candidate], baseline)})
        results[str(horizon)] = {
            "origins": len(errors), "first_origin": str(errors.index.min().date()),
            "last_origin": str(errors.index.max().date()), "candidates": candidates,
            "truncation_checks_passed": int(frame.loc[frame["candidate"] == "direct_quantile",
                                                      "future_truncation_identical"].sum()),
            "max_e32_actual_loss_difference_usd": float(
                frame["e32_actual_loss_difference_usd"].max()),
            "max_e32_actual_cet1_difference_pp": float(
                frame["e32_actual_cet1_difference_pp"].max()),
        }
    return results, pd.DataFrame(years)


def decision(primary: dict) -> dict:
    candidates = primary["candidates"]
    quantile = candidates["direct_quantile"]
    conditions = {
        "ratio_to_mean_at_most_085": quantile["ratio_to_e32_mean"] <= e32.PASS_RATIO,
        "paired_ci_below_mean": quantile["paired"]["e32_mean"]["difference_95ci_usd_m"][1] < 0,
        "paired_ci_below_median": quantile["paired"]["direct_median"]
                                      ["difference_95ci_usd_m"][1] < 0,
        "point_error_no_worse_than_no_change": quantile["loss_mae_usd_m"]
            <= candidates["e32_no_change"]["loss_mae_usd_m"],
    }
    passed = all(conditions.values())
    return {"conditions": conditions, "passed": passed,
            "verdict": "Development gain from direct quantile model" if passed else
                       "no demonstrated gain from the direct quantile model"}


def artifact_paths(output: Path) -> dict[str, Path]:
    return {name: output.with_name(f"{output.stem}_{name}.csv")
            for name in ("predictions", "training", "per_year")}


def run(output: Path = OUTPUT) -> dict:
    paths = artifact_paths(output)
    for path in [output, *paths.values()]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite or rerun E33: {path}")
    rules_text = RULES.read_text().split("\n## Results\n", 1)[0]
    controls, frozen = load_frozen_controls()
    m = pd.read_csv(e32.MARKET_PATH, parse_dates=["date"]).set_index("date")
    predictions, history, audits = replay(m, load_book(), controls)
    results, years = summarise(predictions)
    csvs = {"predictions": predictions.to_csv(index=False), "training": history.to_csv(index=False),
            "per_year": years.to_csv(index=False)}
    inputs = [e32.MARKET_PATH, e32.BOOK_PATH, e32.OUTPUT,
              e32.OUTPUT.with_name("e32_forecast_replay_predictions.csv"),
              Path(e32.__file__), Path(__file__),
              e32.ROOT / "tests/test_e33_direct_loss.py",
              *sorted((e32.ROOT / "src/riskengine/module_b").glob("*.py"))]
    result = {
        "experiment": "E33", "created_utc": datetime.now(UTC).isoformat(),
        "rules": str(RULES.relative_to(e32.ROOT)),
        "pre_run_rules_sha256": hashlib.sha256(rules_text.encode()).hexdigest(),
        "pre_run_rules_text": rules_text,
        "fixed_model": {"quantile": QUANTILE, "alpha": ALPHA, "solver": "highs",
                        "state": outlook.STATE, "targets": OUTCOMES,
                        "target_scaling": "Training median and IQR; zero IQR scale 1"},
        "horizons": results, "decision": decision(results["60"]), "audits": audits,
        "bootstrap": frozen["bootstrap"], "pricing": frozen["pricing"],
        "availability": frozen["availability"], "development_only": True,
        "engine_changed": False, "coherent_market_scenario": False,
        "target_meaning": "Loss USD millions and CET1 percentage points on the fixed 2019 book",
        "sha256": {str(path.relative_to(e32.ROOT)): e32.sha256(path) for path in inputs},
        "artifact_sha256": {str(paths[key].relative_to(e32.ROOT)):
                            hashlib.sha256(content.encode()).hexdigest()
                            for key, content in csvs.items()},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "scipy", "scikit-learn")},
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, content in csvs.items():
        with paths[name].open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    with output.open("x", encoding="utf-8", newline="") as handle:
        handle.write(encoded)
    print(f"Saved {output}\n{result['decision']['verdict']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run once, offline, on the fixed book")
    args = parser.parse_args()
    if not args.run:
        parser.error("Specify --run; no fetch or tuning mode exists")
    try:
        run()
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
