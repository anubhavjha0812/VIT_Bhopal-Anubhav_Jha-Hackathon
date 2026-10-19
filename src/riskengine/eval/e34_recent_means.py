"""E34: fixed rolling/EWMA market-move forecasts. No tuning or engine wiring.

uv run python -m riskengine.eval.e34_recent_means --run
Rules: docs/E34_RECENT_MEANS.md. Existing results are read-only controls.
"""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval import e32_forecast_replay as e32
from riskengine.eval import e33_direct_loss as e33
from riskengine.module_b.book import load_book

RULES = e32.ROOT / "docs/E34_RECENT_MEANS.md"
OUTPUT = e32.DATA / "outputs/e34_recent_means.json"
ROLLING_TARGETS, HALF_LIFE_MONTHS = 60, 24
CANDIDATES = ("e32_mean", "e32_no_change", "rolling_60m", "ewma_h24m")
CONTROLS = ("e32_mean", "e32_no_change")


def ewma_weights(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    if len(index) == 0 or (index > origin).any():
        raise ValueError("EWMA needs nonempty earlier observations")
    age = origin.to_period("M").ordinal - index.to_period("M").asi8
    weights = np.exp2(-age / HALF_LIFE_MONTHS)
    return weights / weights.sum()


@dataclass
class RecentForecast:
    vectors: pd.DataFrame
    audit: dict


def forecast_at(m, horizon, origin, hist=None) -> RecentForecast:
    hist = e32.target_history(m.loc[:origin], horizon) if hist is None else hist
    train = e32.completed_training(hist, origin).sort_index()
    if len(train) < 3:
        raise ValueError("At least three completed targets are required")
    rolling = train.iloc[-ROLLING_TARGETS:]
    weights = ewma_weights(train.index, origin)
    exponential = weights @ train[e32.TARGETS].to_numpy(dtype=float)
    vectors = pd.DataFrame({
        "rolling_60m": rolling[e32.TARGETS].mean(),
        "ewma_h24m": pd.Series(exponential, index=e32.TARGETS),
    }).T
    _, cutoff = e32.state_at(m, origin)
    audit = {
        "training_count": len(train), "rolling_count": len(rolling),
        "rolling_first_origin": str(rolling.index.min().date()),
        "latest_training_origin": str(train.index.max().date()),
        "latest_training_end": str(train["window_end"].max().date()),
        "latest_training_available": str(train["available_on"].max().date()),
        "observation_cutoff": str(cutoff.date()),
        "sigma_annual": e33.equity_scale(m, cutoff),
        "ewma_effective_n": float(1 / (weights @ weights)),
        "ewma_latest_weight": float(weights[-1]), "ewma_oldest_weight": float(weights[0]),
        "ewma_weights_sha256": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest(),
        "training_origin_violations": int((train.index >= origin).sum()),
        "target_end_violations": int((train["window_end"] > origin).sum()),
        "availability_violations": int((train["available_on"] > origin).sum()),
        "state_date_violations": int((train["state_observed"] > train.index).sum()),
    }
    if not np.isfinite(vectors.to_numpy()).all():
        raise ValueError("Nonfinite recent-mean forecast")
    if any(audit[k] for k in audit if k.endswith("violations")):
        raise AssertionError("Recent-mean timing violation")
    return RecentForecast(vectors, audit)


def point_metrics(frame, mean_mae):
    return {**e33.metrics(frame, mean_mae),
            "target_mae": {t: float((frame[f"forecast_{t}"] - frame[f"realised_{t}"])
                                    .abs().mean()) for t in e32.TARGETS}}


def score_horizon(frame, candidates, controls, interval, horizon=None):
    """Shared scoring for event samples; callers fix resampling."""
    errors = frame.pivot(index="origin", columns="candidate", values="loss_error_usd_m")
    if errors.isna().any().any() or set(errors) != set(candidates):
        raise ValueError("All candidates must have identical origins")
    mean_mae = float(errors["e32_mean"].mean())
    result = {}
    for candidate in candidates:
        pairs = {}
        for control in controls:
            difference = (errors[candidate] - errors[control]).to_numpy()
            pairs[control] = {"difference_usd_m": float(difference.mean()),
                              "difference_95ci_usd_m": interval(difference)}
            if horizon is not None:
                pairs[control]["diebold_mariano"] = e32.dm(difference, horizon)
        result[candidate] = {
            **point_metrics(frame[frame["candidate"] == candidate], mean_mae), "paired": pairs}
    return result


def replay(m, book, controls):
    m = e32.prepare_market(m)
    price = e33.make_pricer(book, e32.book_base(m))
    origins, records, audits = e32.eligible_origins(m), [], []
    for horizon in e32.HORIZONS:
        hist = e32.target_history(m, horizon)
        frozen = controls[controls["horizon"] == horizon]
        expected = frozen[frozen["candidate"] == "e32_mean"].sort_values("origin")
        if not pd.DatetimeIndex(expected["origin"]).equals(origins):
            raise ValueError("Recent-mean origins differ from frozen E32")
        for n, origin in enumerate(origins, 1):
            predicted = forecast_at(m, horizon, origin, hist)
            prefix = forecast_at(m.loc[:origin], horizon, origin)
            pd.testing.assert_frame_equal(predicted.vectors, prefix.vectors, check_exact=True)
            if predicted.audit != prefix.audit:
                raise AssertionError("Recent-mean future-removal audit failed")
            audit = {"origin": str(origin.date()), "horizon": horizon,
                     "future_truncation_identical": True, **predicted.audit}
            audits.append(audit)
            prior = frozen[frozen["origin"] == origin].set_index("candidate")
            actual = e32.moves(m, origin, horizon)
            actual_loss, actual_cet1 = price(tuple(float(actual[t]) for t in e32.TARGETS),
                                            audit["sigma_annual"])
            reference = prior.loc["e32_mean"]
            delta = abs(actual_loss - reference["realised_loss_usd"])
            cet1_delta = 100 * abs(actual_cet1 - reference["realised_cet1"])
            if delta > 0.01 or cet1_delta > 1e-10:
                raise AssertionError("Recent means changed the E32 realised outcome")
            for candidate in CANDIDATES:
                if candidate in predicted.vectors.index:
                    vector = predicted.vectors.loc[candidate]
                    loss, cet1 = price(tuple(float(vector[t]) for t in e32.TARGETS),
                                       audit["sigma_annual"])
                else:
                    row = prior.loc[candidate]
                    vector = pd.Series({t: float(row[f"forecast_{t}"]) for t in e32.TARGETS})
                    loss, cet1 = float(row["loss_usd"]), float(row["cet1"])
                records.append({
                    "origin": origin, "horizon": horizon, "candidate": candidate,
                    "window_end": m.index[m.index.get_loc(origin) + horizon],
                    "loss_usd_m": loss / 1e6, "cet1_pp": 100 * cet1,
                    "realised_loss_usd_m": actual_loss / 1e6,
                    "realised_cet1_pp": 100 * actual_cet1,
                    "loss_error_usd_m": abs(loss - actual_loss) / 1e6,
                    "cet1_error_pp": 100 * abs(cet1 - actual_cet1),
                    **{f"forecast_{t}": float(vector[t]) for t in e32.TARGETS},
                    **{f"realised_{t}": float(actual[t]) for t in e32.TARGETS},
                    "future_truncation_identical": True,
                    "training_count": audit["training_count"],
                    "rolling_count": audit["rolling_count"],
                    "e32_actual_loss_difference_usd": float(delta),
                    "e32_actual_cet1_difference_pp": float(cet1_delta),
                })
            if n % 25 == 0 or n == len(origins):
                print(f"E34 H={horizon}: {n}/{len(origins)} origins; audits pass", flush=True)
    return pd.DataFrame(records), audits


def summarise(predictions):
    result, years = {}, []
    for horizon, frame in predictions.groupby("horizon", sort=False):
        n = frame["origin"].nunique()
        indices = e32.bootstrap_indices(n)

        def interval(difference, indices=indices):
            return e32.paired_interval(difference, indices)

        candidates = score_horizon(frame, CANDIDATES, CONTROLS, interval, int(horizon))
        result[str(horizon)] = {
            "origins": n, "first_origin": str(frame["origin"].min().date()),
            "last_origin": str(frame["origin"].max().date()), "candidates": candidates,
            "truncation_checks_passed": int(frame.loc[frame["candidate"] == "ewma_h24m",
                                                      "future_truncation_identical"].sum()),
            "max_e32_actual_loss_difference_usd": float(
                frame["e32_actual_loss_difference_usd"].max()),
            "max_e32_actual_cet1_difference_pp": float(
                frame["e32_actual_cet1_difference_pp"].max()),
        }
        for year, yearly in frame.groupby(frame["origin"].dt.year):
            baseline = yearly.loc[yearly["candidate"] == "e32_mean", "loss_error_usd_m"].mean()
            for candidate in CANDIDATES:
                metrics = point_metrics(yearly[yearly["candidate"] == candidate], baseline)
                years.append({"year": int(year), "horizon": int(horizon),
                              "candidate": candidate,
                              **{k: v for k, v in metrics.items() if k != "target_mae"},
                              **{f"mae_{t}": value for t, value in metrics["target_mae"].items()}})
    return result, pd.DataFrame(years)


def decision(primary):
    candidates = primary["candidates"]
    ewma = candidates["ewma_h24m"]
    conditions = {
        "ratio_to_mean_at_most_085": ewma["ratio_to_e32_mean"] <= e32.PASS_RATIO,
        "paired_ci_below_mean": ewma["paired"]["e32_mean"]["difference_95ci_usd_m"][1] < 0,
        "point_error_no_worse_than_no_change": ewma["loss_mae_usd_m"]
            <= candidates["e32_no_change"]["loss_mae_usd_m"],
    }
    passed = all(conditions.values())
    return {"conditions": conditions, "passed": passed,
            "verdict": "Development gain from EWMA" if passed else "no demonstrated gain from EWMA"}


def run(output=OUTPUT):
    paths = {name: output.with_name(f"{output.stem}_{name}.csv")
             for name in ("predictions", "per_year")}
    for path in (output, *paths.values()):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite or rerun E34: {path}")
    rules = RULES.read_text().split("\n## Results\n", 1)[0]
    controls, frozen = e33.load_frozen_controls()
    m = pd.read_csv(e32.MARKET_PATH, parse_dates=["date"]).set_index("date")
    predictions, audits = replay(m, load_book(), controls)
    summary, years = summarise(predictions)
    inputs = [e32.MARKET_PATH, e32.BOOK_PATH, e32.OUTPUT,
              e32.OUTPUT.with_name("e32_forecast_replay_predictions.csv"),
              Path(e32.__file__), Path(e33.__file__), Path(__file__),
              e32.ROOT / "tests/test_e34_recent_means.py",
              *sorted((e32.ROOT / "src/riskengine/module_b").glob("*.py"))]
    csvs = {"predictions": predictions.to_csv(index=False), "per_year": years.to_csv(index=False)}
    result = {
        "experiment": "E34", "created_utc": datetime.now(UTC).isoformat(),
        "rules": str(RULES.relative_to(e32.ROOT)), "pre_run_rules_text": rules,
        "pre_run_rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
        "fixed_parameters": {"rolling_targets": ROLLING_TARGETS,
                             "rolling_warm_up": "All eligible targets when fewer than 60",
                             "ewma_half_life_calendar_months": HALF_LIFE_MONTHS,
                             "ewma_weights": "Normalised 2**(-age_months/24)"},
        "horizons": summary, "decision": decision(summary["60"]), "audits": audits,
        "bootstrap": {**frozen["bootstrap"],
                      "difference": "Candidate minus the control named in each paired entry"},
        "pricing": frozen["pricing"], "availability": frozen["availability"],
        "development_only": True, "engine_changed": False,
        "sha256": {str(p.relative_to(e32.ROOT)): e32.sha256(p) for p in inputs},
        "artifact_sha256": {str(paths[k].relative_to(e32.ROOT)):
                            hashlib.sha256(content.encode()).hexdigest()
                            for k, content in csvs.items()},
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run once, offline; fixed parameters")
    if not parser.parse_args().run:
        parser.error("Specify --run; no tuning or fetch mode exists")
    try:
        run()
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
