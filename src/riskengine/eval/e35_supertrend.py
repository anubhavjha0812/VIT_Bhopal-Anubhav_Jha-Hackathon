"""E35: one fixed Supertrend regime forecast, without engine changes.

--fetch downloads separate S&P 500 OHLC only; --run scores the cached bars once.
Rules: docs/E35_SUPERTREND.md. No close-only proxy or parameter search.
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
from riskengine.eval import e34_recent_means as e34
from riskengine.module_b.book import load_book

RULES = e32.ROOT / "docs/E35_SUPERTREND.md"
OUTPUT = e32.DATA / "outputs/e35_supertrend.json"
CACHE = e32.DATA / "cache/e35"
BARS_PATH, META_PATH = CACHE / "spx_ohlc.csv", CACHE / "spx_ohlc_meta.json"
PERIOD, MULTIPLIER, MIN_REGIME = 10, 3.0, 3
NAME = "supertrend_regime_mean"
CANDIDATES = (*e34.CONTROLS, NAME)


def prepare_ohlc(bars):
    if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.has_duplicates
            or bars.empty):
        raise ValueError("OHLC needs nonempty, unique dated bars")
    bars = bars.sort_index()[["high", "low", "close"]].astype(float)
    if (not np.isfinite(bars.to_numpy()).all() or (bars["low"] <= 0).any()
            or (bars["high"] < bars["low"]).any()
            or (bars["close"] > bars["high"]).any()
            or (bars["close"] < bars["low"]).any()):
        raise ValueError("Invalid real high/low/close bars; no filling permitted")
    return bars


def validate_alignment(m, bars):
    missing = m.index.difference(bars.index)
    if len(missing):
        raise ValueError(f"OHLC missing {len(missing)} existing market dates")
    difference = (m["spx"] - bars.loc[m.index, "close"]).abs()
    if difference.max() > 0.01:
        raise ValueError("OHLC close disagrees with existing market by more than $0.01")
    return {"market_dates": len(m), "missing_dates": 0,
            "max_close_difference": float(difference.max())}


def supertrend(bars, period=PERIOD, multiplier=MULTIPLIER):
    """Causal final bands; Wilder ATR seeded with an arithmetic TR mean."""
    if period < 1 or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Invalid fixed indicator parameters")
    bars = prepare_ohlc(bars)
    high, low, close = (bars[c].to_numpy() for c in ("high", "low", "close"))
    previous = np.r_[close[0], close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
    tr[0] = high[0] - low[0]
    n = len(bars)
    atr, upper, lower, line = (np.full(n, np.nan) for _ in range(4))
    direction = np.zeros(n, dtype=int)
    for i in range(period - 1, n):
        first = i == period - 1
        atr[i] = tr[:period].mean() if first else ((period - 1) * atr[i - 1] + tr[i]) / period
        midpoint = (high[i] + low[i]) / 2
        basic_up, basic_down = midpoint + multiplier * atr[i], midpoint - multiplier * atr[i]
        if first:
            upper[i], lower[i], direction[i] = basic_up, basic_down, -1
        else:
            upper[i] = (basic_up if basic_up < upper[i - 1] or close[i - 1] > upper[i - 1]
                        else upper[i - 1])
            lower[i] = (basic_down if basic_down > lower[i - 1] or close[i - 1] < lower[i - 1]
                        else lower[i - 1])
            if direction[i - 1] == -1:
                direction[i] = 1 if close[i] > upper[i] else -1
            else:
                direction[i] = -1 if close[i] < lower[i] else 1
        line[i] = lower[i] if direction[i] == 1 else upper[i]
    return pd.DataFrame({"true_range": tr, "atr": atr, "upper": upper, "lower": lower,
                         "line": line, "direction": direction}, index=bars.index)


def download_bars():
    import yfinance as yf

    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    raw = yf.download("^GSPC", start="1990-01-01", end="2024-01-01", auto_adjust=False,
                      threads=False, progress=False, timeout=30)
    if raw.empty:
        raise RuntimeError("No real S&P OHLC downloaded; no proxy will be scored")
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("^GSPC", axis=1, level=1)
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw.sort_index().rename_axis("date")


def fetch():
    if BARS_PATH.exists() or META_PATH.exists():
        raise FileExistsError("Refusing to overwrite E35 OHLC/provenance")
    CACHE.mkdir(parents=True, exist_ok=True)
    raw = download_bars()
    bars = prepare_ohlc(raw)
    m = e32.prepare_market(pd.read_csv(e32.MARKET_PATH, parse_dates=["date"]).set_index("date"))
    alignment = validate_alignment(m, bars)
    csv = raw.to_csv(index=True)
    meta = {"provider": "Yahoo Finance via yfinance", "ticker": "^GSPC",
            "requested_start": "1990-01-01", "requested_end_exclusive": "2024-01-01",
            "auto_adjust": False, "retrieved_utc": datetime.now(UTC).isoformat(),
            "yfinance_version": importlib.metadata.version("yfinance"),
            "rows": len(raw), "first_date": str(raw.index.min().date()),
            "last_date": str(raw.index.max().date()), "alignment": alignment,
            "csv_sha256": hashlib.sha256(csv.encode()).hexdigest(),
            "vintages": "Retrieved now; no verified historical publication/vintage data",
            "scoring_run": False}
    with BARS_PATH.open("x", encoding="utf-8", newline="") as handle:
        handle.write(csv)
    with META_PATH.open("x", encoding="utf-8", newline="") as handle:
        json.dump(meta, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved {len(raw)} real OHLC bars; no scoring run", flush=True)


@dataclass
class RegimeForecast:
    vectors: pd.DataFrame
    audit: dict


def forecast_at(m, bars, horizon, origin, hist=None, signal=None):
    hist = e32.target_history(m.loc[:origin], horizon) if hist is None else hist
    signal = supertrend(bars.loc[:origin]) if signal is None else signal
    train = e32.completed_training(hist, origin).sort_index()
    if len(train) < 3:
        raise ValueError("At least three completed training targets are required")
    _, cutoff = e32.state_at(m, origin)
    current = signal.loc[cutoff]
    directions = signal["direction"].reindex(pd.DatetimeIndex(train["state_observed"])).to_numpy()
    if current["direction"] not in (-1, 1) or not np.isin(directions, [-1, 1]).all():
        raise ValueError("Missing or uninitialised buffered Supertrend state")
    same = train[directions == current["direction"]]
    fallback = len(same) < MIN_REGIME
    selected = train if fallback else same
    vector = selected[e32.TARGETS].mean()
    audit = {
        "training_count": len(train), "same_regime_count": len(same),
        "selected_count": len(selected), "fallback": fallback,
        "training_up_count": int((directions == 1).sum()),
        "training_down_count": int((directions == -1).sum()),
        "regime_direction": int(current["direction"]),
        "indicator_line": float(current["line"]), "indicator_atr": float(current["atr"]),
        "selected_origins_sha256": hashlib.sha256(
            "\n".join(str(d.date()) for d in selected.index).encode()).hexdigest(),
        "latest_training_origin": str(train.index.max().date()),
        "latest_training_end": str(train["window_end"].max().date()),
        "latest_training_available": str(train["available_on"].max().date()),
        "latest_training_state": str(train["state_observed"].max().date()),
        "observation_cutoff": str(cutoff.date()), "indicator_bar_date": str(cutoff.date()),
        "sigma_annual": e33.equity_scale(m, cutoff),
        "training_origin_violations": int((train.index >= origin).sum()),
        "target_end_violations": int((train["window_end"] > origin).sum()),
        "availability_violations": int((train["available_on"] > origin).sum()),
        "state_date_violations": int((train["state_observed"] > train.index).sum()),
    }
    if any(audit[k] for k in audit if k.endswith("violations")):
        raise AssertionError("Supertrend timing violation")
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite regime forecast")
    return RegimeForecast(pd.DataFrame([vector], index=[NAME]), audit)


def replay(m, bars, book, controls):
    m, bars = e32.prepare_market(m), prepare_ohlc(bars)
    alignment = validate_alignment(m, bars)
    signal, origins = supertrend(bars), e32.eligible_origins(m)
    price = e33.make_pricer(book, e32.book_base(m))
    records, audits = [], []
    for horizon in e32.HORIZONS:
        hist = e32.target_history(m, horizon)
        frozen = controls[controls["horizon"] == horizon]
        expected = frozen[frozen["candidate"] == "e32_mean"].sort_values("origin")
        if not pd.DatetimeIndex(expected["origin"]).equals(origins):
            raise ValueError("Supertrend origins differ from frozen E32")
        for n, origin in enumerate(origins, 1):
            predicted = forecast_at(m, bars, horizon, origin, hist, signal)
            prefix = forecast_at(m.loc[:origin], bars.loc[:origin], horizon, origin)
            pd.testing.assert_frame_equal(predicted.vectors, prefix.vectors, check_exact=True)
            if predicted.audit != prefix.audit:
                raise AssertionError("Supertrend future-removal audit failed")
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
                raise AssertionError("Supertrend changed frozen actual outcomes")
            for candidate in CANDIDATES:
                if candidate == NAME:
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
                    "future_truncation_identical": True, "fallback": audit["fallback"],
                    "regime_direction": audit["regime_direction"],
                    "same_regime_count": audit["same_regime_count"],
                    "e32_actual_loss_difference_usd": float(delta),
                    "e32_actual_cet1_difference_pp": float(cet1_delta),
                })
            if n % 25 == 0 or n == len(origins):
                print(f"E35 H={horizon}: {n}/{len(origins)} origins; audits pass", flush=True)
    return pd.DataFrame(records), audits, alignment


def summarise(predictions):
    result, years = {}, []
    for horizon, frame in predictions.groupby("horizon", sort=False):
        indices = e32.bootstrap_indices(frame["origin"].nunique())
        candidates = e34.score_horizon(
            frame, CANDIDATES, e34.CONTROLS,
            lambda d, ix=indices: e32.paired_interval(d, ix), int(horizon))
        own = frame[frame["candidate"] == NAME]
        result[str(horizon)] = {
            "origins": len(own), "first_origin": str(own["origin"].min().date()),
            "last_origin": str(own["origin"].max().date()), "candidates": candidates,
            "fallbacks": int(own["fallback"].sum()),
            "up_origins": int((own["regime_direction"] == 1).sum()),
            "down_origins": int((own["regime_direction"] == -1).sum()),
            "truncation_checks_passed": int(own["future_truncation_identical"].sum()),
            "max_actual_loss_difference_usd": float(frame["e32_actual_loss_difference_usd"].max()),
            "max_actual_cet1_difference_pp": float(frame["e32_actual_cet1_difference_pp"].max()),
        }
        for year, yearly in frame.groupby(frame["origin"].dt.year):
            mean = yearly.loc[yearly["candidate"] == "e32_mean", "loss_error_usd_m"].mean()
            for name in CANDIDATES:
                metrics = e34.point_metrics(yearly[yearly["candidate"] == name], mean)
                years.append({"year": int(year), "horizon": int(horizon), "candidate": name,
                              **{k: v for k, v in metrics.items() if k != "target_mae"},
                              **{f"mae_{t}": v for t, v in metrics["target_mae"].items()}})
    return result, pd.DataFrame(years)


def decision(primary):
    own = primary["candidates"][NAME]
    conditions = {
        "ratio_to_mean_at_most_085": own["ratio_to_e32_mean"] <= e32.PASS_RATIO,
        "paired_ci_below_mean": own["paired"]["e32_mean"]["difference_95ci_usd_m"][1] < 0,
        "no_worse_than_no_change": own["loss_mae_usd_m"]
            <= primary["candidates"]["e32_no_change"]["loss_mae_usd_m"],
    }
    passed = all(conditions.values())
    return {"conditions": conditions, "passed": passed,
            "verdict": "Development gain from the fixed Supertrend regime forecast" if passed
            else "no demonstrated gain from the fixed Supertrend regime forecast"}


def run(output=OUTPUT):
    paths = {name: output.with_name(f"{output.stem}_{name}.csv")
             for name in ("predictions", "per_year")}
    for path in (output, *paths.values()):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite or rerun E35: {path}")
    rules = RULES.read_text().split("\n## Results\n", 1)[0]
    meta = json.loads(META_PATH.read_text())
    if e32.sha256(BARS_PATH) != meta["csv_sha256"]:
        raise ValueError("Cached OHLC differs from download provenance")
    controls, frozen = e33.load_frozen_controls()
    m = pd.read_csv(e32.MARKET_PATH, parse_dates=["date"]).set_index("date")
    bars = pd.read_csv(BARS_PATH, parse_dates=["date"]).set_index("date")
    predictions, audits, alignment = replay(m, bars, load_book(), controls)
    summary, years = summarise(predictions)
    inputs = [e32.MARKET_PATH, e32.BOOK_PATH, BARS_PATH, META_PATH, e32.OUTPUT,
              e32.OUTPUT.with_name("e32_forecast_replay_predictions.csv"),
              Path(e32.__file__), Path(e33.__file__), Path(e34.__file__), Path(__file__),
              e32.ROOT / "tests/test_e35_supertrend.py",
              *sorted((e32.ROOT / "src/riskengine/module_b").glob("*.py"))]
    csvs = {"predictions": predictions.to_csv(index=False), "per_year": years.to_csv(index=False)}
    result = {
        "experiment": "E35", "created_utc": datetime.now(UTC).isoformat(),
        "pre_run_rules_text": rules,
        "pre_run_rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
        "fixed_parameters": {"atr_length": PERIOD, "multiplier": MULTIPLIER,
                             "minimum_regime_targets": MIN_REGIME,
                             "candidate": NAME, "initial_direction": "down",
                             "atr_seed": "Mean of first 10 true ranges; Wilder recursion"},
        "ohlc_provenance": meta, "ohlc_alignment": alignment,
        "horizons": summary, "decision": decision(summary["60"]), "audits": audits,
        "bootstrap": {**frozen["bootstrap"], "difference": "Candidate minus named control"},
        "pricing": frozen["pricing"], "availability": frozen["availability"],
        "development_only": True, "engine_changed": False,
        "sha256": {str(p.relative_to(e32.ROOT)): e32.sha256(p) for p in inputs},
        "artifact_sha256": {str(paths[k].relative_to(e32.ROOT)):
                            hashlib.sha256(content.encode()).hexdigest()
                            for k, content in csvs.items()},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "scipy", "yfinance")},
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true", help="Download separate real OHLC only")
    mode.add_argument("--run", action="store_true", help="One offline score of cached OHLC")
    args = parser.parse_args()
    try:
        fetch() if args.fetch else run()
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
