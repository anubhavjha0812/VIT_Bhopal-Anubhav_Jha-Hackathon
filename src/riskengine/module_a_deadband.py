"""E42: does a trade deadband cut Module A's cost? Rules: docs/MODULE_A_DEADBAND.md (fixed).

uv run python -m riskengine.module_a_deadband --run   # one scored run, refuses a repeat

Same data, windows and cost as E17 (module_a_history). Research only: no default, stored
output or API number changes.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from riskengine.module_a import cap_weights
from riskengine.module_a_history import (
    DEV,
    REPLAY,
    TEST,
    TWEETS,
    Run,
    block_boot,
    compare,
    setup,
    sharpe,
)
from riskengine.paths import DATA

GRID = (0.0, 0.005, 0.01, 0.02)
MIN_TURNOVER_CUT, MAX_DRAWDOWN_WORSE = 0.30, 0.01
OUT = DATA / "outputs" / "module_a_deadband.json"
LOCK = DATA / "outputs" / "module_a_deadband.run_lock"


def run_deadband(target: pd.DataFrame, rets_next: pd.DataFrame, d: float) -> Run:
    """Daily rebalance, but a name trades only if |target - current| > d.

    d = 0 is the E17 chosen run (every day, full move) exactly.
    """
    from riskengine.module_a import COST_BPS

    w = pd.Series(0.0, index=target.columns)
    weights, turns = [], []
    for day in target.index:
        live = rets_next.loc[day].notna()
        goal = target.loc[day].where(live, 0.0)
        goal = goal / goal.sum() if goal.sum() > 0 else goal
        if d == 0 or (w[~live] > 0).any():
            new = goal
        else:
            new = w.where(np.abs(goal - w) <= d, goal)
            if new.sum() > 0:
                new = new.copy()
                new[live] = cap_weights(new[live])
        turns.append((new - w).abs().sum())
        weights.append(new)
        w = new
    wdf = pd.DataFrame(weights, index=target.index)
    turnover = pd.Series(turns, index=target.index)
    gross = (wdf * rets_next.fillna(0.0)).sum(axis=1)
    return Run(gross - turnover * COST_BPS / 1e4, turnover)


def paired_ci(a: pd.Series, b: pd.Series, start: str, end: str, drop: tuple | None = None):
    mask = (a.index >= start) & (a.index <= end)
    if drop:
        mask &= ~((a.index >= drop[0]) & (a.index <= drop[1]))
    pair = np.column_stack([a[mask].to_numpy(), b[mask].to_numpy()])
    diff = sharpe(pair[:, 0]) - sharpe(pair[:, 1])
    return diff, block_boot(pair, lambda v: sharpe(v[:, 0]) - sharpe(v[:, 1]))


def decide(test: dict, tweets: dict, chosen: float) -> dict:
    """The fixed rule; `test`/`tweets` hold stats for the chosen d and the d=0 baseline."""
    base, new = test["baseline"], test["chosen"]
    cut = 1 - new["avg_daily_turnover"] / base["avg_daily_turnover"]
    checks = {
        "chosen_d_above_0": chosen > 0,
        "turnover_down_30pct": cut >= MIN_TURNOVER_CUT,
        "test_sharpe_higher": new["sharpe"] > base["sharpe"],
        "drawdown_not_worse_by_1pt": (
            new["max_drawdown"] >= base["max_drawdown"] - MAX_DRAWDOWN_WORSE
        ),
        "tweets_sharpe_not_lower": tweets["chosen"]["sharpe"] >= tweets["baseline"]["sharpe"],
    }
    verdict = "deadband is a measured cost saver" if all(checks.values()) else "not supported"
    return {"checks": checks, "turnover_cut": float(cut), "decision": verdict,
            "defaults_changed": False}  # fmt: skip


def window(tilt: Run, base: Run, equal: Run, start: str, end: str, drop=None) -> dict:
    out = {"chosen": compare(tilt, equal, start, end, drop)["tilt"]}
    out["baseline"] = compare(base, equal, start, end, drop)["tilt"]
    out["equal"] = compare(tilt, equal, start, end, drop)["equal"]
    diff, ci = paired_ci(tilt.net, base.net, start, end, drop)
    out["sharpe_diff_vs_baseline"], out["sharpe_diff_vs_baseline_ci95"] = diff, ci
    return out


def evaluate(signals: pd.DataFrame, prices: pd.DataFrame, tweets=None) -> dict:
    _, tilt_t, rets_next, equal = setup(signals, prices, DEV[0], TEST[1])
    runs = {d: run_deadband(tilt_t, rets_next, d) for d in GRID}
    dev = {str(d): compare(r, equal, *DEV)["sharpe_diff"] for d, r in runs.items()}
    best = max(GRID, key=lambda d: dev[str(d)])
    out = {"grid": list(GRID), "dev_sharpe_diff_vs_equal": dev, "chosen_d": best}
    if best == 0:
        out["decision"] = {"decision": "no deadband beats the daily move on development",
                           "defaults_changed": False}  # fmt: skip
        return out
    out["dev_turnover"] = {
        str(d): float(r.turnover[DEV[0] : DEV[1]].mean()) for d, r in runs.items()
    }
    chosen, base = runs[best], runs[0.0]
    out["test"] = window(chosen, base, equal, *TEST)
    out["test_without_replay"] = window(chosen, base, equal, *TEST, drop=REPLAY)
    t_sig, t_prices = tweets
    _, t_tilt, t_rets, t_equal = setup(t_sig, t_prices, *TWEETS)
    out["tweets"] = window(
        run_deadband(t_tilt, t_rets, best), run_deadband(t_tilt, t_rets, 0.0), t_equal, *TWEETS
    )
    out["decision"] = decide(out["test"], out["tweets"], best)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="the one scored run")
    if not parser.parse_args().run:
        parser.error("use --run (one scored run)")
    from riskengine.history import TWEET_SIGNALS_PATH, load_history
    from riskengine.ingest.prices import HISTORY_PATH, TWEETS_PATH, load_prices

    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # refuses a repeat
    os.close(fd)
    hist = load_history()
    primary = hist[hist["why"].isna() & (hist["role"] == "subject")]
    tweets = load_history(TWEET_SIGNALS_PATH)
    res = evaluate(
        primary,
        load_prices(HISTORY_PATH),
        (tweets[tweets["role"] == "subject"], load_prices(TWEETS_PATH)),
    )
    with OUT.open("x") as f:
        f.write(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
