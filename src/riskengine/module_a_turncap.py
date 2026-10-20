"""E43: does a daily turnover cap cut Module A's cost?
Rules (fixed): docs/experiments/MODULE_A_TURNOVER_CAP.md

uv run python -m riskengine.module_a_turncap --run   # one scored run, refuses a repeat

Same data, windows and cost as E17/E42. Research only: no default, stored output or API number
changes.
"""

import argparse
import json
import os

import pandas as pd

from riskengine.module_a import COST_BPS
from riskengine.module_a_deadband import decide, window
from riskengine.module_a_history import DEV, REPLAY, TEST, TWEETS, Run, compare, setup
from riskengine.paths import DATA

GRID = (None, 0.05, 0.10, 0.15)  # None = no cap (the E17 chosen run)
OUT = DATA / "outputs" / "module_a_turncap.json"
LOCK = DATA / "outputs" / "module_a_turncap.run_lock"


def run_cap(target: pd.DataFrame, rets_next: pd.DataFrame, cap: float | None) -> Run:
    """Daily rebalance, but total weight moved in a day is at most `cap`."""
    w = pd.Series(0.0, index=target.columns)
    weights, turns = [], []
    for day in target.index:
        live = rets_next.loc[day].notna()
        goal = target.loc[day].where(live, 0.0)
        goal = goal / goal.sum() if goal.sum() > 0 else goal
        gap = float((goal - w).abs().sum())
        if cap is None or (w[~live] > 0).any() or gap <= cap:
            new = goal
        else:
            new = w + (goal - w) * (cap / gap)
        turns.append((new - w).abs().sum())
        weights.append(new)
        w = new
    wdf = pd.DataFrame(weights, index=target.index)
    turnover = pd.Series(turns, index=target.index)
    gross = (wdf * rets_next.fillna(0.0)).sum(axis=1)
    return Run(gross - turnover * COST_BPS / 1e4, turnover)


def evaluate(signals: pd.DataFrame, prices: pd.DataFrame, tweets) -> dict:
    _, tilt_t, rets_next, equal = setup(signals, prices, DEV[0], TEST[1])
    runs = {c: run_cap(tilt_t, rets_next, c) for c in GRID}
    key = lambda c: "none" if c is None else str(c)  # noqa: E731
    dev = {key(c): compare(r, equal, *DEV)["sharpe_diff"] for c, r in runs.items()}
    best = max(GRID, key=lambda c: dev[key(c)])
    out = {"grid": [key(c) for c in GRID], "dev_sharpe_diff_vs_equal": dev, "chosen_cap": key(best)}
    if best is None:
        out["decision"] = {
            "decision": "no cap beats the daily move on development",
            "defaults_changed": False,
        }
        return out
    out["dev_turnover"] = {
        key(c): float(r.turnover[DEV[0] : DEV[1]].mean()) for c, r in runs.items()
    }
    chosen, base = runs[best], runs[None]
    out["test"] = window(chosen, base, equal, *TEST)
    out["test_without_replay"] = window(chosen, base, equal, *TEST, drop=REPLAY)
    t_sig, t_prices = tweets
    _, t_tilt, t_rets, t_equal = setup(t_sig, t_prices, *TWEETS)
    out["tweets"] = window(
        run_cap(t_tilt, t_rets, best), run_cap(t_tilt, t_rets, None), t_equal, *TWEETS
    )
    out["decision"] = decide(out["test"], out["tweets"], 1.0)  # a cap was chosen
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="the one scored run")
    if not parser.parse_args().run:
        parser.error("use --run (one scored run)")
    from riskengine.history import TWEET_SIGNALS_PATH, load_history
    from riskengine.ingest.prices import HISTORY_PATH, TWEETS_PATH, load_prices

    os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY))  # refuses a repeat
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
