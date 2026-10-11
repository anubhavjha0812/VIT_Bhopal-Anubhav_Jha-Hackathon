"""Module A over years, not weeks (rules: docs/DECISIONS.md, E17).

uv run python -m riskengine.module_a_history

1. Signal test: does a company's recent news sentiment rank tomorrow's returns? Daily Spearman
   IC across companies with news, with a moving-block bootstrap interval.
2. Portfolio test: four Module A candidates (rebalance every 1 or 5 days x move weights fully or
   20% of the way). Chosen on 2010-2016, scored once on 2017 - 10 Jun 2020.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskengine.module_a import COST_BPS, daily_signal, target_weights
from riskengine.paths import DATA

DEV = ("2010-01-01", "2016-12-31")
TEST = ("2017-01-01", "2020-06-10")
REPLAY = ("2020-04-09", "2020-06-10")  # already seen in the 43-day demo
TWEETS = ("2021-09-30", "2022-09-30")  # E17 fresh test: equinxx tweets
MIN_NAMES = 5
BLOCK, N_BOOT, SEED = 20, 1000, 0
CANDIDATES = [(k, lam) for k in (1, 5) for lam in (1.0, 0.2)]
OUT = DATA / "outputs" / "module_a_history.json"


def closes_from(prices: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot(index="date", columns="ticker", values="close")
    return wide.drop(columns=["^GSPC", "^VIX"], errors="ignore").sort_index()


def has_news(signals: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """A usable text in Module A's lookback window (its mean of 1.0 is 1 there, 0 elsewhere)."""
    return daily_signal(signals.assign(sentiment=1.0), days) > 0


def daily_ic(signal: pd.DataFrame, news: pd.DataFrame, rets_next: pd.DataFrame) -> pd.Series:
    """Spearman across companies with news and a next-day return, on days with >= MIN_NAMES."""
    out = {}
    for day in signal.index:
        ok = news.loc[day] & rets_next.loc[day].notna()
        if ok.sum() >= MIN_NAMES:
            s, r = signal.loc[day, ok], rets_next.loc[day, ok]
            if s.nunique() > 1:
                out[day] = spearmanr(s, r).statistic
    return pd.Series(out, dtype=float)


def block_boot(values: np.ndarray, stat, block: int = BLOCK, n: int = N_BOOT, seed: int = SEED):
    """95% moving-block bootstrap interval of stat(values[idx]) (values: days x ...)."""
    rng = np.random.default_rng(seed)
    t = len(values)
    starts = np.arange(max(1, t - block + 1))
    draws = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, t // block + 1)])
        draws.append(stat(values[idx[:t] % t]))
    return np.percentile(draws, [2.5, 97.5]).tolist()


@dataclass(frozen=True)
class Run:
    net: pd.Series  # daily net return, earned the next day
    turnover: pd.Series


def run(target: pd.DataFrame, rets_next: pd.DataFrame, every: int, lam: float) -> Run:
    """Path-dependent weights: on every `every`-th day move `lam` of the way to the target."""
    names = target.columns
    w = pd.Series(0.0, index=names)
    weights, turns = [], []
    for i, day in enumerate(target.index):
        live = rets_next.loc[day].notna()
        new = w.copy()
        if i % every == 0 or (w[~live] > 0).any():
            goal = target.loc[day].where(live, 0.0)
            goal = goal / goal.sum() if goal.sum() > 0 else goal
            new = (1 - lam) * w.where(live, 0.0) + lam * goal
            new = new / new.sum() if new.sum() > 0 else new
        turns.append((new - w).abs().sum())
        weights.append(new)
        w = new
    wdf = pd.DataFrame(weights, index=target.index)
    turnover = pd.Series(turns, index=target.index)
    gross = (wdf * rets_next.fillna(0.0)).sum(axis=1)
    return Run(gross - turnover * COST_BPS / 1e4, turnover)


def tilt_targets(signal: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Module A's tilt, computed over the companies that trade that day."""
    rows = []
    for day in signal.index:
        ok = live.loc[day]
        w = pd.Series(0.0, index=signal.columns)
        if ok.any():
            w[ok] = target_weights(signal.loc[[day], ok], "sentiment").iloc[0]
        rows.append(w)
    return pd.DataFrame(rows, index=signal.index)


def equal_targets(live: pd.DataFrame) -> pd.DataFrame:
    w = live.astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)) if np.std(x) > 0 else 0.0


def stats(net: pd.Series, turnover: pd.Series) -> dict:
    equity = (1 + net).cumprod()
    years = len(net) / 252
    return {
        "ann_return": float(equity.iloc[-1] ** (1 / years) - 1),
        "ann_vol": float(net.std() * np.sqrt(252)),
        "sharpe": sharpe(net.to_numpy()),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "avg_daily_turnover": float(turnover.mean()),
        "days": len(net),
    }


def compare(tilt: Run, equal: Run, start: str, end: str, drop: tuple | None = None) -> dict:
    """Tilt vs equal weight over [start, end] (optionally without the `drop` window)."""
    mask = (tilt.net.index >= start) & (tilt.net.index <= end)
    if drop:
        mask &= ~((tilt.net.index >= drop[0]) & (tilt.net.index <= drop[1]))
    a, b = tilt.net[mask], equal.net[mask]
    pair = np.column_stack([a.to_numpy(), b.to_numpy()])
    diff = sharpe(a.to_numpy()) - sharpe(b.to_numpy())
    ci = block_boot(pair, lambda v: sharpe(v[:, 0]) - sharpe(v[:, 1]))
    return {
        "tilt": stats(a, tilt.turnover[mask]),
        "equal": stats(b, equal.turnover[mask]),
        "sharpe_diff": diff,
        "sharpe_diff_ci95": ci,
        "by_year": {
            str(y): {
                "tilt": float((1 + a[a.index.year == y]).prod() - 1),
                "equal": float((1 + b[b.index.year == y]).prod() - 1),
            }
            for y in sorted(set(a.index.year))
        },
    }


def ic_summary(ic: pd.Series, start: str, end: str, drop: tuple | None = None) -> dict:
    s = ic[(ic.index >= start) & (ic.index <= end)]
    if drop:
        s = s[~((s.index >= drop[0]) & (s.index <= drop[1]))]
    return {
        "mean_ic": float(s.mean()),
        "ci95": block_boot(s.to_numpy(), np.mean),
        "days": len(s),
        "share_positive": float((s > 0).mean()),
    }


def setup(signals: pd.DataFrame, prices: pd.DataFrame, start: str, end: str):
    """Daily IC, tilt targets, next-day returns and the equal-weight run over [start, end]."""
    closes = closes_from(prices)
    closes = closes.loc[(closes.index >= start) & (closes.index <= end)]
    days = closes.index
    rets_next = closes.pct_change().shift(-1)  # return earned from day t's close to t+1
    live = closes.notna() & rets_next.notna()
    signal = daily_signal(signals, days).reindex(columns=closes.columns, fill_value=0.0)
    news = has_news(signals, days).reindex(columns=closes.columns, fill_value=False)
    ic = daily_ic(signal, news, rets_next)
    equal = run(equal_targets(live), rets_next, 1, 1.0)
    return ic, tilt_targets(signal, live), rets_next, equal


def fresh_test(
    signals: pd.DataFrame, prices: pd.DataFrame, every: int, lam: float, start: str, end: str
) -> dict:
    """The candidate chosen on dev, scored once on a fresh window (no choice made here)."""
    ic, tilt_t, rets_next, equal = setup(signals, prices, start, end)
    chosen = run(tilt_t, rets_next, every, lam)
    return {
        "candidate": {"rebalance_every": every, "lambda": lam},
        "ic": ic_summary(ic, start, end),
        "portfolio": compare(chosen, equal, start, end),
    }


def evaluate(signals: pd.DataFrame, prices: pd.DataFrame) -> dict:
    ic, tilt_t, rets_next, equal = setup(signals, prices, DEV[0], TEST[1])

    dev_scores = {}
    runs = {}
    for every, lam in CANDIDATES:
        r = run(tilt_t, rets_next, every, lam)
        runs[(every, lam)] = r
        d = compare(r, equal, *DEV)
        dev_scores[f"every {every} day(s), lambda {lam}"] = d["sharpe_diff"]
    best = max(CANDIDATES, key=lambda c: dev_scores[f"every {c[0]} day(s), lambda {c[1]}"])
    chosen = runs[best]
    return {
        "ic": {
            "dev": ic_summary(ic, *DEV),
            "test": ic_summary(ic, *TEST),
            "test_without_replay": ic_summary(ic, *TEST, drop=REPLAY),
            "by_year": {
                str(y): float(ic[ic.index.year == y].mean()) for y in sorted(set(ic.index.year))
            },  # noqa: E501
        },
        "dev_sharpe_diff": dev_scores,
        "chosen": {"rebalance_every": best[0], "lambda": best[1]},
        "dev": compare(chosen, equal, *DEV),
        "test": compare(chosen, equal, *TEST),
        "test_without_replay": compare(chosen, equal, *TEST, drop=REPLAY),
    }


def main() -> None:
    from riskengine.history import TWEET_SIGNALS_PATH, load_history
    from riskengine.ingest.prices import HISTORY_PATH, TWEETS_PATH, load_prices

    hist = load_history()
    prices = load_prices(HISTORY_PATH)
    subject = hist["role"] == "subject"
    variants = {
        "primary (rules applied)": hist[hist["why"].isna() & subject],
        "symbol only": hist[hist["by_symbol"] & subject],
        "name-linked, no rules": hist[subject],
    }
    res = {name: evaluate(df, prices) for name, df in variants.items()}
    if TWEET_SIGNALS_PATH.exists():  # E17 fresh test, with the candidate chosen on dev
        tweets = load_history(TWEET_SIGNALS_PATH)
        pick = res["primary (rules applied)"]["chosen"]
        res["fresh test: tweets 2021-22"] = fresh_test(
            tweets[tweets["role"] == "subject"], load_prices(TWEETS_PATH),
            pick["rebalance_every"], pick["lambda"], *TWEETS,
        )  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["primary (rules applied)"], indent=2))


if __name__ == "__main__":
    main()
