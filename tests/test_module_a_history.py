"""E17 Module A history: synthetic data only."""

import numpy as np
import pandas as pd

from riskengine.module_a_history import block_boot, daily_ic, equal_targets, run, sharpe

DAYS = pd.bdate_range("2015-01-01", periods=300)
NAMES = list("ABCDEFG")


def test_ic_is_one_when_the_signal_is_tomorrows_return():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (300, 7)), index=DAYS, columns=NAMES)
    news = pd.DataFrame(True, index=DAYS, columns=NAMES)
    ic = daily_ic(rets, news, rets)
    assert np.allclose(ic, 1.0)


def test_ic_skips_days_with_too_few_names():
    rets = pd.DataFrame(np.tile(0.01 * np.arange(7.0), (3, 1)), index=DAYS[:3], columns=NAMES)
    news = pd.DataFrame(False, index=DAYS[:3], columns=NAMES)
    news.iloc[:, :4] = True  # only 4 companies have news
    assert daily_ic(rets, news, rets).empty


def test_equal_weight_spreads_over_live_names_only():
    live = pd.DataFrame(True, index=DAYS[:2], columns=NAMES)
    live.loc[DAYS[0], "G"] = False  # not listed yet
    w = equal_targets(live)
    assert np.isclose(w.loc[DAYS[0], "G"], 0) and np.isclose(w.loc[DAYS[0]].sum(), 1)
    assert np.isclose(w.loc[DAYS[1]].sum(), 1) and np.isclose(w.loc[DAYS[1], "A"], 1 / 7)


def test_slow_weights_trade_less():
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal(0, 0.01, (300, 7)), index=DAYS, columns=NAMES)
    target = pd.DataFrame(rng.dirichlet(np.ones(7), 300), index=DAYS, columns=NAMES)
    fast = run(target, rets, every=1, lam=1.0)
    slow = run(target, rets, every=5, lam=0.2)
    assert slow.turnover.iloc[1:].mean() < fast.turnover.iloc[1:].mean() / 4


def test_block_bootstrap_interval_covers_the_mean():
    x = np.random.default_rng(3).normal(0.001, 0.01, 1000)
    lo, hi = block_boot(x, np.mean)
    assert lo < x.mean() < hi
    assert sharpe(np.full(10, 0.0)) == 0.0


def test_fresh_test_runs_the_given_candidate_only():
    from riskengine.module_a_history import fresh_test

    rng = np.random.default_rng(4)
    days = pd.bdate_range("2021-09-01", periods=80)
    prices = pd.DataFrame(
        [
            {"date": d, "ticker": t, "close": 100 * (1 + 0.01 * rng.normal()) ** i}
            for t in NAMES
            for i, d in enumerate(days)
        ]
    )
    signals = pd.DataFrame(
        {
            "published_at": pd.to_datetime(np.repeat(days[:60], 7)).tz_localize("America/New_York"),
            "ticker": NAMES * 60,
            "sentiment": rng.normal(0, 0.5, 420),
        }
    )
    out = fresh_test(signals, prices, 5, 0.2, str(days[0].date()), str(days[-1].date()))
    assert out["candidate"] == {"rebalance_every": 5, "lambda": 0.2}
    assert set(out) == {"candidate", "ic", "portfolio"}
    assert out["portfolio"]["tilt"]["days"] == out["portfolio"]["equal"]["days"]
