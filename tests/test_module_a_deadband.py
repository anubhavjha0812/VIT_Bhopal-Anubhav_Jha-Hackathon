import numpy as np
import pandas as pd

from riskengine.module_a_deadband import decide, run_deadband
from riskengine.module_a_history import run


def toy(n=30, names=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"T{i}" for i in range(names)]
    raw = pd.DataFrame(rng.random((n, names)) + 0.2, index=idx, columns=cols)
    target = raw.div(raw.sum(axis=1), axis=0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, names)), index=idx, columns=cols)
    return target, rets


def test_zero_deadband_is_the_daily_full_move():
    target, rets = toy()
    a, b = run_deadband(target, rets, 0.0), run(target, rets, 1, 1.0)
    assert np.allclose(a.net, b.net) and np.allclose(a.turnover, b.turnover)


def test_wider_deadband_trades_less_and_stays_valid():
    target, rets = toy()
    tight, wide = run_deadband(target, rets, 0.005), run_deadband(target, rets, 0.05)
    assert wide.turnover.sum() < run_deadband(target, rets, 0.0).turnover.sum()
    assert wide.turnover.sum() <= tight.turnover.sum()


def test_a_dead_name_is_still_exited():
    target, rets = toy()
    rets.iloc[10, 0] = np.nan  # T0 has no next-day return on day 10
    held = run_deadband(target, rets, 0.5)
    assert held.net.notna().all()


def stats(sharpe, turnover, dd):
    return {"sharpe": sharpe, "avg_daily_turnover": turnover, "max_drawdown": dd}


def test_rule_needs_every_check():
    test = {"baseline": stats(1.0, 0.25, -0.30), "chosen": stats(1.1, 0.10, -0.30)}
    tweets = {"baseline": stats(0.5, 0.25, -0.1), "chosen": stats(0.6, 0.1, -0.1)}
    assert decide(test, tweets, 0.01)["decision"] == "deadband is a measured cost saver"
    worse = {"baseline": stats(1.0, 0.25, -0.30), "chosen": stats(1.1, 0.20, -0.30)}
    assert decide(worse, tweets, 0.01)["checks"]["turnover_down_30pct"] is False
    low = {"baseline": stats(0.5, 0.25, -0.1), "chosen": stats(0.4, 0.1, -0.1)}
    assert decide(test, low, 0.01)["decision"] == "not supported"
    assert decide(test, tweets, 0.0)["checks"]["chosen_d_above_0"] is False
