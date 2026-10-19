import numpy as np
import pandas as pd

from riskengine.module_a_history import run
from riskengine.module_a_turncap import run_cap


def toy(n=30, names=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"T{i}" for i in range(names)]
    raw = pd.DataFrame(rng.random((n, names)) + 0.2, index=idx, columns=cols)
    target = raw.div(raw.sum(axis=1), axis=0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, names)), index=idx, columns=cols)
    return target, rets


def test_no_cap_is_the_daily_full_move():
    target, rets = toy()
    a, b = run_cap(target, rets, None), run(target, rets, 1, 1.0)
    assert np.allclose(a.net, b.net) and np.allclose(a.turnover, b.turnover)


def test_cap_limits_daily_turnover_after_the_first_day():
    target, rets = toy()
    r = run_cap(target, rets, 0.05)
    assert (r.turnover.iloc[1:] <= 0.05 + 1e-12).all()
    assert r.turnover.sum() < run_cap(target, rets, None).turnover.sum()


def test_a_dead_name_is_still_exited():
    target, rets = toy()
    rets.iloc[10, 0] = np.nan
    assert run_cap(target, rets, 0.01).net.notna().all()
