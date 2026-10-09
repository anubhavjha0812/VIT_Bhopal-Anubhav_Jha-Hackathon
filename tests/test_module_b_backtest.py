import numpy as np
import pandas as pd
import pytest

from riskengine.module_b.backtest import WINDOW, as_market, find_episodes, moves


def fake_market(baa: list[float]) -> pd.DataFrame:
    n = len(baa)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"baa": baa, "aaa": 1.0, "vix": 20.0, "spx": 100.0, "usdinr": 50.0}, idx)


def test_episode_rule_finds_a_widening_and_skips_overlaps():
    baa = [2.0] * 10 + [3.0] * 200  # +100 bp on day 10, then flat
    eps = find_episodes(fake_market(baa))
    assert len(eps) == 1
    start, end = eps[0]
    assert start == pd.Timestamp("2000-01-03")
    assert len(pd.bdate_range(start, end)) == WINDOW + 1


def test_no_episode_below_75_bp():
    assert find_episodes(fake_market([2.0] * 50 + [2.7] * 100)) == []


def test_moves_and_market_from_moves():
    m = fake_market([2.0] * 3 + [2.5] * 10)
    m.loc[m.index[5], "spx"] = 90.0
    mv = moves(m, m.index[0], 5)
    assert mv["d_baa"] == pytest.approx(0.5)
    assert mv["spx_ret"] == pytest.approx(np.log(0.9))
    mk = as_market("t", mv, sigma_1y=0.2)
    assert mk.d_baa == pytest.approx(0.5)
    assert mk.z == pytest.approx(-np.log(0.9) / 0.2)  # a fall is a positive z
