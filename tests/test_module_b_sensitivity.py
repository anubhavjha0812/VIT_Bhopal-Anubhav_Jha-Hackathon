"""Module B sensitivity checks on the real book and market data (docs/DECISIONS.md, M1).

The expected behaviour was fixed before these tests were written; a failure means the formula
is investigated, not the expectation changed.
"""

import numpy as np
import pytest

from riskengine.module_b.book import load_book
from riskengine.module_b.params import pd_1y, stressed_pd
from riskengine.module_b.revalue import CET1_FLOOR, revalue, reverse_stress
from riskengine.module_b.shock import EPISODES

MULTIPLES = [0.0, 0.5, 1.0, 1.5, 2.0]
EPS = 1e-6  # USD / ratio noise allowed in "does not fall"


@pytest.fixture(scope="module")
def book():
    return load_book()


@pytest.fixture(scope="module")
def obligor(book):
    return book.groupby("ticker")["ead"].sum().idxmax()  # the largest loan + bond exposure


def test_m1_1_no_shock_no_loss(book):
    r = revalue(book, None, 0, "covid2020", 0.0)
    assert r.loss_usd == pytest.approx(0.0, abs=1e-3)
    assert r.cet1_after == pytest.approx(r.cet1_before, abs=1e-12)
    sc = r.scenario
    assert (sc["rating_after"] == sc["rating_before"]).all()
    assert np.allclose(sc["spread_after"], sc["spread_before"])
    assert np.allclose(sc["pd_after"], sc["pd_before"])


@pytest.mark.parametrize("episode", sorted(EPISODES))
def test_m1_2_bigger_market_shock_bigger_loss(book, episode):
    runs = [revalue(book, None, 0, episode, k) for k in MULTIPLES]
    losses = [r.loss_usd for r in runs]
    cet1 = [r.cet1_after for r in runs]
    baa = [k * runs[0].market.d_baa for k in MULTIPLES]
    z = [k * runs[0].market.z for k in MULTIPLES]
    assert all(b >= a - EPS for a, b in zip(losses, losses[1:], strict=False)), losses
    assert all(b <= a + EPS for a, b in zip(cet1, cet1[1:], strict=False)), cet1
    assert all(b >= a for a, b in zip(baa, baa[1:], strict=False))
    assert all(b >= a for a, b in zip(z, z[1:], strict=False))


def test_m1_3_multiple_zero_removes_the_episode(book, obligor):
    r = revalue(book, obligor, 2, "covid2020", 0.0)
    assert r.fx_after == pytest.approx(r.fx_before)
    sc = r.scenario
    others = sc.drop(index=obligor)
    assert (others["rating_after"] == others["rating_before"]).all()
    assert sc.loc[obligor, "rating_after"] != sc.loc[obligor, "rating_before"]
    moved = others["spread_after"] - others["spread_before"]
    assert np.allclose(moved, others["contagion"])
    assert (others["contagion"] >= -EPS).all()
    assert np.allclose(others["pd_after"], others["pd_before"])  # z = 0: no market stress


def test_m1_4_more_notches_more_credit_risk(book, obligor):
    runs = [revalue(book, obligor, n, "covid2020", 0.0) for n in range(7)]
    pd_after = [r.scenario.loc[obligor, "pd_after"] for r in runs]
    ecl = [
        r.positions.query("ticker == @obligor and instrument == 'revolver'")["loss"].sum()
        for r in runs
    ]
    loss = [r.loss_usd for r in runs]
    rwa = [r.rwa_after_inr for r in runs]
    for series in (pd_after, ecl, loss, rwa):
        assert all(b >= a - EPS for a, b in zip(series, series[1:], strict=False)), series


@pytest.mark.parametrize("rating", ["AAA", "A", "BBB-", "BB", "B-"])
def test_m1_5_stressed_pd_rises_with_z(rating):
    pd = pd_1y(rating)
    zs = [0.5, 1.0, 2.0, 3.0]
    stressed = [stressed_pd(pd, z) for z in zs]
    assert all(b > a for a, b in zip(stressed, stressed[1:], strict=False))
    # the scenario takes max(normal PD, stressed PD), so it is never below the normal PD
    assert all(max(pd, s) >= pd for s in stressed)


def test_m1_6_default(book, obligor):
    r = revalue(book, obligor, 99, "covid2020", 0.0)
    assert r.scenario.loc[obligor, "rating_after"] == "D"
    pos = r.positions[r.positions["ticker"] == obligor]
    rev = pos[pos["instrument"] == "revolver"]
    assert (rev["stage"] == 3).all()
    assert np.allclose(-rev["value_after"], rev["lgd"] * rev["ead"])
    bonds = pos[pos["instrument"] == "bond"]
    assert np.allclose(bonds["value_after"], bonds["notional"] * (1 - bonds["lgd"]))


def test_m1_7_reverse_stress_lands_on_the_floor(book):
    k = reverse_stress(book, None, 0, "covid2020")
    assert np.isfinite(k)
    assert revalue(book, None, 0, "covid2020", k).cet1_after == pytest.approx(CET1_FLOOR, abs=5e-4)
    assert revalue(book, None, 0, "covid2020", 0.9 * k).cet1_after > CET1_FLOOR
