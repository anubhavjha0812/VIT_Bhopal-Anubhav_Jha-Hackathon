import math

import pytest

from riskengine.module_b.book import build_book
from riskengine.module_b.revalue import (
    CET1_FLOOR,
    CET1_START,
    bond_price,
    duration_convexity,
    ecl,
    irb_rwa,
    market_risk_rwa,
    revalue,
    reverse_stress,
    sa_ccr_ead,
    stage,
)


@pytest.fixture(scope="module")
def book():
    return build_book()


def test_irb_matches_basel_benchmark():
    # BCBS: PD 1%, LGD 45%, M 2.5 → risk weight 92.32%
    assert irb_rwa(0.01, 0.45, 2.5, 1.0) == pytest.approx(0.9232, abs=1e-4)
    assert irb_rwa(1.0, 0.45, 2.5, 1.0) == 0.0  # defaulted: covered by provisions


def test_par_bond_and_convexity_beat_duration_alone():
    c, y, t = 0.04, 0.04, 7.0
    assert bond_price(c, y, t) == pytest.approx(1.0)
    d, cx = duration_convexity(c, y, t)
    dy = 0.02  # 200 bp
    full = bond_price(c, y + dy, t) - 1
    with_convexity = -d * dy + 0.5 * cx * dy**2
    assert abs(with_convexity - full) < abs(-d * dy - full) / 5
    assert 5 < d < 7 and cx > d  # a sanity range for a 7-year bond


def test_staging_rules():
    assert stage("A", "A-") == 1
    assert stage("A", "BBB") == 2  # 3 notches
    assert stage("BBB-", "BB+") == 2  # fallen angel
    assert stage("BB-", "B+") == 1  # junk at the start, 1 notch
    assert stage("B-", "D") == 3


def test_ecl_by_stage():
    assert ecl(0.01, 0.45, 100, 5, 1) == pytest.approx(0.45)
    assert ecl(0.01, 0.45, 100, 5, 2) == pytest.approx(100 * 0.45 * (1 - 0.99**5))
    assert ecl(0.01, 0.45, 100, 5, 3) == pytest.approx(45)


def test_market_risk_rises_with_maturity_and_junk():
    assert market_risk_rwa("A", 1, 100) < market_risk_rwa("A", 10, 100)
    a, bb, b = (market_risk_rwa(r, 5, 100) for r in ("A", "BB", "B"))
    assert a < bb < b
    assert market_risk_rwa("A", 5, 100) == pytest.approx(12.5 * (0.016 + 0.0275) * 100)


def test_sa_ccr_ead():
    assert sa_ccr_ead(100, 5) == pytest.approx(1.4 * 0.005 * (1 - math.exp(-0.25)) / 0.05 * 100)


def test_nothing_happens_nothing_changes(book):
    r = revalue(book, None, 0, multiple=0.0)
    assert r.loss_usd == pytest.approx(0, abs=1e-6)
    assert r.cet1_after == pytest.approx(r.cet1_before) == pytest.approx(CET1_START)


def test_bought_protection_gains_and_sold_loses_in_stress(book):
    r = revalue(book, None, 0, multiple=1.0)
    by = r.positions.groupby("instrument")["loss"].sum()
    assert by["cds_bought"] < 0 < by["cds_sold"]  # negative loss = gain
    assert by["bond"] > 0


def test_default_wipes_to_recovery_and_stage_3(book):
    r = revalue(book, "TSLA", 99, multiple=0.0)
    tsla = r.positions[r.positions["ticker"] == "TSLA"]
    bonds = tsla[tsla["instrument"] == "bond"]
    assert bonds["value_after"].sum() == pytest.approx(0.55 * bonds["notional"].sum())
    assert (tsla.loc[tsla["instrument"] == "revolver", "stage"] == 3).all()
    bought = tsla[tsla["instrument"] == "cds_bought"]
    assert bought["value_after"].tolist() == pytest.approx((0.45 * bought["notional"]).tolist())


def test_alert_lowers_cet1_and_rwa_moves_with_the_rupee(book):
    r = revalue(book, "BA", 1, multiple=1.0)
    assert r.cet1_after < r.cet1_before
    assert r.fx_after > r.fx_before
    assert r.rwa_after_inr == pytest.approx(r.positions["rwa_after"].sum() * r.fx_after)


def test_downgrade_raises_rwa_market_alone_uses_long_run_pd(book):
    alone = revalue(book, None, 0, multiple=1.0).positions
    cut = revalue(book, "BA", 3, multiple=0.0).positions
    rev = alone[alone["instrument"] == "revolver"]
    assert rev["rwa_after"].tolist() == pytest.approx(rev["rwa_before"].tolist())
    ba = cut[(cut["ticker"] == "BA") & (cut["instrument"] == "revolver")]
    assert (ba["rwa_after"] > ba["rwa_before"]).all()


def test_reverse_stress_lands_on_the_floor(book):
    k = reverse_stress(book, "BA", 1)
    assert 0 < k < 20
    assert revalue(book, "BA", 1, multiple=k).cet1_after == pytest.approx(CET1_FLOOR, abs=1e-3)


@pytest.mark.parametrize("share", [0.001, 0.01, 0.02])
def test_cet1_change_does_not_depend_on_the_half_percent(book, share):
    """E14 sensitivity: the 0.5% share sets dollar sizes, not the CET1 change."""
    base = revalue(book, "BA", 1, multiple=1.0)
    other = revalue(build_book(share), "BA", 1, multiple=1.0)
    assert other.cet1_after == pytest.approx(base.cet1_after, abs=1e-3)  # within 0.1 point
