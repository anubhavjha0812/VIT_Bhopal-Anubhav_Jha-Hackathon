"""E22 pass checks: the interest-rate shock (docs/DECISIONS.md, E22, fixed before the code)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from riskengine.module_b.backtest import ALERTS_PATH, CRISES_PATH, alert_check, crisis_backtest
from riskengine.module_b.book import load_book
from riskengine.module_b.revalue import duration_convexity, revalue, treasury_after
from riskengine.module_b.shock import EPISODES, market, rates_up_market

CENT = 0.01


@pytest.fixture(scope="module")
def book():
    return load_book()


# 1. rates fixed (or no move) reproduces E14 exactly
@pytest.mark.parametrize("episode", sorted(EPISODES))
def test_e22_1_no_rate_move_is_e14(book, episode):
    fixed = revalue(book, None, 0, episode, 1.0, "fixed").positions
    still = revalue(book, None, 0, replace(market(episode), d_rate=0.0), 1.0, "moved").positions
    for col in ("value_before", "value_after", "loss", "rwa_before", "rwa_after"):
        assert np.allclose(fixed[col], still[col], atol=CENT), col


def test_e22_1_stored_e14_results_reproduce(book):
    crises = crisis_backtest(book, rates="fixed")
    stored = pd.read_csv(CRISES_PATH)
    for col in ("realised_loss", "predicted_loss", "naive_loss"):
        assert np.allclose(crises[col], stored[col], atol=CENT), col
    alerts = alert_check(book)
    stored = pd.read_csv(ALERTS_PATH)
    for col in ("predicted_loss", "naive_loss", "realised_loss"):
        assert np.allclose(alerts[col], stored[col], atol=CENT), col


# 2. direction: rates up → bonds lose, pay-fixed swaps gain; rates down → the opposite
@pytest.mark.parametrize("bp", [100, -100])
def test_e22_2_direction(book, bp):
    pos = revalue(book, None, 0, rates_up_market(bp / 100), 1.0, "moved").positions
    change = pos["value_after"] - pos["value_before"]
    bonds, swaps = pos["instrument"] == "bond", pos["instrument"] == "irs_pay_fixed"
    sign = -1 if bp > 0 else 1
    assert (np.sign(change[bonds]) == sign).all()
    assert (np.sign(change[swaps]) == -sign).all()


# 3. size: full repricing within 5% of duration + convexity, ±100 bp
@pytest.mark.parametrize("bp", [100, -100])
def test_e22_3_bond_size_matches_duration_convexity(book, bp):
    r = revalue(book, None, 0, rates_up_market(bp / 100), 1.0, "moved")
    pos, sc = r.positions, r.scenario
    rate0, dy = r.market.rate0 / 100, bp / 10_000
    for _, p in pos[pos["instrument"] == "bond"].iterrows():
        y = rate0 + sc.loc[p["ticker"], "spread_before"] / 100  # par: coupon = yield
        d, cx = duration_convexity(y, y, p["years"])
        estimate = p["value_before"] * (-d * dy + 0.5 * cx * dy**2)
        full = p["value_after"] - p["value_before"]
        assert full == pytest.approx(estimate, rel=0.05), p["position"]


# 4. swaps are at par on the book date
def test_e22_4_swaps_zero_at_book_date(book):
    moved = revalue(book, None, 0, rates_up_market(0.0), 1.0, "moved").positions
    fixed = revalue(book, None, 0, rates_up_market(0.0), 1.0, "fixed").positions
    swaps = moved["instrument"] == "irs_pay_fixed"
    assert np.allclose(moved.loc[swaps, "value_after"], fixed.loc[swaps, "value_after"])
    assert np.allclose(moved.loc[swaps, "value_before"], moved.loc[swaps, "value_after"])


# 5. monotonic: bigger rise, bigger rate loss
def test_e22_5_monotonic(book):
    losses = [revalue(book, None, 0, rates_up_market(bp / 100), 1.0, "moved").loss_usd
              for bp in (0, 100, 200, 300)]  # fmt: skip
    assert all(b >= a - CENT for a, b in zip(losses, losses[1:], strict=False)), losses


# 6. floor: no scenario rate below 0
@pytest.mark.parametrize("episode", sorted(EPISODES))
def test_e22_6_floor(episode):
    mk = market(episode)
    for k in (0.5, 1, 2, 5, 10, 20):
        assert treasury_after(mk, k, "moved") >= 0.0
    assert treasury_after(mk, 20, "moved") == 0.0  # a deep fall hits the floor
