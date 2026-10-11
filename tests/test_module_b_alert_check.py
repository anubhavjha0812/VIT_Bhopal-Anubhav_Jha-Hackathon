"""E22b mechanics on synthetic data: window placement, overlap rule, market-adjusted return."""

import numpy as np
import pandas as pd
import pytest

from riskengine.module_b.alert_check import drop_overlaps, market_adjusted, window

DAYS = pd.bdate_range("2015-01-01", periods=100)


def test_window_starts_strictly_before_the_alert_date():
    alert = DAYS[10] + pd.Timedelta(hours=15)  # intraday on a trading day
    start, end = window(DAYS, alert, 20)
    assert start == DAYS[9]  # the day before: the alert day's own move is inside the window
    assert end == DAYS[29]


def test_window_on_a_weekend_uses_the_last_friday():
    saturday = pd.Timestamp("2015-01-10")
    start, _ = window(DAYS, saturday, 5)
    assert start == pd.Timestamp("2015-01-09")


def test_window_outside_history_is_none():
    assert window(DAYS, DAYS[0], 5) is None  # no close before it
    assert window(DAYS, DAYS[-3], 5) is None  # end beyond the data


def test_overlapping_alerts_of_one_company_are_dropped():
    a = pd.DataFrame({
        "published_at": [DAYS[i] for i in (10, 15, 31, 12)],
        "ticker": ["BA", "BA", "BA", "DIS"],
        "start": [DAYS[i] for i in (9, 14, 30, 11)],
    })  # fmt: skip
    kept = drop_overlaps(a, DAYS, 20)
    assert kept.tolist() == [True, False, True, True]  # other companies are independent


def test_market_adjusted_return():
    px = pd.DataFrame({"BA": [100.0, 90.0], "^GSPC": [1000.0, 1100.0]}, index=DAYS[:2])
    r = market_adjusted(px, "BA", DAYS[0], DAYS[1])
    assert r == pytest.approx(np.log(0.9) - np.log(1.1))
