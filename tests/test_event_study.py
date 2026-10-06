import numpy as np
import pandas as pd
import pytest

from riskengine.impact.event_study import abnormal_moves, event_day_index

DAYS = pd.DatetimeIndex(["2020-04-13", "2020-04-14", "2020-04-15"])


@pytest.mark.parametrize(
    ("utc_time", "expected"),
    [
        ("2020-04-13 14:00:00+00:00", 0),  # 10:00 New York, market open -> same day
        ("2020-04-13 21:00:00+00:00", 1),  # 17:00 New York, after close -> next day
        ("2020-04-12 15:00:00+00:00", 0),  # Sunday -> Monday
        ("2020-04-13 00:00:00+00:00", 0),  # date-only stamp = Sunday 20:00 NY -> Monday
    ],
)
def test_event_day(utc_time, expected):
    pos = event_day_index(pd.Series(pd.to_datetime([utc_time])), DAYS)
    assert pos[0] == expected


def test_car_removes_market_move():
    days = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, len(days))
    stock = 2.0 * market  # beta exactly 2, no own news...
    stock[350] += 0.05  # ...except a +5% surprise on day 350
    rets = pd.DataFrame({"^GSPC": market, "X": stock}, index=days)
    rows = pd.DataFrame(
        {
            "ticker": ["X"],
            "published_at": [
                pd.Timestamp(days[350], tz="America/New_York") + pd.Timedelta(hours=10)
            ],
        }
    )
    out = abnormal_moves(rows, rets)
    assert out["beta"].iloc[0] == pytest.approx(2.0, abs=1e-6)
    assert out["car01"].iloc[0] == pytest.approx(0.05, abs=1e-6)


def test_too_little_history_is_dropped():
    days = pd.bdate_range("2020-01-01", periods=100)
    rets = pd.DataFrame({"^GSPC": 0.01, "X": 0.01}, index=days)
    rows = pd.DataFrame({"ticker": ["X"], "published_at": [pd.Timestamp(days[90], tz="UTC")]})
    assert np.isnan(abnormal_moves(rows, rets)["car01"].iloc[0])
