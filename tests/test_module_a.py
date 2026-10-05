import pandas as pd
import pytest

from riskengine.module_a import backtest, cap_weights, daily_signal, target_weights

DAYS = pd.DatetimeIndex(["2020-04-13", "2020-04-14", "2020-04-15", "2020-04-16"])


def test_text_is_only_usable_the_next_trading_day():
    signals = pd.DataFrame(
        {
            # 15:00 New York on the 13th -> usable on the 14th, not the 13th
            "published_at": pd.to_datetime(["2020-04-13 19:00:00+00:00"]),
            "ticker": ["AAPL"],
            "sentiment": [0.8],
        }
    )
    sig = daily_signal(signals, DAYS)
    assert sig.loc["2020-04-13", "AAPL"] == 0.0
    assert sig.loc["2020-04-14", "AAPL"] == pytest.approx(0.8)


def test_cap_weights_respects_cap_and_sums_to_one():
    w = cap_weights(pd.Series([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), cap=0.15)
    assert w.max() <= 0.15 + 1e-9
    assert w.sum() == pytest.approx(1.0)


def test_sentiment_tilt_overweights_positive_names():
    others = {f"X{i}": [0.0] for i in range(12)}
    signal = pd.DataFrame({"A": [0.5], "B": [0.0], "C": [-0.5], **others}, index=DAYS[:1])
    w = target_weights(signal, "sentiment").iloc[0]
    assert w["A"] > w["B"] > w["C"]
    assert w.sum() == pytest.approx(1.0)


def test_backtest_charges_costs_and_uses_next_day_return():
    closes = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 100.0]}, index=DAYS[:3])
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0]}, index=DAYS[:3])
    bt = backtest(weights, closes, cost_bps=10)
    # day 1: buy A (turnover 1 -> 10 bp cost), earn +10% next day
    assert bt.returns.iloc[0] == pytest.approx(0.10 - 0.001)
    assert bt.returns.iloc[1] == pytest.approx(0.0)
