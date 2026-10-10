import math

import numpy as np
import pandas as pd
import pytest

from riskengine.module_b.backtest import FEATURES
from riskengine.module_b.outlook import (
    diebold_mariano,
    forecast,
    history,
    month_ends,
    predictions,
    training,
)


@pytest.fixture(scope="module")
def market():
    """Synthetic 1990-2023 market where Baa mean-reverts to 2% and VIX follows it."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("1990-01-01", "2023-12-29")
    baa = np.empty(len(idx))
    baa[0] = 2.0
    for t in range(1, len(idx)):
        baa[t] = 2.0 + 0.97 * (baa[t - 1] - 2.0) + 0.03 * rng.standard_normal()
    vix = 20 + 5 * (baa - 2) + rng.standard_normal(len(idx))
    walk = np.exp(np.cumsum(0.01 * rng.standard_normal(len(idx))))
    return pd.DataFrame({"baa": baa, "aaa": baa - 1, "vix": vix, "spx": 100 * walk,
                         "usdinr": 50.0}, idx)  # fmt: skip


def test_month_ends_are_last_trading_days(market):
    ends = month_ends(market.loc["2020-01":"2020-03"])
    assert list(ends.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-28", "2020-03-31"]


def test_training_never_sees_a_window_ending_after_the_origin(market):
    hist = history(market)
    for origin in hist.loc["1995":"2023"].index[::25]:
        train = training(hist, origin)
        assert (train["window_end"] <= origin).all()
        assert len(train) > 0
        later = hist[(hist.index < origin) & (hist["window_end"] > origin)]
        assert later.index.isin(train.index).sum() == 0  # the last ~3 month-ends are left out


def test_naive_is_the_mean_of_earlier_moves(market):
    hist = history(market)
    origin = hist.loc["2000-06"].index[0]
    train = training(hist, origin)
    _, naive = forecast(train, hist.loc[origin])
    assert naive.tolist() == pytest.approx(train[FEATURES].mean().tolist())


def test_model_beats_naive_when_spreads_mean_revert(market):
    df = predictions(market)
    model = (df["model_d_baa"] - df["realised_d_baa"]).abs().mean()
    naive = (df["naive_d_baa"] - df["realised_d_baa"]).abs().mean()
    assert model < 0.9 * naive
    assert df.index.min() >= pd.Timestamp("1995-01-01")
    assert df.index.max() < pd.Timestamp("2023-10-01")


def test_diebold_mariano():
    d = np.array([1.0, -1.0, 2.0, 0.0, 3.0])
    stat, p = diebold_mariano(d, lag=0)  # no lag: the plain t-statistic
    assert stat == pytest.approx(d.mean() / math.sqrt(d.var() / len(d)))
    assert 0 < p < 1
