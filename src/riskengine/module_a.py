"""Module A: sentiment-driven rebalancer for the mock index vs equal weight.

No look-ahead: a text is usable from the first trading day *after* its New York calendar
date; weights set at that day's close earn the next day's return. Costs are charged on
turnover.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskengine.config import Config

TILT = 0.5  # weight multiplier per 1 cross-sectional z-score of sentiment
MAX_WEIGHT = 0.15
LOOKBACK_DAYS = 3  # trading days of texts averaged into the signal
COST_BPS = 5.0
NY = "America/New_York"


@dataclass(frozen=True)
class Backtest:
    weights: pd.DataFrame  # date x ticker, set at that day's close
    returns: pd.Series  # net daily return, earned the next day
    metrics: dict[str, float]


def daily_signal(signals: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Mean sentiment of texts usable by each trading day (rolling LOOKBACK_DAYS), 0 if none."""
    s = signals.copy()
    ny_day = s["published_at"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    idx = np.searchsorted(trading_days.values, ny_day.values, side="right")
    s = s[idx < len(trading_days)]
    s["day"] = trading_days[idx[idx < len(trading_days)]]

    g = s.groupby(["day", "ticker"])["sentiment"]
    total = g.sum().unstack().reindex(trading_days).fillna(0.0)
    count = g.size().unstack().reindex(trading_days).fillna(0)
    total = total.rolling(LOOKBACK_DAYS, min_periods=1).sum()
    count = count.rolling(LOOKBACK_DAYS, min_periods=1).sum()
    return (total / count.where(count > 0)).fillna(0.0)


def cap_weights(w: pd.Series, cap: float = MAX_WEIGHT) -> pd.Series:
    """Long-only, sum 1, no name above `cap` (excess spread over uncapped names)."""
    cap = max(cap, 1.0 / len(w))  # a cap below 1/N cannot sum to 1
    w = w.clip(lower=0) / w.clip(lower=0).sum()
    for _ in range(len(w)):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        w[free] += excess * w[free] / w[free].sum()
    return w


def target_weights(signal: pd.DataFrame, kind: str) -> pd.DataFrame:
    n = signal.shape[1]
    if kind == "equal":
        return pd.DataFrame(1.0 / n, index=signal.index, columns=signal.columns)
    if kind != "sentiment":
        raise NotImplementedError(f"rebalancer {kind!r} is not built yet")
    std = signal.std(axis=1).replace(0, np.nan)
    z = signal.sub(signal.mean(axis=1), axis=0).div(std, axis=0).fillna(0.0)
    raw = (1 + TILT * z).clip(lower=0)
    return raw.apply(cap_weights, axis=1)


def backtest(weights: pd.DataFrame, closes: pd.DataFrame, cost_bps: float = COST_BPS) -> Backtest:
    rets = closes.pct_change()
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()  # buying the initial portfolio
    gross = (weights * rets.shift(-1)).sum(axis=1)  # weights at close t earn t -> t+1
    net = (gross - turnover * cost_bps / 1e4).iloc[:-1]
    equity = (1 + net).cumprod()
    metrics = {
        "total_return": equity.iloc[-1] - 1,
        "ann_vol": net.std() * np.sqrt(252),
        "sharpe": net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0.0,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "avg_daily_turnover": turnover.mean(),
    }
    return Backtest(weights, net, {k: float(v) for k, v in metrics.items()})


def run_module_a(
    signals: pd.DataFrame, prices: pd.DataFrame, config: Config
) -> dict[str, Backtest]:
    """Backtests for equal weight and the configured rebalancer over the replay window."""
    closes = prices.pivot(index="date", columns="ticker", values="close").drop(
        columns="^GSPC", errors="ignore"
    )
    start, end = pd.Timestamp(config.window_start), pd.Timestamp(config.window_end)
    closes = closes.loc[(closes.index >= start) & (closes.index <= end)]
    signal = daily_signal(signals, closes.index).reindex(columns=closes.columns, fill_value=0.0)

    kinds = dict.fromkeys(["equal", config.rebalancer])
    return {k: backtest(target_weights(signal, k), closes) for k in kinds}
