"""Market-model event study: how much did a stock really move after a headline?

Event day = first trading day whose New York close (16:00) is after the headline time.
AR = r_stock - (alpha + beta * r_market), alpha/beta from trading days -280..-31.
Target = |CAR[0,+1]| = |AR(day 0) + AR(day +1)|.
"""

import numpy as np
import pandas as pd

NY = "America/New_York"
EST_START, EST_END, MIN_OBS = 280, 31, 120


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide daily simple returns: date x ticker."""
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    return close.pct_change()


def event_day_index(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Position in `trading_days` of each headline's event day (len(trading_days) if none)."""
    ny = published_at.dt.tz_convert(NY)
    day = ny.dt.tz_localize(None).dt.normalize()
    after_close = ny.dt.hour >= 16
    # first trading day >= the headline's date, or > it when published after the close
    idx = np.searchsorted(trading_days.values, day.values, side="left")
    later = np.searchsorted(trading_days.values, day.values, side="right")
    return np.where(after_close.to_numpy(), later, idx)


def abnormal_moves(rows: pd.DataFrame, rets: pd.DataFrame, market: str = "^GSPC") -> pd.DataFrame:
    """For rows with `ticker` and `published_at`: event_day, car01, abs_car01, beta."""
    days = rets.index
    pos = event_day_index(rows["published_at"], days)
    out = pd.DataFrame(index=rows.index, columns=["event_day", "car01", "beta"], dtype=float)
    out["event_day"] = pd.NaT

    cache: dict[tuple[str, int], tuple[float, float, float] | None] = {}
    m = rets[market].to_numpy()
    for i, (ticker, p) in enumerate(zip(rows["ticker"], pos, strict=True)):
        if p + 1 >= len(days) or ticker not in rets:
            continue
        key = (ticker, int(p))
        if key not in cache:
            cache[key] = _car01(rets[ticker].to_numpy(), m, int(p))
        res = cache[key]
        if res is None:
            continue
        idx = rows.index[i]
        out.loc[idx, ["car01", "beta"]] = res[0], res[1]
        out.loc[idx, "event_day"] = days[p]
    out["abs_car01"] = out["car01"].abs()
    return out


def _car01(r: np.ndarray, m: np.ndarray, p: int) -> tuple[float, float, float] | None:
    lo, hi = p - EST_START, p - EST_END
    if lo < 1:
        return None
    y, x = r[lo:hi], m[lo:hi]
    ok = ~(np.isnan(y) | np.isnan(x))
    if ok.sum() < MIN_OBS:
        return None
    beta, alpha = np.polyfit(x[ok], y[ok], 1)
    ar = r[p : p + 2] - (alpha + beta * m[p : p + 2])
    if np.isnan(ar).any():
        return None
    return float(ar.sum()), float(beta), float(alpha)
