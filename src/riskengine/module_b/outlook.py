"""State-aware 60-day market move and its monthly backtest (DECISIONS.md, E15), scored once.

uv run python -m riskengine.module_b.outlook   # data/outputs/module_b_monthly_backtest.csv

At the last trading day of each month, Jan 1995 - Sep 2023, the next 60 trading days' moves (as in
Check 2) are predicted by least squares on the state at that day (Baa level, VIX level, Baa change
over the previous 20 trading days), fitted only on earlier month-ends whose 60-day window had
already ended. The naive guess is the average of the same earlier moves. Both, and the realised
moves, are valued on our 31 Dec 2019 book. Pass: model loss MAE <= 0.85 x naive loss MAE.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskengine.module_b.backtest import FEATURES, OUT, WINDOW, as_market, moves
from riskengine.module_b.book import build_book
from riskengine.module_b.revalue import revalue
from riskengine.module_b.shock import BOOK_DATE, CALM, TRADING_DAYS, market_history

MONTHLY_PATH = OUT / "module_b_monthly_backtest.csv"
STATE = ["baa", "vix", "baa_20d"]
MOMENTUM = 20  # trading days
TRAIN_FROM, FIRST_ORIGIN, LAST_ORIGIN = "1990-02", "1995-01", "2023-09"
PASS = 0.85  # model MAE must be at most 85% of the naive's
NW_LAG = 2  # nearby months' 60-day windows overlap


def month_ends(m: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(m.index.to_series().groupby(m.index.to_period("M")).max())


def history(m: pd.DataFrame) -> pd.DataFrame:
    """One row per month-end from TRAIN_FROM: the state, the next 60-day moves, the window end."""
    rows = []
    for day in month_ends(m.loc[TRAIN_FROM:]):
        i = m.index.get_loc(day)
        if i < MOMENTUM or i + WINDOW >= len(m):
            continue
        baa = m["baa"].iloc[i]
        rows.append({"origin": day, "window_end": m.index[i + WINDOW], "baa": baa,
                     "vix": m["vix"].iloc[i], "baa_20d": baa - m["baa"].iloc[i - MOMENTUM],
                     **moves(m, day, WINDOW)})  # fmt: skip
    return pd.DataFrame(rows).set_index("origin")


def training(hist: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Earlier month-ends whose 60-day window had ended by the origin: nothing from the future."""
    return hist[hist["window_end"] <= origin]


def forecast(
    train: pd.DataFrame, state: pd.Series, targets: list[str] = FEATURES
) -> tuple[pd.Series, pd.Series]:
    """(model, naive) 60-day moves for one state, from a training set."""
    x = np.column_stack([np.ones(len(train)), train[STATE].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, train[targets].to_numpy(), rcond=None)
    model = np.concatenate([[1.0], state[STATE].to_numpy(dtype=float)]) @ beta
    return pd.Series(model, targets), train[targets].mean()


def predictions(m: pd.DataFrame, targets: list[str] = FEATURES) -> pd.DataFrame:
    """Realised, model and naive moves at every test origin (columns '<kind>_<feature>')."""
    hist = history(m)
    rows = []
    for origin in hist.loc[FIRST_ORIGIN:LAST_ORIGIN].index:
        now = hist.loc[origin]
        model, naive = forecast(training(hist, origin), now, targets)
        row = {"origin": origin, **{s: now[s] for s in STATE}}
        for kind, mv in (("realised", now[targets]), ("model", model), ("naive", naive)):
            row.update({f"{kind}_{f}": float(mv[f]) for f in targets})
        rows.append(row)
    return pd.DataFrame(rows).set_index("origin")


def diebold_mariano(d: np.ndarray, lag: int = NW_LAG) -> tuple[float, float]:
    """DM statistic and two-sided p-value for a loss differential, Newey-West variance."""
    d = np.asarray(d, dtype=float)
    n, e = len(d), d - d.mean()
    lrv = e @ e / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    stat = d.mean() / math.sqrt(lrv / n)
    return stat, 2 * norm.sf(abs(stat))


def monthly_backtest(book: pd.DataFrame | None = None, rates: str = "fixed") -> pd.DataFrame:
    """E15; with rates="moved" (E22) the Treasury move is also predicted and applied."""
    book = build_book() if book is None else book
    m = market_history().loc[:"2023"].ffill()
    spx = np.log(m["spx"]).diff()
    sigma_1y = spx.loc[CALM[0] : CALM[1]].std() * math.sqrt(TRADING_DAYS)
    targets = [*FEATURES, "d_ust"] if rates == "moved" else FEATURES
    df = predictions(m, targets)
    for kind in ("realised", "model", "naive"):
        losses, cet1 = [], []
        for origin, row in df.iterrows():
            mv = {f: row[f"{kind}_{f}"] for f in targets}
            mk = as_market(f"{kind} {origin.date()}", mv, sigma_1y)
            r = revalue(book, None, 0, mk, rates=rates)
            losses.append(r.loss_usd)
            cet1.append(r.cet1_after)
        df[f"{kind}_loss"], df[f"{kind}_cet1"] = losses, cet1
    for kind in ("model", "naive"):
        df[f"{kind}_loss_error"] = (df[f"{kind}_loss"] - df["realised_loss"]).abs()
        df[f"{kind}_cet1_error"] = (df[f"{kind}_cet1"] - df["realised_cet1"]).abs()
        df[f"{kind}_sign_match"] = np.sign(df[f"{kind}_loss"]) == np.sign(df["realised_loss"])
    return df


def summary(df: pd.DataFrame) -> dict:
    model, naive = df["model_loss_error"].mean(), df["naive_loss_error"].mean()
    stat, p = diebold_mariano(df["model_loss_error"] - df["naive_loss_error"])
    return {"origins": len(df), "model_mae": model, "naive_mae": naive, "ratio": model / naive,
            "passed": bool(model <= PASS * naive), "pass_ratio": PASS,
            "model_cet1_mae": df["model_cet1_error"].mean(),
            "naive_cet1_mae": df["naive_cet1_error"].mean(),
            "model_sign": df["model_sign_match"].mean(),
            "naive_sign": df["naive_sign_match"].mean(),
            "model_closer": (df["model_loss_error"] < df["naive_loss_error"]).mean(),
            "dm_stat": stat, "dm_p": p}  # fmt: skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = monthly_backtest()
    df.to_csv(MONTHLY_PATH)
    s = summary(df)
    print(f"{s['origins']} month-ends {FIRST_ORIGIN} to {LAST_ORIGIN} (book date {BOOK_DATE})")
    for key in ("model", "naive"):
        print(f"  {key:6} loss MAE ${s[f'{key}_mae'] / 1e6:.1f}m, "
              f"CET1 {100 * s[f'{key}_cet1_mae']:.2f} pt, sign {s[f'{key}_sign']:.0%}")  # fmt: skip
    print(f"  model / naive = {s['ratio']:.2f} (pass if <= {PASS}): "
          f"{'PASS' if s['passed'] else 'FAIL'}")  # fmt: skip
    print(f"  model closer in {s['model_closer']:.0%} of months; "
          f"Diebold-Mariano {s['dm_stat']:.2f}, p = {s['dm_p']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
