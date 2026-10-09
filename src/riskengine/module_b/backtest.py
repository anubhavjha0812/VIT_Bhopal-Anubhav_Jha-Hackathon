"""The two Module B checks of E14, scored once.

uv run python -m riskengine.module_b.backtest   # data/outputs/module_b_*.csv

Check 1, real alerts: each gated alert's predicted change in the bond and CDS book (the alert plus
the configured COVID episode) against the book repriced at the realised FRED Aaa/Baa moves over the
next 20 trading days (with the alert's own rating cut, which really happened). The naive guess is
the flat COVID episode without the alert. FRED gives rating buckets, not company spreads.

Check 2, crisis backtest: episodes are every non-overlapping window in 1990-2023 where FRED Baa
widens by 75 bp or more within 60 trading days. For each, its first 5 trading days (changes in Baa,
Aaa, VIX, S&P 500, USD/INR) are matched, leaving it out, to the 3 nearest other episodes
(standardised Euclidean distance); their average 60-day moves are the prediction, valued on our
book (P&L and CET1) next to the realised 60-day moves and a naive average of all other episodes.
z for these episodes uses the S&P 500 (our 15 stocks only start in 2008).
"""

import math

import numpy as np
import pandas as pd

from riskengine.config import load_config
from riskengine.gate import gate
from riskengine.ingest.linking import Linker
from riskengine.module_b.book import build_book
from riskengine.module_b.revalue import revalue
from riskengine.module_b.shock import (
    BOOK_DATE,
    CALM,
    DEFAULT_EPISODE,
    TRADING_DAYS,
    Market,
    market,
    market_history,
    parse_notches,
)
from riskengine.paths import DATA
from riskengine.pipeline import load_signals

OUT = DATA / "outputs"
ALERTS_PATH = OUT / "module_b_alerts_check.csv"
CRISES_PATH = OUT / "module_b_crisis_backtest.csv"
HORIZON = 20  # trading days after an alert
WIDENING, WINDOW, FIRST_DAYS, NEIGHBOURS = 0.75, 60, 5, 3
FEATURES = ["d_baa", "d_aaa", "d_vix", "spx_ret", "fx_ret"]
MARKET_BOOK = ("bond", "cds_bought", "cds_sold")


def _market_loss(result) -> float:
    pos = result.positions
    return float(pos.loc[pos["instrument"].isin(MARKET_BOOK), "loss"].sum())


# --- Check 1 -------------------------------------------------------------------------------------


def alert_check(book: pd.DataFrame | None = None) -> pd.DataFrame:
    book = build_book() if book is None else book
    cfg = load_config()
    fired = gate(load_signals(), cfg.impact_trigger, Linker.from_csv())
    fired = fired[fired["fired"]].sort_values("published_at")
    days = market_history().index
    rows = []
    for a in fired.itertuples():
        start = days[days <= a.published_at.tz_localize(None).normalize()][-1]
        end = days[min(days.get_loc(start) + HORIZON, len(days) - 1)]
        n = parse_notches(a.text, a.event)
        multiple = 1.0 if cfg.shock == "flat" else a.impact / 10
        predicted = _market_loss(revalue(book, a.ticker, n, DEFAULT_EPISODE, multiple))
        naive = _market_loss(revalue(book, None, 0, DEFAULT_EPISODE, 1.0))
        realised = _market_loss(revalue(book, a.ticker, n, (str(start.date()), str(end.date()))))
        rows.append({
            "published_at": a.published_at, "ticker": a.ticker, "notches": n, "impact": a.impact,
            "text": a.text[:120], "window": f"{start.date()} to {end.date()}",
            "predicted_loss": predicted, "naive_loss": naive, "realised_loss": realised,
            "sign_match": np.sign(predicted) == np.sign(realised),
            "error": abs(predicted - realised), "naive_error": abs(naive - realised),
        })  # fmt: skip
    return pd.DataFrame(rows)


# --- Check 2 -------------------------------------------------------------------------------------


def find_episodes(m: pd.DataFrame | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping (start, start + 60 trading days) windows where Baa widens ≥ 75 bp."""
    m = market_history().loc["1990":"2023"] if m is None else m
    baa, out, i = m["baa"].to_numpy(), [], 0
    while i + WINDOW < len(baa):
        if (baa[i + 1 : i + WINDOW + 1] - baa[i]).max() >= WIDENING:
            out.append((m.index[i], m.index[i + WINDOW]))
            i += WINDOW + 1
        else:
            i += 1
    return out


def moves(m: pd.DataFrame, start: pd.Timestamp, days: int) -> dict:
    """Market moves from `start` over `days` trading days."""
    i = m.index.get_loc(start)
    a, b = m.iloc[i], m.iloc[min(i + days, len(m) - 1)]
    return {"d_baa": b["baa"] - a["baa"], "d_aaa": b["aaa"] - a["aaa"],
            "d_vix": b["vix"] - a["vix"], "spx_ret": np.log(b["spx"] / a["spx"]),
            "fx_ret": b["usdinr"] / a["usdinr"] - 1}  # fmt: skip


def as_market(name: str, mv: dict, sigma_1y: float) -> Market:
    """Our book at its book date, hit by these moves."""
    base = market(DEFAULT_EPISODE)  # book-date levels
    z = -mv["spx_ret"] / sigma_1y
    return Market(name, base.aaa0, base.baa0, mv["d_aaa"], mv["d_baa"], float(z), base.fx0,
                  mv["fx_ret"], base.rate0)  # fmt: skip


def crisis_backtest(book: pd.DataFrame | None = None) -> pd.DataFrame:
    book = build_book() if book is None else book
    m = market_history().loc["1990":"2023"].ffill()
    spx = np.log(m["spx"]).diff()
    sigma_1y = spx.loc[CALM[0] : CALM[1]].std() * math.sqrt(TRADING_DAYS)
    eps = find_episodes(m)
    first = pd.DataFrame([moves(m, s, FIRST_DAYS) for s, _ in eps])[FEATURES]
    full = pd.DataFrame([moves(m, s, WINDOW) for s, _ in eps])
    scaled = (first - first.mean()) / first.std()
    rows = []
    for i, (start, end) in enumerate(eps):
        others = [j for j in range(len(eps)) if j != i]
        dist = ((scaled.iloc[others] - scaled.iloc[i]) ** 2).sum(axis=1) ** 0.5
        near = dist.nsmallest(NEIGHBOURS).index
        guesses = {"realised": full.iloc[i], "predicted": full.loc[near].mean(),
                   "naive": full.iloc[others].mean()}  # fmt: skip
        row = {"start": start.date(), "end": end.date(), "d_baa": full.iloc[i]["d_baa"],
               "neighbours": "; ".join(str(eps[j][0].date()) for j in near)}  # fmt: skip
        for key, mv in guesses.items():
            r = revalue(book, None, 0, as_market(f"{key} {start.date()}", mv.to_dict(), sigma_1y))
            row[f"{key}_loss"] = r.loss_usd
            row[f"{key}_cet1"] = r.cet1_after
        rows.append(row)
    df = pd.DataFrame(rows)
    for key in ("predicted", "naive"):
        df[f"{key}_loss_error"] = (df[f"{key}_loss"] - df["realised_loss"]).abs()
        df[f"{key}_cet1_error"] = (df[f"{key}_cet1"] - df["realised_cet1"]).abs()
        df[f"{key}_sign_match"] = np.sign(df[f"{key}_loss"]) == np.sign(df["realised_loss"])
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    book = build_book()
    alerts = alert_check(book)
    alerts.to_csv(ALERTS_PATH, index=False)
    cols = ["ticker", "window", "predicted_loss", "naive_loss", "realised_loss", "sign_match"]
    print(alerts[cols].to_string(index=False))
    crises = crisis_backtest(book)
    crises.to_csv(CRISES_PATH, index=False)
    shown = ["start", "d_baa", "realised_loss", "predicted_loss", "naive_loss",
             "realised_cet1", "predicted_cet1", "naive_cet1"]  # fmt: skip
    print(crises[shown].round(4).to_string(index=False))
    print(f"{len(crises)} episodes (book date {BOOK_DATE}); mean absolute error:")
    for key in ("predicted", "naive"):
        print(f"  {key:9} loss ${crises[f'{key}_loss_error'].mean() / 1e6:.1f}m, "
              f"CET1 {100 * crises[f'{key}_cet1_error'].mean():.2f} pt, "
              f"sign {crises[f'{key}_sign_match'].mean():.0%}")  # fmt: skip


if __name__ == "__main__":
    main()
