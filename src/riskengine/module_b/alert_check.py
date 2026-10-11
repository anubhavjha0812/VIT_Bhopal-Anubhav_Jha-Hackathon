"""E22b: does a credit alert itself carry information? Run once (docs/DECISIONS.md, E22b).

uv run python -m riskengine.module_b.alert_check   # data/outputs/e22b_alerts.csv, _results.json

Alerts: the E17 history headlines (2010 – 8 Apr 2020) through the E13 credit rules, the impact
formula and the unchanged gate; plus the 4 replay-window alerts (already seen, reported apart).
Prediction: the alert alone, `revalue(..., multiple=0)`. Outcome (a proxy, not a credit loss):
the obligor's 20-day log return minus the S&P 500's, from the last close before the alert date.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from riskengine.config import load_config
from riskengine.gate import gate
from riskengine.ingest.linking import Linker
from riskengine.module_b.backtest import OUT
from riskengine.module_b.book import load_book
from riskengine.module_b.revalue import revalue
from riskengine.module_b.shock import DEFAULT_EPISODE, PRICES_PATH, parse_notches
from riskengine.nlp.events import credit_event, names_agency
from riskengine.nlp.impact import impact_formula

ALERTS = OUT / "e22b_alerts.csv"
RESULTS = OUT / "e22b_results.json"
HORIZON = 20  # trading days
START, END = "2010-01-01", "2020-04-08"  # primary sample; the replay window starts 9 Apr 2020
MARKET = "^GSPC"
MIN_N, ALPHA = 20, 0.05
BOOT, SEED = 10_000, 0


def closes() -> pd.DataFrame:
    p = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    return p.pivot(index="date", columns="ticker", values="close").sort_index()


def window(days: pd.DatetimeIndex, alert_date: pd.Timestamp, horizon: int = HORIZON):
    """(start, end): the last trading day strictly before the alert date, and `horizon` trading
    days after it. None if either is outside the price history."""
    i = days.searchsorted(alert_date.normalize(), side="left") - 1
    if i < 0 or i + horizon >= len(days):
        return None
    return days[i], days[i + horizon]


def market_adjusted(px: pd.DataFrame, ticker: str, start, end) -> float:
    """log return of the stock minus that of the S&P 500 between two closes (NaN if missing)."""
    a, b = px.loc[start], px.loc[end]
    return float(np.log(b[ticker] / a[ticker]) - np.log(b[MARKET] / a[MARKET]))


def drop_overlaps(alerts: pd.DataFrame, days: pd.DatetimeIndex, horizon: int = HORIZON):
    """Per company keep an alert only if its start is at least `horizon` trading days after the
    start of the last kept one. Returns a boolean Series (True = kept)."""
    keep = pd.Series(False, index=alerts.index)
    last: dict[str, int] = {}
    for i, row in alerts.sort_values("published_at").iterrows():
        pos = days.get_loc(row["start"])
        if row["ticker"] not in last or pos - last[row["ticker"]] >= horizon:
            keep[i] = True
            last[row["ticker"]] = pos
    return keep


def baseline(px: pd.DataFrame, tickers: list[str], horizon: int = HORIZON) -> pd.DataFrame:
    """Per company: share of negative and mean market-adjusted `horizon`-day returns from every
    start day in the primary period."""
    lp = np.log(px)
    fwd = lp.shift(-horizon) - lp
    adj = fwd[tickers].sub(fwd[MARKET], axis=0).loc[START:END]
    return pd.DataFrame({"base_rate": (adj < 0).sum() / adj.notna().sum(), "base_mean": adj.mean()})


def history_alerts(linker: Linker) -> pd.DataFrame:
    from riskengine.history import load_history

    h = load_history()
    h = h[h["why"].isna() & (h["role"] == "subject")]
    h = h[(h["published_at"] >= START) & (h["published_at"] < pd.Timestamp(END, tz="UTC")
                                           + pd.Timedelta(days=1))]  # fmt: skip
    h = h.assign(event=h["text"].map(lambda t: credit_event(t) or "Other"))
    h = h[h["event"] != "Other"].copy()
    h["impact"] = [
        impact_formula(e, s, "news", names_agency(t))
        for e, s, t in zip(h["event"], h["sentiment"], h["text"], strict=True)
    ]
    h["source"], h["doc_id"] = "news", [f"h{i}" for i in range(len(h))]
    g = gate(h.reset_index(drop=True), load_config().impact_trigger, linker)
    return g[g["fired"]].assign(sample="primary (history 2010 – 8 Apr 2020)")


def replay_alerts(linker: Linker) -> pd.DataFrame:
    from riskengine.pipeline import load_signals

    g = gate(load_signals(), load_config().impact_trigger, linker)
    return g[g["fired"]].assign(sample="secondary (replay window, seen)")


def score(alerts: pd.DataFrame, px: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    days = px.index
    rows = []
    in_book = set(book["ticker"])
    rates = load_config().rates
    cache: dict[tuple[str, int], float] = {}
    for _, a in alerts.iterrows():
        notches = parse_notches(a["text"], a["event"])
        row = {"sample": a["sample"], "published_at": a["published_at"], "ticker": a["ticker"],
               "event": a["event"], "impact": int(a["impact"]), "gate": a["reason"],
               "notches": notches, "text": " ".join(str(a["text"]).split())[:200]}  # fmt: skip
        w = window(days, a["published_at"].tz_convert(None))
        if a["ticker"] not in in_book:
            row["dropped"] = "not in the book"
        elif w is None:
            row["dropped"] = "window outside the price history"
        else:
            row["start"], row["end"] = w
            row["adj_return"] = market_adjusted(px, a["ticker"], *w)
            if not np.isfinite(row["adj_return"]):
                row["dropped"] = "missing close"
            key = (a["ticker"], notches)
            if key not in cache:
                cache[key] = revalue(book, a["ticker"], notches, DEFAULT_EPISODE, 0.0,
                                     rates).loss_usd  # fmt: skip
            row["predicted_loss"] = cache[key]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["dropped"] = df.get("dropped", pd.Series(dtype=object))
    for _, g in df[df["dropped"].isna()].groupby("sample"):
        kept = drop_overlaps(g, days)
        df.loc[kept[~kept].index, "dropped"] = "within 20 trading days of an earlier alert"
    return df


def summarise(df: pd.DataFrame, base: pd.DataFrame) -> dict:
    out = {}
    rng = np.random.default_rng(SEED)
    for sample, g in df.groupby("sample"):
        s = g[g["dropped"].isna()].merge(base, left_on="ticker", right_index=True)
        n, k = len(s), int((s["adj_return"] < 0).sum())
        res = {"alerts_fired": len(g), "dropped": g["dropped"].value_counts().to_dict(),
               "n": n, "negative": k}  # fmt: skip
        if n:
            p0 = float(s["base_rate"].mean())
            r = s["adj_return"].to_numpy()
            boot = [rng.choice(r, n).mean() for _ in range(BOOT)]
            rho = spearmanr(s["predicted_loss"], r) if n > 2 else None
            res |= {
                "share_negative": k / n,
                "base_rate_p0": p0,
                "binomial_p_one_sided": binomtest(k, n, p0, alternative="greater").pvalue,
                "mean_adj_return": float(r.mean()),
                "mean_adj_return_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                "baseline_mean_adj_return": float(s["base_mean"].mean()),
                "spearman_predicted_loss_vs_return": None if rho is None else float(rho.statistic),
                "spearman_p": None if rho is None else float(rho.pvalue),
            }
        out[sample] = res
    prim = next(v for k, v in out.items() if k.startswith("primary"))
    if prim["n"] < MIN_N:
        prim["verdict"] = f"descriptive only: n = {prim['n']} < {MIN_N}"
    else:
        prim["verdict"] = "PASS" if prim["binomial_p_one_sided"] < ALPHA else "FAIL"
    return out


def main() -> None:
    linker, px, book = Linker.from_csv(), closes(), load_book()
    alerts = pd.concat([history_alerts(linker), replay_alerts(linker)], ignore_index=True)
    df = score(alerts, px, book)
    base = baseline(px, sorted(set(book["ticker"]) & set(px.columns)))
    res = {
        "rule": "docs/DECISIONS.md E22b",
        "horizon_trading_days": HORIZON,
        "samples": summarise(df, base),
        "baseline_by_company": base.round(4).to_dict("index"),
    }
    df.to_csv(ALERTS, index=False)
    RESULTS.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res["samples"], indent=2, default=float))


if __name__ == "__main__":
    main()
