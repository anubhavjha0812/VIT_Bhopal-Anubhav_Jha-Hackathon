"""One credit alert through Module B, ready for the API and the dashboard (DECISIONS.md, E14).

Scenario, revaluation (IFRS 9, IRB, CET1), correlated Monte Carlo, reverse stress and a memo. The
memo is a fixed template, no LLM: every number comes from this run, and the trigger text, the
words the credit rules matched and the gate reason are quoted.
"""

import math

import numpy as np
import pandas as pd

from riskengine.module_b.montecarlo import DRAWS, LEVEL, before_after
from riskengine.module_b.revalue import CET1_FLOOR, TAX, Result, revalue, reverse_stress
from riskengine.module_b.shock import BOOK_DATE, EPISODES
from riskengine.nlp.events import AGENCY, DEFAULT, DOWNGRADE

EPISODE_LABELS = {"gfc2008": "GFC 2008", "euro2011": "euro crisis 2011", "covid2020": "COVID 2020"}
TOP = 10  # largest position losses shown
BINS = 40  # Monte Carlo histogram


def alert_multiple(impact: int, shock: str) -> float:
    """config.shock: `flat` applies the full episode, `scaled` the episode × impact/10."""
    return 1.0 if shock == "flat" else impact / 10


def matched_words(text: str) -> list[str]:
    """What the E13 credit rules matched in the text: the agency and the downgrade/default."""
    found: list[str] = []
    for rx in (AGENCY, DOWNGRADE, DEFAULT):
        for m in rx.finditer(text):
            word = " ".join(m.group(0).split())
            if word not in found:
                found.append(word)
    return [w for w in found if not any(w != o and w in o for o in found)]  # longest only


def _num(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _capital(r: Result) -> dict:
    return {
        "cet1_before": r.cet1_before,
        "cet1_after": r.cet1_after,
        "floor": CET1_FLOOR,
        "loss_usd": r.loss_usd,
        "loss_after_tax_inr": (1 - TAX) * r.loss_usd * r.fx_after,
        "capital_before_inr": r.capital_before_inr,
        "capital_after_inr": r.capital_after_inr,
        "rwa_before_inr": r.rwa_before_inr,
        "rwa_after_inr": r.rwa_after_inr,
        "fx_before": r.fx_before,
        "fx_after": r.fx_after,
    }


def summary(book: pd.DataFrame, ticker: str, notches: int, episode: str, multiple: float) -> dict:
    """The quick numbers for an alert list: loss and CET1 before → after."""
    return _capital(revalue(book, ticker, notches, episode, multiple))


def _histogram(before: np.ndarray, after: np.ndarray) -> dict:
    top = max(float(before.max()), float(after.max()), 1.0)
    edges = np.linspace(0.0, top, BINS + 1)
    return {"edges": edges.tolist(), "before": np.histogram(before, edges)[0].tolist(),
            "after": np.histogram(after, edges)[0].tolist()}  # fmt: skip


def run(
    book: pd.DataFrame, ticker: str | None, notches: int, episode: str, multiple: float
) -> dict:
    r = revalue(book, ticker, notches, episode, multiple)
    pos, sc, mk = r.positions, r.scenario, r.market
    by_ticker = pos.groupby("ticker")["loss"].sum()
    obligors = sc.assign(loss=by_ticker).reset_index(names="ticker")
    top = pos.reindex(pos["loss"].abs().sort_values(ascending=False).index).head(TOP)
    stages = pos.loc[pos["instrument"] == "revolver", "stage"].value_counts().sort_index()
    before, after = before_after(book, sc, market_shock=multiple > 0)
    k = reverse_stress(book, ticker, notches, episode)
    start, end = EPISODES[episode]
    out = {
        "book_date": BOOK_DATE,
        "scenario": {
            "ticker": ticker, "notches": notches, "episode": episode,
            "episode_label": EPISODE_LABELS[episode], "start": start, "end": end,
            "multiple": multiple,
            "rating_before": sc.loc[ticker, "rating_before"] if ticker else None,
            "rating_after": sc.loc[ticker, "rating_after"] if ticker else None,
            "d_baa_bp": 100 * multiple * mk.d_baa, "d_aaa_bp": 100 * multiple * mk.d_aaa,
            "fx_move": multiple * mk.fx_move, "z": multiple * mk.z,
        },
        "capital": _capital(r),
        "by_instrument": pos.groupby("instrument")["loss"].sum().round(2).to_dict(),
        "top_losses": top[["position", "ticker", "instrument", "rating", "notional", "loss",
                           "stage"]].to_dict("records"),
        "obligors": obligors[["ticker", "rating_before", "rating_after", "spread_before",
                              "spread_after", "pd_before", "pd_after", "loss"]].to_dict("records"),
        "stages": {str(s): int(n) for s, n in stages.items()},
        "monte_carlo": {
            "draws": DRAWS, "level": LEVEL, "histogram": _histogram(before.losses, after.losses),
            **{name: {"expected": d.expected, "var": d.var, "es": d.es}
               for name, d in (("before", before), ("after", after))},
        },
        "reverse_stress": _num(k),
    }  # fmt: skip
    return out


def _m(usd: float) -> str:
    return f"${usd / 1e6:,.1f}m"


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def memo(rep: dict, alert: dict | None = None) -> str:
    """Fixed template: the alert, the scenario, the result, the limits."""
    s, c, mc = rep["scenario"], rep["capital"], rep["monte_carlo"]
    lines = []
    if alert:
        quote = " ".join(alert["text"].split())
        lines.append(
            f'Trigger ({alert["published_at"][:10]}, {alert["source"]}): "{quote}" '
            f"Credit rules matched: {', '.join(alert['matched']) or 'none'}. "
            f"Gate: {alert['gate']}; event {alert['event']}, impact {alert['impact']}."
        )
    if s["ticker"]:
        cut = "default" if s["rating_after"] == "D" else f"{s['notches']}-notch cut"
        lines.append(
            f"Scenario: {s['ticker']} {s['rating_before']} → {s['rating_after']} ({cut}), "
            f"other names widen by their return correlation with it."
        )
    lines.append(
        f"Market: {s['episode_label']} ({s['start']} to {s['end']}) × {s['multiple']:.2f}: "
        f"Baa {s['d_baa_bp']:+.0f} bp, Aaa {s['d_aaa_bp']:+.0f} bp, USD/INR "
        f"{100 * s['fx_move']:+.1f}%, market factor z = {s['z']:.2f}."
    )
    worst = max(rep["obligors"], key=lambda o: o["loss"])
    lines.append(
        f"Result on the {rep['book_date']} book: loss {_m(c['loss_usd'])} before tax "
        f"(largest: {worst['ticker']} {_m(worst['loss'])}). CET1 {_pct(c['cet1_before'])} → "
        f"{_pct(c['cet1_after'])}, floor {_pct(c['floor'])}, headroom "
        f"{100 * (c['cet1_after'] - c['floor']):.2f} pt. RWA ₹{c['rwa_before_inr'] / 1e7:,.0f} cr "
        f"→ ₹{c['rwa_after_inr'] / 1e7:,.0f} cr at USD/INR {c['fx_after']:.2f}."
    )
    stage2 = rep["stages"].get("2", 0) + rep["stages"].get("3", 0)
    lines.append(
        f"IFRS 9: {stage2} of {sum(rep['stages'].values())} revolvers in Stage 2 or 3. "
        f"Default loss, {mc['draws']:,} correlated draws: {mc['level']:.0%} ES "
        f"{_m(mc['before']['es'])} → {_m(mc['after']['es'])}, VaR {_m(mc['before']['var'])} → "
        f"{_m(mc['after']['var'])}."
    )
    k = rep["reverse_stress"]
    lines.append(
        f"Reverse stress: CET1 reaches {_pct(c['floor'])} at {k:.2f}× this episode."
        if k is not None
        else "Reverse stress: CET1 stays above the floor even at 20× this episode."
    )
    lines.append(
        "Limits: FRED spreads by rating bucket, not company spreads; junk spreads understated; "
        "market risk of CDS and swaps not modelled."
    )
    return "\n\n".join(lines)
