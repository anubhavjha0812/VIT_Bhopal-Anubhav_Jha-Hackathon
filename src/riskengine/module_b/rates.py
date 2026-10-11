"""E22: the interest-rate shock, run once (docs/DECISIONS.md, E22).

uv run python -m riskengine.module_b.rates   # data/outputs/e22_*.json / .csv

Per scenario, rates fixed vs moved: loss split into its credit part (rates fixed) and its rate
part (the difference), CET1 after and the reverse-stress multiple; the +200 bp scenario alone;
and Tests 2 and 3 re-run with the Treasury move (reported, never used to decide). The E14 and
E15 output files are not touched.
"""

import json

import numpy as np

from riskengine.module_b.backtest import CRISES_PATH, OUT, crisis_backtest
from riskengine.module_b.book import load_book
from riskengine.module_b.outlook import MONTHLY_PATH, monthly_backtest
from riskengine.module_b.outlook import summary as monthly_summary
from riskengine.module_b.revalue import revalue, reverse_stress, treasury_after
from riskengine.module_b.shock import EPISODES, RATES_UP, market, rates_up_market

RESULTS = OUT / "e22_rates.json"
CRISES = OUT / "e22_crisis_backtest.csv"
MONTHLY = OUT / "e22_monthly_backtest.csv"


def scenario_rows(book) -> dict:
    out = {}
    for name in sorted(EPISODES):
        fixed = revalue(book, None, 0, name, 1.0, "fixed")
        moved = revalue(book, None, 0, name, 1.0, "moved")
        by = (
            moved.positions.groupby("instrument")["loss"].sum()
            - fixed.positions.groupby("instrument")["loss"].sum()
        )
        out[name] = {
            "d_ust10y_pp": market(name).d_rate,
            "treasury_after_pct": 100 * treasury_after(market(name), 1.0, "moved"),
            "loss_rates_fixed": fixed.loss_usd,
            "loss_rates_moved": moved.loss_usd,
            "rate_part": moved.loss_usd - fixed.loss_usd,
            "rate_part_by_instrument": by.round(2).to_dict(),
            "cet1_after_rates_fixed": fixed.cet1_after,
            "cet1_after_rates_moved": moved.cet1_after,
            "reverse_stress_rates_fixed": reverse_stress(book, None, 0, name, rates="fixed"),
            "reverse_stress_rates_moved": reverse_stress(book, None, 0, name, rates="moved"),
        }
    up = revalue(book, None, 0, rates_up_market(RATES_UP), 1.0, "moved")
    out["rates_up_200bp"] = {
        "d_ust10y_pp": RATES_UP,
        "loss": up.loss_usd,
        "loss_by_instrument": up.positions.groupby("instrument")["loss"].sum().round(2).to_dict(),
        "cet1_before": up.cet1_before,
        "cet1_after": up.cet1_after,
    }
    return out


def tests_rerun(book) -> dict:
    import pandas as pd

    crises = crisis_backtest(book, rates="moved")
    crises.to_csv(CRISES, index=False)
    monthly = monthly_backtest(book, rates="moved")
    monthly.to_csv(MONTHLY)
    e14 = pd.read_csv(CRISES_PATH)
    e15 = pd.read_csv(MONTHLY_PATH, index_col=0)

    def mae(df: pd.DataFrame) -> dict:
        return {k: float(df[f"{k}_loss_error"].mean()) for k in ("predicted", "naive")}

    m22, m15 = monthly_summary(monthly), monthly_summary(e15)
    keys = ("model_mae", "naive_mae", "ratio", "model_sign", "naive_sign", "dm_p")
    return {
        "test2_crises": {"rates_fixed_E14": mae(e14), "rates_moved_E22": mae(crises)},
        "test3_monthly": {
            "rates_fixed_E15": {k: float(m15[k]) for k in keys},
            "rates_moved_E22": {k: float(m22[k]) for k in keys},
        },
    }


def main() -> None:
    book = load_book()
    res = {"rule": "docs/DECISIONS.md E22", "scenarios": scenario_rows(book),
           "reported_only": tests_rerun(book)}  # fmt: skip
    RESULTS.write_text(json.dumps(res, indent=2, default=lambda x: float(x) if np.isfinite(x)
                                  else None))  # fmt: skip
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
