"""Module A placebo test: does the sentiment tilt beat a random tilt of the same size?

uv run python -m riskengine.module_a_placebo   # data/outputs/module_a_placebo.json

Each placebo keeps the real daily sentiment values but shuffles which stock gets which value on
each day (seeded). Same weights rule, same cap, same costs. If the real return or Sharpe is not
above most placebos, the tilt has not shown it adds value over chance. Run once, not tuned.
"""

import json

import numpy as np
import pandas as pd

from riskengine.config import load_config
from riskengine.ingest.prices import load_prices
from riskengine.module_a import backtest, daily_signal, target_weights
from riskengine.paths import DATA
from riskengine.pipeline import load_signals, subjects

N_PLACEBO, SEED = 1000, 29


def placebo_metrics(signal: pd.DataFrame, closes: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    vals = signal.to_numpy()
    for _ in range(n):
        shuffled = np.array([rng.permutation(r) for r in vals])
        sig = pd.DataFrame(shuffled, index=signal.index, columns=signal.columns)
        rows.append(backtest(target_weights(sig, "sentiment"), closes).metrics)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    prices = load_prices()
    closes = prices.pivot(index="date", columns="ticker", values="close").drop(
        columns="^GSPC", errors="ignore"
    )
    start, end = pd.Timestamp(cfg.window_start), pd.Timestamp(cfg.window_end)
    closes = closes.loc[(closes.index >= start) & (closes.index <= end)]
    signal = daily_signal(subjects(load_signals()), closes.index).reindex(
        columns=closes.columns, fill_value=0.0
    )
    real = backtest(target_weights(signal, "sentiment"), closes).metrics
    equal = backtest(target_weights(signal, "equal"), closes).metrics
    plc = placebo_metrics(signal, closes, N_PLACEBO, SEED)
    out = {
        "n_placebo": N_PLACEBO, "seed": SEED,
        "real": real, "equal_weight": equal,
        "placebo_mean": plc.mean().to_dict(),
        "placebo_5_95": {
            c: [float(plc[c].quantile(0.05)), float(plc[c].quantile(0.95))] for c in plc
        },
        "p_total_return_ge_real": float((plc["total_return"] >= real["total_return"]).mean()),
        "p_sharpe_ge_real": float((plc["sharpe"] >= real["sharpe"]).mean()),
        "sentiment_minus_equal_return": real["total_return"] - equal["total_return"],
    }  # fmt: skip
    path = DATA / "outputs" / "module_a_placebo.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
