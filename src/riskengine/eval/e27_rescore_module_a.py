"""E27 follow-up: re-score stored signals with entity sentiment, then Module A and its placebo.

uv run python -m riskengine.eval.e27_rescore_module_a

Writes data/outputs/signals_entity.jsonl.gz and module_a_placebo_entity.json. The stored
signals and the earlier placebo file are not touched. Run once, not tuned.
"""

import json

import pandas as pd

from riskengine.ingest.linking import Linker
from riskengine.ingest.prices import load_prices
from riskengine.module_a import backtest, daily_signal, target_weights
from riskengine.module_a_placebo import N_PLACEBO, SEED, placebo_metrics
from riskengine.nlp.sentiment import get_scorer
from riskengine.paths import DATA
from riskengine.pipeline import _entity_scores, _impacts, load_signals, subjects


def main() -> None:
    from riskengine.config import load_config

    signals = load_signals()
    signals["sentiment"] = _entity_scores(signals, get_scorer("entity"), Linker.from_csv())
    signals["impact"] = _impacts(signals)
    signals["model"] = "entity+model_rules+formula"
    out_path = DATA / "outputs" / "signals_entity.jsonl.gz"
    signals.to_json(out_path, orient="records", lines=True, date_format="iso")
    cfg = load_config()
    prices = load_prices()
    closes = prices.pivot(index="date", columns="ticker", values="close").drop(
        columns="^GSPC", errors="ignore"
    )
    start, end = pd.Timestamp(cfg.window_start), pd.Timestamp(cfg.window_end)
    closes = closes.loc[(closes.index >= start) & (closes.index <= end)]
    signal = daily_signal(subjects(signals), closes.index).reindex(
        columns=closes.columns, fill_value=0.0
    )
    real = backtest(target_weights(signal, "sentiment"), closes).metrics
    equal = backtest(target_weights(signal, "equal"), closes).metrics
    plc = placebo_metrics(signal, closes, N_PLACEBO, SEED)
    res = {
        "n_placebo": N_PLACEBO,
        "seed": SEED,
        "real": real,
        "equal_weight": equal,
        "placebo_mean": plc.mean().to_dict(),
        "placebo_5_95": {
            c: [float(plc[c].quantile(0.05)), float(plc[c].quantile(0.95))] for c in plc
        },
        "p_total_return_ge_real": float((plc["total_return"] >= real["total_return"]).mean()),
        "p_sharpe_ge_real": float((plc["sharpe"] >= real["sharpe"]).mean()),
        "sentiment_minus_equal_return": real["total_return"] - equal["total_return"],
    }
    (DATA / "outputs" / "module_a_placebo_entity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
