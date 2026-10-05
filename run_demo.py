"""One-command demo: engine signals -> Module A backtest, on the committed sample.

uv run python run_demo.py              # replay stored signals (fast, no model download)
uv run python run_demo.py --rescore    # re-score the sample with the configured models
uv run python run_demo.py --smoke      # CI: score a few texts live + run Module A
"""

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from riskengine.config import DEFAULT_PATH, load_config
from riskengine.ingest.prices import load_prices
from riskengine.module_a import run_module_a
from riskengine.nlp.sentiment import get_scorer
from riskengine.paths import DATA
from riskengine.pipeline import SIGNALS_PATH, load_sample, load_signals, run, score_docs

OUTPUTS = DATA / "outputs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--rescore", action="store_true", help="re-run the models on the sample")
    parser.add_argument("--smoke", action="store_true", help="fast run for CI")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = load_config(args.config)
    print("== Config")
    for name, value in asdict(config).items():
        print(f"  {name:<15}{value}")

    if args.smoke:
        live = score_docs(load_sample().head(16), config, get_scorer(config.sentiment))
        print(f"\n== Smoke: scored {live['doc_id'].nunique()} texts live")
        print(live[["ticker", "sentiment", "event", "impact"]].head().to_string(index=False))

    if args.rescore or not SIGNALS_PATH.exists():
        signals = run(config)
    else:
        signals = load_signals()
        print(f"\n(replaying stored signals from {SIGNALS_PATH.relative_to(DATA.parent)})")

    print(f"\n== Engine: {len(signals)} signals, {signals['doc_id'].nunique()} texts")
    print(signals.groupby("source")["doc_id"].nunique().to_string())
    print("\nEvent classes:")
    print(signals["event"].value_counts().to_string())
    high = signals[signals["impact"] > config.impact_trigger].drop_duplicates("doc_id")
    print(f"\nHigh-impact (> {config.impact_trigger}) texts: {len(high)}. Top 5:")
    top = high.sort_values(["impact", "published_at"], ascending=[False, True]).head(5)
    for r in top.itertuples():
        print(
            f"  {r.published_at:%Y-%m-%d} {r.ticker:<5} {r.event:<17} {r.impact:>2}  {r.text[:70]}"
        )

    results = run_module_a(signals, load_prices(), config)
    metrics = pd.DataFrame({k: bt.metrics for k, bt in results.items()}).T
    print("\n== Module A (window backtest, 5 bp costs, next-day trading)")
    print(metrics.round(4).to_string())

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUTS / "module_a_metrics.csv", float_format="%.6f")
    weights = results[config.rebalancer].weights
    weights.rename_axis("date").to_csv(OUTPUTS / "weights_history.csv", float_format="%.4f")
    print(f"\nSaved weights and metrics to {OUTPUTS.relative_to(DATA.parent)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
