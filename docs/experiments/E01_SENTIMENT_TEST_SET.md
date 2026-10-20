# E1: Sentiment test set

Rules fixed 2026-10-06. Written down before the experiment was run; unchanged here.

## Rules

- **Set:** Twitter Financial News Sentiment (`zeroshot/twitter-financial-news-sentiment`,
  MIT), file `sent_valid.csv`, all 2,388 rows. Never used for training or tuning.
- **Labels:** Bearish → negative, Bullish → positive, Neutral → neutral.
- **Prediction:** the model's most likely of the three classes (argmax).
- **Metrics:** accuracy and macro-F1 over the three classes, plus per-class F1.
- **Not used to test FinBERT:** Financial PhraseBank, because FinBERT was trained on it.
  It may be reported for our own model only, with that caveat.

## Result

- **Question:** Sentiment on a public tweet test (TFNS, 2,388)
- **Outcome:** engine int8 acc 0.877, macro-F1 0.836 [0.818, 0.853]; FinBERT baseline 0.726 / 0.669
- **Verdict:** beats baseline
- **Stored results:** `data/outputs/eval_results.json`, `docs/RESULTS.md`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
