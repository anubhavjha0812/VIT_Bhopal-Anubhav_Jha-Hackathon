# E5: Fine-tuning setup

Rules fixed 2026-10-06, before any training run. Written down before the experiment was run; unchanged here.

## Rules

- **Training data:** only the *train* files, `sent_train.csv` (9,543) and `topic_train.csv`
  (16,990). The *valid* files used as test sets in E1/E2 are never downloaded by the training
  notebook.
- **Model selection:** 10% of each train file, stratified, seed 42, is held out as a dev set.
  Epoch choice and the final checkpoint use dev macro-F1 only.
- **Backbones:** `ProsusAI/finbert` (accuracy) and `BAAI/bge-small-en-v1.5` (speed), same
  recipe for both: max length 64, batch 32, AdamW lr 3e-5 (bge-small 5e-5), 4 epochs,
  linear warm-up 10%, seed 42.
- **Events:** the topic model predicts the 20 TFN topics; topics are mapped to our classes
  with the E2 table *after* prediction, so training never sees our class names.
- Each fine-tuned model is scored once on the E1/E2 test sets after training ends. Results
  are reported whether better or worse than the baseline.
- **Which model becomes the engine default** (fixed after training, before any test scoring):
  the one with the higher best dev macro-F1; if the two are within 0.01, the smaller one
  (faster on CPU). All trained models are still reported on the test sets.

## Result

- **Question:** Fine-tune bge-small and FinBERT on TFNS / TFN-topic
- **Outcome:** bge-small sentiment 0.885 acc; FinBERT 0.893
- **Verdict:** bge-small chosen (size/speed)
- **Stored results:** `eval_results.json`, `models/kaggle_v1/`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
