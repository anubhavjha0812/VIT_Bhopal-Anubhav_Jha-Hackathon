# E11: Engine switch to our models

Rules fixed 2026-10-09, before the topic int8 model is scored. Written down before the experiment was run; unchanged here.

## Rules

- **Sentiment:** the E9 int8 model (bge-small fine-tuned, int8 ONNX).
- **Events:** the E5 bge-small topic model, exported to int8 as in E9, replaces the E5 choice
  (FinBERT topic). This choice is made after the E2 test scores were seen (0.960 vs 0.966), so
  it is stated as a trade: 0.6 points of accuracy for a model about 1/13 the size and ~3.5×
  the speed, which the engine needs on CPU. Both stay reported.
- **Topic int8 test:** scored once on E2 with E9 metrics. It is used only if its E2 accuracy is
  within 1 point of the bge-small torch topic model; otherwise the torch model is used.
- **Impact:** stays the formula (E8 v2 is not used; it failed the 2020 test).
- **Demo:** the committed sample is re-scored once with the new models; signal counts, event
  mix, high-impact count and Module A metrics are reported next to the baseline run.

## Result

- **Question:** Switch the engine to our models
- **Outcome:** demo re-scored, 89.7% same event class
- **Verdict:** switched
- **Stored results:** `signals.jsonl.gz`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
