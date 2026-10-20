# E2: Event-class test set

Rules fixed 2026-10-06. Written down before the experiment was run; unchanged here.

## Rules

- **Set:** Twitter Financial News Topic (`zeroshot/twitter-financial-news-topic`, MIT),
  file `topic_valid.csv`, all 4,117 rows. Never used for training or tuning.
- **Mapping to our classes** (fixed before scoring any model):

  | TFN-topic label | Our class |
  |---|---|
  | Fed \| Central Banks, Macro | Macroeconomic |
  | M&A \| Investments | M&A |
  | Politics | Geopolitical |
  | all other 16 labels | Other |

- Our classes Default, Credit downgrade, Covenant breach and Product launch have no clean
  TFN-topic equivalent, so they are **not** scored here. They are scored on our hand-labelled
  gold set (E3).
- **Metric:** macro-F1 over the four mapped classes, plus per-class precision, recall and F1.

## Result

- **Question:** Event class on a public test (TFN-topic, 4,117, 4 classes)
- **Outcome:** topic int8 acc 0.951, macro-F1 0.886; keywords 0.740 / 0.505
- **Verdict:** beats baseline
- **Stored results:** same

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
