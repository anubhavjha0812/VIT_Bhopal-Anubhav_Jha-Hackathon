# E3: Gold set

Rules sampling fixed 2026-10-06, before any credit-class model is trained. Written down before the experiment was run; unchanged here.

## Rules

- **Pool:** FNSPID headlines for our 15 stocks, 2019-01-01 to 2020-06-10, after cleaning
  (`ingest/news.py`).
- **Sample:** 300 headlines, seed 7. Half (150) from headlines containing any broad
  credit-related word (rating, Moody's, Fitch, downgrade, default, bankrupt, Chapter 11,
  covenant, lender, debt, bond, credit, loan), half (150) at random from the rest. The
  credit-word half exists only so rare credit classes appear at all; it is broader than the
  keyword rules on purpose, so it does not favour them.
- **Labelling:** by hand, one of our 8 classes per headline, from `data/gold/gold_to_label.csv`
  which shows no model output and no keyword guess. Unsure → "Other".
- **Frozen** once labelled (`data/gold/gold.csv`). These 300 headlines are excluded from every
  training set (including any Gemma-labelled data) and are scored once per final model.
- **Metric:** macro-F1 over the 8 classes, plus per-class precision, recall and support.

## Result

- **Question:** Real-headline development reference for events
- **Outcome:** 300 AI-resolved v6 labels, D1 153 / D2 147; no human/expert validation
- **Verdict:** development reference available
- **Stored results:** private_notes/annotation_uploads/task1/reference_v6/, e24_ai_dev_baseline_v6.md

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
