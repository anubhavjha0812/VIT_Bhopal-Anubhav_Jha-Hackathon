# E7: Sentiment teacher and distilled student

Rules fixed 2026-10-09, before any training run. Written down before the experiment was run; unchanged here.

## Rules

- **Leak found in E5 data:** 84 texts in `sent_train.csv` match a text in the E1 test set once
  URLs, case and punctuation are removed. The E5 models were trained with them; their E1 scores
  stay reported as they are, with this caveat. E7 removes them.
- **Training data:** `sent_train.csv` + Financial PhraseBank `Sentences_75Agree` (3,453, CC BY-NC-SA
  3.0) + FiQA-2018 sentiment train and validation splits (`pauri32/fiqa-2018`, 1,063; label
  0 → positive, 1 → neutral, 2 → negative). Every text whose normalised form (lower case, URLs
  removed, non-alphanumerics → space) matches an E1 test text is dropped. The test file is read
  for its `text` column only, never its labels. Texts that appear more than once with different
  labels (FiQA repeats a sentence per aspect) are dropped; remaining duplicates keep one copy. FiQA's test split (150) is not used for anything.
- **Consequence:** Financial PhraseBank can no longer be reported as a test for our models (E1).
- **Dev set:** the same 10% of `sent_train.csv` as E5 (stratified, seed 42), minus the dropped
  rows. PhraseBank and FiQA rows are never in dev, so dev measures the TFNS domain only.
- **Teacher:** `microsoft/deberta-v3-base`, max length 64, batch 32, AdamW lr 2e-5, 4 epochs,
  10% linear warm-up, seeds 42, 43, 44. Each seed keeps its best epoch by dev macro-F1; the seed
  with the best dev macro-F1 is the teacher.
- **Student (distillation):** `BAAI/bge-small-en-v1.5`, same data, dev set and recipe as E5 except
  6 epochs. Loss = 0.5 × cross-entropy on the hard label + 0.5 × T² × KL(teacher ‖ student) at
  temperature T = 2. Teacher probabilities come from the chosen teacher on the training texts.
  Seeds 42, 43, 44; best seed by dev macro-F1.
- **Test:** the chosen teacher and the chosen student are each scored once on E1, with accuracy,
  macro-F1, a 95% bootstrap interval (1,000 resamples, seed 0) and E4 efficiency.
- **Engine default:** the student replaces the E5 bge-small model if its dev macro-F1 is higher.
  This is decided before test scoring.

## Result

- **Question:** Teacher (DeBERTa) + distilled student
- **Outcome:** teacher 0.905 acc (target 0.91 missed); student 0.885, not chosen on tweet dev
- **Verdict:** engine unchanged
- **Stored results:** `eval_results.json`, `models/kaggle_v2/`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
