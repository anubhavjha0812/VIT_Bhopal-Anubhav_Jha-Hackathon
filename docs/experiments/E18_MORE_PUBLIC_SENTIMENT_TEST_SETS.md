# E18: More public sentiment test sets

Rules fixed 2026-10-11, before any of them is scored. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** E1 is one test set. Scores on the other widely used public sets let our engine be
  compared with other published results on identical files, not only on our own choice of data.
- **Sets** (each scored once per model, all rows unless dropped below):

  | Set | File | Rows | Labels |
  |---|---|---|---|
  | Financial PhraseBank | `takala/financial_phrasebank` v1.0, `Sentences_50Agree` and `Sentences_AllAgree` | 4,846 / 2,264 | as given |
  | FiQA-2018 sentiment | `pauri32/fiqa-2018`, test split | 150 | 0 → positive, 1 → neutral, 2 → negative (as E7) |
  | NWGI | `oliverwang15/news_with_gpt_instructions`, test split, column `news` | 4,047 | any positive → positive, neutral, any negative → negative |

- **Label caveat:** NWGI's labels were written by a GPT model, not by people. It is reported as
  a check against a widely used set, never as human-verified accuracy.
- **Models:** the engine model (E5 bge-small, int8, E9), its full-precision copy, the E5
  fine-tuned FinBERT, the FinBERT baseline, and the E7 teacher, E7 student and E10 shared encoder. Prediction and metrics as E1
  (argmax, accuracy, macro-F1, per-class F1, 95% bootstrap intervals with 1,000 resamples and
  seed 0), plus E4 efficiency. Input is cut at the models' 64 tokens, as in the engine.
- **Leak check (before any scoring):** each test text is compared with every text the model
  was trained on: `sent_train.csv` for the E5 models; plus PhraseBank `Sentences_75Agree` and
  FiQA train and validation for the E7 and E10 models; PhraseBank for both FinBERT models (their
  base model was fine-tuned on it).
  - Exact: the E7 normalised forms are equal.
  - Near: cosine similarity of word 1–2-gram TF-IDF vectors (fit on the two sets) ≥ 0.9.
  - Test rows with an exact or near match are counted, and the score is given on all rows and
    on the clean rows. The clean rows are the primary figure.
  - A model whose training set contains the test set itself is marked **in-sample**: the E7, E10
    and both FinBERT models on PhraseBank. Their PhraseBank scores are shown with that mark and are
    never counted as a result.
- **Claims:** "higher than" another published figure is written only when the files, rows and
  label mapping are the same and our 95% interval lies wholly above that figure. Otherwise the
  result is "within the interval" (a tie), or the comparison is "not like for like".
- **Engine:** these scores change nothing in the engine. They are reported as they come.
- **Amendment (2026-10-11, after the engine rows, before any comparison with another
  figure):** overlapping intervals are not by themselves a tie. When per-row predictions exist
  for both sides (our own models, or a public model we can run on the same rows), the
  comparison is a paired bootstrap of the macro-F1 difference on the same resampled rows
  (1,000 resamples, seed 0): "higher" only if its 95% interval lies above 0, "lower" if below,
  otherwise a tie. When only a published number exists, the rule above applies. Per-row
  predictions are now saved (`data/outputs/e18_predictions.csv.gz`), and what was scored is
  frozen in `data/outputs/e18_manifest.json` (dataset revisions, file and model hashes, library
  versions). Changes to input length or training data are separate experiments, each fixed in
  advance; E18 only measures.
- **Results, engine models (2026-10-11).** Leak check: the E5 models saw 0 rows of all four
  sets; the E7 and E10 models saw all of PhraseBank AllAgree, 3,461 of 4,846 PhraseBank 50Agree
  rows and 1 FiQA test row; both FinBERT models saw all of PhraseBank. Macro-F1, 95% interval:

  | Set | Engine (int8) | Full precision |
  |---|---|---|
  | PhraseBank AllAgree (2,264) | 0.866 [0.849, 0.881] | 0.880 [0.863, 0.895] |
  | PhraseBank 50Agree (4,846) | 0.685 [0.670, 0.700] | 0.702 [0.686, 0.717] |
  | FiQA-2018 test (150) | 0.573 [0.486, 0.649] | 0.602 [0.521, 0.676] |
  | NWGI test (4,047, GPT labels) | 0.406 [0.390, 0.419] | 0.414 [0.399, 0.428] |

  What this does and does not show:
  - *Measured:* no training overlap, so these are clean held-out scores.
  - *Reading:* a model trained only on tweets keeps 0.87 on the sentences annotators fully agree
    on, a promising sign of transfer from tweets to news sentences.
  - *Limitation:* long texts. NWGI is news paragraphs and the model reads 64 tokens; its
    neutral class scores 0.28 F1.
  - *Limitation:* FiQA has 150 rows, so its interval is about 16 points wide.
  - *Limitation:* NWGI labels come from a GPT model; they are not a human-labelled benchmark.
  - *Reading:* int8 costs 1–3 macro-F1 points against full precision on every set, for a
    smaller CPU-only model.
- **Results, all models (2026-10-11, Kaggle job `notebooks/e18_sentiment`, per-row predictions
  saved; model weights checked against the repo's sha256).** The int8 rows were re-run there on
  CPU and agree with the laptop run within 0.001 macro-F1; speed figures are the laptop's.
  Macro-F1 on rows each model never saw in training (all rows unless a count is given):

  | Model | PhraseBank AllAgree | PhraseBank 50Agree | FiQA test | NWGI test |
  |---|---|---|---|---|
  | Engine, int8 (E5, TFNS only) | 0.865 | 0.685 | 0.573 | 0.406 |
  | Same, full precision | 0.880 | 0.702 | 0.602 | 0.414 |
  | E5 fine-tuned FinBERT | in-sample | in-sample | 0.617 | 0.474 |
  | FinBERT off the shelf | in-sample | in-sample | 0.562 | 0.554 |
  | E7 distilled bge-small | in-sample | 0.637 (1,385) | 0.676 (149) | 0.541 |
  | E7 DeBERTa teacher | in-sample | 0.657 (1,385) | 0.705 (149) | 0.574 |
  | E10 shared encoder | in-sample | 0.652 (1,385) | 0.641 (149) | 0.558 |

  Paired bootstrap, macro-F1 difference on the same clean rows (`data/outputs/e18_paired.json`):
  - Full precision vs int8: +0.016 [0.010, 0.023] on 50Agree, +0.015 [0.008, 0.023] on AllAgree,
    +0.008 [0.001, 0.015] on NWGI (all **higher**); FiQA +0.029 [−0.021, 0.082], a tie.
  - Engine vs FinBERT off the shelf: FiQA tie (+0.011 [−0.080, 0.097]); NWGI **lower**
    (−0.148 [−0.162, −0.133]).
  - Engine vs the E7 distilled model, same size: FiQA **lower** (−0.106 [−0.190, −0.015]), NWGI
    **lower** (−0.135 [−0.146, −0.123]). Engine vs teacher: lower on both.
  - *Reading:* the engine model, trained on tweets only, transfers worse to news text than the
    models that also saw PhraseBank and FiQA training rows. These test sets are not used to pick
    a new engine model: any switch is a new experiment, chosen on a dev set fixed in advance.

## Result

- **Question:** Engine sentiment on 4 more public sets, leak-checked
- **Outcome:** 0 rows seen; PhraseBank AllAgree 0.865, 50Agree 0.685, FiQA 0.573, NWGI 0.406; distilled model better on news (paired)
- **Verdict:** reported
- **Stored results:** `eval_results.json`, `e18_*.json`, `e18_predictions.csv.gz`, `leak_check.json`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
