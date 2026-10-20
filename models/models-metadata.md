# Models metadata

The five models that ship in `models/`. All load offline from this folder; nothing is downloaded at run time.
Numbers are copied from each model's `source_metrics.json` (training-time dev scores, not test scores).
Test results are in `docs/experiments/EXPERIMENTS.md`.

| Folder | Task | Output | Size |
|---|---|---|---:|
| `sentiment/` | Headline sentiment | negative / positive / neutral | 34 MB |
| `topic/` | Event class (9 learned classes) | Geopolitical, Macroeconomic, M&A, Product news, Earnings, Analyst rating, Legal/Regulatory, Management change, Other | 34 MB |
| `e25-student-v3/` | Event class, student model (running engine) | same 9 classes | 34 MB |
| `e27-entity-bge-small-en-v1.5-int8/` | Company-specific sentiment | negative / neutral / positive | 34 MB |
| `impact-v3/` | Impact score (1–10 style signal strength) | learned score | 17 KB |

Folders `kaggle_v1` to `kaggle_v5` (training checkpoints, about 2.8 GB) and `data/models/` are **not** in the repository and are not used by the engine.

## Shared architecture (first four models)

- Backbone `BAAI/bge-small-en-v1.5`: BERT, 12 layers, hidden size 384, 12 attention heads, feed-forward 1,536, vocabulary 30,522, up to 512 positions (about 33.4 M parameters).
- Head: `BertForSequenceClassification` (pooled output, dropout, linear layer to the class count).
- Exported to ONNX with int8 quantisation, run with ONNX Runtime on CPU. Tokenizer: `tokenizer.json` in each folder.
- Each folder holds `model.onnx`, `config.json`, `tokenizer.json`, `tokenizer_config.json` (not the entity model), `source_metrics.json`.

## 1. `sentiment/`

| | |
|---|---|
| Used for | Sentiment score of every headline and tweet (Module A signal, Analyze page, Live, Replay). |
| Labels | 0 negative, 1 positive, 2 neutral. |
| Training | Fine-tuned on a Tesla T4 (about 1 minute): 8,588 train rows, 955 dev rows; seed 42, max length 64, batch 32, 4 epochs, lr 5e-5, warm-up 0.1. |
| Dev score | Best dev macro-F1 0.8521 (epoch 2). |
| Test results | E1: acc 0.877, macro-F1 0.836 on the TFNS test (E1, E9 for int8). |
| Rules | `docs/experiments/E05_FINE_TUNING_SETUP.md`, `E09_INT8_ONNX_VERSION.md`. |

## 2. `topic/`

| | |
|---|---|
| Used for | The learned event model (E16): nine event classes, used with the credit rules. |
| Labels | 0 Geopolitical, 1 Macroeconomic, 2 M&A, 3 Product news, 4 Earnings, 5 Analyst rating, 6 Legal/Regulatory, 7 Management change, 8 Other. |
| Training | 13,226 train rows, 1,471 dev rows (public topic data mapped to nine classes). |
| Dev score | Best dev macro-F1 over the nine classes 0.8478. Analyst rating scored 0.0 in the early epochs. |
| Test results | E16: int8 macro-F1 0.841; E2 for the earlier four-class test. |
| Rules | `docs/experiments/E16_NINE_LEARNED_EVENT_CLASSES.md`. |

## 3. `e25-student-v3/` (running event model)

| | |
|---|---|
| Used for | Event classes in the running engine: `events = "student_rules"` in `config/default.toml` (this model plus the frozen rules C). |
| Labels | 0 M&A, 1 Geopolitical, 2 Macroeconomic, 3 Product news, 4 Earnings, 5 Analyst rating, 6 Legal/Regulatory, 7 Management change, 8 Other (note: a different order from `topic/`). |
| Training | Student distilled from a teacher's labels on a pool of real news (seed 43, 4 epochs, best epoch 3). |
| Dev score | Valid macro-F1 0.8721, accuracy 0.9184. |
| Test results | E44b sealed real-news test (822 rows, AI-only labels): macro-F1 0.669 against 0.604 for the previous running system (C). It was the only candidate to pass the preregistered rule, so the engine was switched to it. |
| Rules | `docs/experiments/E25_TEACHER_STUDENT.md`, `E44_EVENT_TEST_V2.md`, `E44_SCORING_PLAN.md`. |

## 4. `e27-entity-bge-small-en-v1.5-int8/`

| | |
|---|---|
| Used for | Sentiment for one company inside a headline that mentions several (entity-aware sentiment, E27). |
| Labels | 0 negative, 1 neutral, 2 positive (a different order from `sentiment/`). |
| Training | Seed 44, 4 epochs, best epoch 4. |
| Dev score | Dev macro-F1 0.8832, accuracy 0.8847. |
| Test results | E27 (SEntFiN test): rows with conflicting sentiment macro-F1 0.476 to 0.821; all rows 0.675 to 0.864. |
| Rules | `docs/experiments/E27_ENTITY_SENTIMENT.md`. |

## 5. `impact-v3/`

| | |
|---|---|
| Files | `impact_v3.joblib` (news), `impact_v3_tweets.joblib` (tweets). Small scikit-learn models stored with joblib. |
| Used for | Learned impact score on top of sentiment, event and market features (volatility-scaled target). |
| Training | Choices made only on rolling year folds (2011, 2015–2018); fresh 2019 and 2020 scored once. |
| Test results | E28: beats the volatility-only baseline in 4 of 4 test sets; small margin in 2020 (0.353 vs 0.331). Ledger verdict: research model. |
| Rules | `docs/experiments/IMPACT_V3.md`. |

## Notes

- Class order differs between `topic/`, `e25-student-v3/` and the sentiment models. Always read `id2label` in `config.json`; do not assume an order.
- Dev scores here are from training time. Do not quote them as test accuracy.
