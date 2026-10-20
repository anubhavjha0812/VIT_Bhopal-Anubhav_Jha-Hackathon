# E10: One shared encoder with two heads

Rules fixed 2026-10-09, before any training run. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** the engine runs a sentiment model and a topic model on every text. One encoder with a
  sentiment head and a topic head gives both in one pass. Those two outputs are also the
  headline inputs of the impact model (sentiment, |sentiment|, event, topic probabilities).
- **Model:** `BAAI/bge-small-en-v1.5` encoder, first-token vector, dropout 0.1, one linear head
  for the 3 sentiment classes and one for the 20 TFN topics.
- **Leak found in E5 topic data:** 999 texts in `topic_train.csv` (903 train, 96 dev) match an
  E2 test text after the E7 normalisation. The E5 topic models were trained with them; their E2
  scores stay reported as they are, with this caveat.
- **Data:** sentiment = the E7 training data and dev set, with the E7 teacher's probabilities;
  topic = `topic_train.csv` with the E5 dev split (10%, stratified, seed 42). Topic texts that
  match an E2 test text after the E7 normalisation, or repeat with different labels, are
  dropped, as in E7.
- **Training:** each step takes one sentiment batch and one topic batch (the smaller set is
  cycled); loss = E7 sentiment loss (0.5 hard + 0.5 × T² × KL, T = 2) + topic cross-entropy.
  Batch 32, max length 64, AdamW lr 5e-5, 6 epochs, 10% warm-up, seeds 42, 43, 44.
- **Choice:** per seed, the epoch with the best mean of sentiment and topic dev macro-F1; then
  the seed with the best such mean.
- **Test:** scored once on E1 (sentiment head) and E2 (topic head, mapped with the E2 table),
  with E7 metrics and E4 efficiency. Compared with the single-task bge-small models (E7 student
  for sentiment, E5 bge-small topic model).
- **Success:** accuracy within 1 point of the single-task model on both tests. Reported either way.

## Result

- **Question:** One encoder, two heads
- **Outcome:** sentiment 0.886 acc, topic 0.950 acc
- **Verdict:** reported, not engine
- **Stored results:** `eval_results.json`, `models/kaggle_v3/`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
