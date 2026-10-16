# E24 D1 inference reproducibility

D1 only (153 AI-reference headlines). No D2/final rows, tuning, selection, freeze or commit.
The previous batch-64 results and probability cache are preserved under their original names.

## Diagnosis

The table compares each old production-style setting with per-row inference using the same
padding policy. Shuffled rows are restored to their original order before comparison.

| Padding | Batch | Order | Top-1 changes /153 | Max absolute probability change |
|---|---:|---|---:|---:|
| longest_in_batch | 1 | original | 0 | 0.00000000 |
| longest_in_batch | 1 | shuffle_seed_24 | 0 | 0.00000000 |
| longest_in_batch | 8 | original | 12 | 0.56076890 |
| longest_in_batch | 8 | shuffle_seed_24 | 12 | 0.36519045 |
| longest_in_batch | 16 | original | 9 | 0.52071702 |
| longest_in_batch | 16 | shuffle_seed_24 | 12 | 0.47383413 |
| longest_in_batch | 64 | original | 13 | 0.36822882 |
| longest_in_batch | 64 | shuffle_seed_24 | 9 | 0.31834957 |
| max_length_64 | 1 | original | 0 | 0.00000000 |
| max_length_64 | 1 | shuffle_seed_24 | 0 | 0.00000000 |
| max_length_64 | 8 | original | 10 | 0.40611368 |
| max_length_64 | 8 | shuffle_seed_24 | 14 | 0.44904298 |
| max_length_64 | 16 | original | 6 | 0.37695003 |
| max_length_64 | 16 | shuffle_seed_24 | 14 | 0.42918783 |
| max_length_64 | 64 | original | 8 | 0.54304242 |
| max_length_64 | 64 | shuffle_seed_24 | 7 | 0.58400917 |

The ONNX graph contains 50 DynamicQuantizeLinear nodes. An in-memory diagnostic
exposes the first activation scale; no model file was edited. For the identical first
headline padded to 64 tokens, the scale changes when 63 other D1 headlines are added:
0.0482209771872 -> 0.048241507262. That row's max probability difference is 0.00704750, with both ORT thread counts 1.
This directly confirms batch-dependent activation quantization; fixed padding alone fails.

Padding also matters: singleton fixed-64 padding versus singleton natural length changes
13 top-1 predictions (max probability difference 0.57473820). Both mechanisms are controlled
by passing exactly one row to ONNX and tokenizing that row independently.

## Fixed evaluation method

`onnx_cpu_per_row_dynamic_padding_v1`; CPU, batch 1, natural per-row padding, truncation 64,
ORT sequential execution, intra-op 1 and inter-op 1. Same model and candidate logic.
Every tested external grouping (1/8/16/64) and original/shuffled order produces bit-identical
probabilities for all 153 rows: zero changed top-1 predictions and max difference 0.
A separately constructed session also matches bit for bit. Because each graph invocation
sees only that headline's tokens, other requested rows cannot affect its activation scales.
The tiny dynamic-int8 ONNX regression test additionally covers grouping, order and sessions.

New caches require an exact method/runtime/assets match, file SHA and row-order check.
Legacy caches without method metadata fail closed. Actual future freezes record method,
batch, ORT version, both thread counts, model/tokenizer hashes and softmax versions; candidate
validation rechecks them. The evaluation module is pinned with the other inference sources.

## D1 candidate comparison

Fixed 12-class macro-F1 is primary (absent classes score zero). D uses the unchanged 0.50
demonstration threshold, temperature None. Neither is tuned and no candidate is selected.

| Candidate | Previous macro-F1 | Reproducible macro-F1 | Previous accuracy | Reproducible accuracy | Changed/fixed/broken |
|---|---:|---:|---:|---:|---|
| A | 0.317148 | 0.321843 | 43.14% (66/153) | 43.79% (67/153) | 13/5/4 |
| B | 0.451667 | 0.454848 | 85.62% (131/153) | 86.93% (133/153) | 3/2/0 |
| C | 0.510977 | 0.515636 | 90.85% (139/153) | 92.16% (141/153) | 2/2/0 |
| D | 0.514798 | 0.517217 | 92.16% (141/153) | 92.81% (142/153) | 1/1/0 |

Rule code was not changed or retuned. Every rule override has the identical label and
source as before. Candidate differences occur only where model-dependent decisions apply.
Product news remains 4 predictions / 4 correct / 4 reference for B–D. M&A remains 0 predictions
/ 2 reference: both positives are still missed. Tiny support limits recall conclusions.
No rule-plateau claim or new model training follows from this inference repair.

## CPU throughput and production difference

153 D1 texts, three timed warm repetitions per method, same process and machine; excludes
load/warm-up. Headline lengths and CPU conditions limit generalization of these timings.

| Path | Texts/sec | Median seconds |
|---|---:|---:|
| evaluation_per_row_pinned_threads | 50.53 | 3.0280 |
| production_batch64_default_threads | 54.16 | 2.8248 |

Evaluation/production elapsed-time ratio: 1.072 (7.2% more elapsed time here).
This comparison includes pinned threads and the evaluation tokenizer path, as well as batch size.
The raw batch-only control below uses one production-style session and identical settings.

| Raw batch-only control | Texts/sec | Median seconds |
|---|---:|---:|
| Batch 1, runtime-default threads | 99.94 | 1.5309 |
| Batch 64, runtime-default threads | 44.05 | 3.4732 |

Natural singleton padding does less work than padded batches; speed depends on headline lengths
and thread settings. Both measurements are provided instead of attributing all cost to batch size.

Production retains batch 64 and runtime-default threads. On D1 it differs on 13/153
model top-1 labels and 13/153 final A labels; max probability difference
0.36822882. These differences are material, not merely rounding.
Production results remain batch-dependent and may differ from the frozen evaluation method.
A production change is a separate decision; no nlp/ implementation was edited.

## Artifacts and limits

- e24_d1_batch_diagnosis.json/.npz: old batching, padding, scale probe and timings.
- e24_d1_model_probabilities_reproducible.npz/.json: new validated cache and method/assets.
- e24_d1_rules_reproducible.json and _predictions.csv: A–D scores and all D1 decisions.
- e24_d1_reproducibility.json and _changes.csv: exact checks, comparisons and changed rows.
- e24_d1_reproducible_per_class.csv and _confusion.csv: precision/recall and fixed-12 matrices.
- private_notes/annotation_uploads/task1/e24_rules_work/: D1-only runners and snapshots.

Reference v6 is AI-resolved, not human/expert validated. Development gains require D2 selection
and a later final evaluation. This work stops before D2. DECISIONS E24 hash remains
bfffa0ad37366c8a; labels, weights, production sources and E23 gate snapshot are unchanged.
