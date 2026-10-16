# Production event inference aligned with E24 evaluation

The production models/topic factory now reuses EvaluationProbs from eval/e24_inference.py.
No numerical inference code was copied. The generic TextClassifier remains batch 64, and
sentiment, other model paths and historical impact scoring retain their existing behavior.
Topic mapping, E13 credit precedence and candidate A logic are unchanged. No B–D rules were
promoted into production. D2, final rows, annotation labels, weights and DECISIONS were not changed.

## D1 parity

All measurements use the same 153 D1 texts and model hash. Raw probabilities are compared
against a fresh evaluation session and the previously saved per-row D1 cache.

| Difference vs evaluation | Before (batch 64) | After (shared per-row) |
|---|---:|---:|
| Model top-1 rows | 13/153 | 0/153 |
| Final A event-label rows | 13/153 | 0/153 |
| Rows with any probability difference | 153/153 | 0/153 |
| Maximum absolute probability difference | 0.36822882 | 0 |

After probabilities are bit-identical, including a seed-24 shuffled request. Final labels
also match after shuffling. No rules, thresholds or temperatures were tuned.

## CPU speed and memory

Separate fresh processes, sequential runs on the same machine; no tests ran during timing.
Each path calls the actual public production event factory. One full D1 warm-up precedes
three timed complete passes; the table uses their median. Peak RSS uses Linux getrusage
ru_maxrss, measured before constructing any evaluation session. It includes interpreter,
imports, the model, warm-up and inference; it is not just model weights or incremental tensor memory.
Sampled current RSS counters are also saved in the JSON. Memory readings are approximate OS counters.

| Measurement | Before | After |
|---|---:|---:|
| Headlines/second | 64.39 | 81.21 |
| Seconds /153 headlines | 2.3761 | 1.8841 |
| Peak production-process RSS (MiB) | 705.39 | 517.34 |
| RSS after warmed inference (MiB) | 706.18 | 518.20 |

Measured throughput changes by +26.1%; elapsed time by -20.7%;
peak production-process RSS changes by -188.04 MiB. This D1 workload has short, varying-length
headlines, so independent padding and pinned thread settings can save work. These figures do not
guarantee throughput for other input lengths, machines or live concurrency. The prior evaluation report
benchmarked a different two-session process; these production measurements use isolated processes.

## Files and method

- src/riskengine/nlp/events.py: get_event_probs() constructs the shared EvaluationProbs; the model
  factory uses it and retains the same topic-to-event mapping and credit override.
- tests/test_e24_inference.py: production/evaluation probabilities and labels, grouping/shuffling,
  generic/sentiment batch defaults, keyword isolation and missing-model failure.
- tests/test_nlp_baseline.py: existing credit-override test mocks the probability factory.
- STEPS.md and docs/EXPERIMENTS.md: record this authorized production change and measurements.
- private_notes/annotation_uploads/task1/e24_production_alignment/: reproducible before/after benchmark,
  report builder, original events.py source and protected-file hashes.
- data/outputs/e24_d1_production_*: before/after JSON, probability arrays, parity rows and this report.

The shared method is onnx_cpu_per_row_dynamic_padding_v1, batch 1, natural per-row padding,
truncation 64, CPU, sequential ORT execution, intra-op 1 and inter-op 1. ORT, torch, transformers
versions and model/tokenizer hashes are in the after JSON. Evaluation implementation and cache remain
unchanged, so previously stored fixed-method D1 results still reproduce exactly.

## Integrity and validation

The E24 locked section stays bfffa0ad37366c8a; model assets and protected generic/sentiment/evaluation
sources are unchanged. An AST comparison against the saved production source confirms all event
definitions outside the factory wiring are identical, including credit rules and topic mappings.
The historical E23 snapshot is preserved. Its events.py whole-file hash now differs because this
authorized change edits that file; the other four recorded code hashes still match. Credit logic
was not changed, and the old snapshot was not rewritten to disguise the authorized wiring change.
Future E24 candidate freezes pin the new production event source normally; no freeze exists yet.

Validation: 557 tests passed, one pre-existing expected failure and one deselected test; repo-wide
ruff check and changed-file formatting pass. Exact results are recorded in
production_alignment_verification.json in the private work folder.
The suite excludes tests/test_e25_teacher.py::test_prompt_examples_are_not_from_any_test_set
because it opens the real 300-row development CSV, including reserved D2. Its source is untouched.
Other development/final gate tests use temporary synthetic fixtures. No real D2/final rows were read.
No candidate selection, freeze, model training, tuning or commit. Stop after this report.
