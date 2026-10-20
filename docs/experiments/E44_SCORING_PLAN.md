# E44b scoring plan

Fixed before predictions, 19 October 2026. The sampling protocol stays unchanged. This is a separate, test-only comparison; the running engine remains frozen E24 C. Only support counts and hashes from labels_final_record.json were read to write this plan. Neither frozen label contents nor any E44 prediction was opened. No model was run.

## Candidates fixed now

- **A**, reference only: reuse `e24_dev.build_predictor(Config("A"), engine probabilities)`. Clarification of the task's “rules-only” wording: existing E24 A is the nine-class event model with the E13 credit override, not a rules-only classifier. Preserve its actual frozen implementation.
- **C**: reuse the frozen candidate, threshold and temperature in `data/outputs/e24_frozen.json` (C, null, null). Use `EvaluationProbs` for models/topic with the exact frozen single-row CPU inference metadata and asset hashes. Do not use batch 64 or a GPU provider.
- **C-noPF**: reuse the same C prediction and the very same cached engine probabilities. Only rows whose C route is `model_product_to_other` recover the raw model top-1 Product news and route `model`. Every other label and route stays identical. Do not copy or edit C routing, regexes or thresholds.
- **E25-v3**: the existing offline v3 student weights are present. Reuse `student_probs` and C routing, with no temperature or threshold. Pin every student asset below. No training or rule change.

All candidates and class lists below are fixed regardless of later results. Keep all eight historical frozen pinned files, including the unused v2 ONNX hash, as integrity checks. The v2 student is not a scored candidate. All model/tokenizer assets, current frozen code, label record, sampling manifest, guide and protocol must match their recorded hashes before prediction and scoring.

## Frozen reference and metrics

822 kept IDs from a 900-row sample; 78 disagreements were dropped by the owner instead of settled. Labels are AI agreement only, with no independent human check. Read only the id column of the frozen labels for prediction; model input comes exclusively from sample.csv. Do not predict dropped IDs. Scoring joins by unique ID and rejects missing, extra or repeated rows. Prediction file order is sample order.

Every class has at least ten gold rows. Fix the primary macro-F1 list to all twelve classes, in the specification order: Default, Credit downgrade, Covenant breach, Geopolitical, Macroeconomic, M&A, Product news, Earnings, Analyst rating, Legal/Regulatory, Management change, Other. Secondary macro-F1 uses these same classes without Other; accuracy uses all 822 rows. Report precision, recall, F1 and full-reference support for all twelve, with zero_division=0. Keep these lists unchanged in every bootstrap draw, even when a resample lacks a class. Do not round values before the decision.

## Intervals

Sort the kept IDs by publication date then id. Read only id and date from provenance.csv with usecols; never parse hint, source, ticker or rank columns. Verifying its bytes against the existing sampling hash is allowed. Validate a parseable date for every kept ID.

Use NumPy default_rng(44), 1,000 moving-block draws, block length ceil(sqrt(822)) = 29. Starts are uniform integers from 0 through n-block_length inclusive. Per draw sample ceil(n/block_length) overlapping contiguous blocks with replacement, concatenate, then truncate to n. Reuse the exact indices for every candidate, primary/secondary macro-F1, accuracy and every class precision/recall/F1. Report point estimates and 2.5/97.5 percentile intervals, plus paired differences versus C with the same intervals. Full-reference support is an integer count, not a performance metric. Save the index matrix SHA-256 to make the bootstrap replayable. No IID replacement or class-list reselection.

## Decision fixed now

C remains the running system. A is reference only and can never qualify. For each challenger separately, report whether primary macro-F1 minus C is **at least 0.02** and the paired 95% interval's lower bound is **strictly greater than zero**. Both conditions are required. Multiple qualifiers are reported separately; there is no automatic winner, engine change or model choice. The owner receives the result. No result is used for tuning, training, rules, thresholds, labels or prompt/model selection.

## Integrity, execution and one run

After this file is written, save its SHA-256 and UTC time on one line in data/outputs/e44_scoring_plan.lock; never rewrite either. Any later departure is described in results as a deviation, never edited here.

Implement a separate scorer and an offline Kaggle notebook. Do not run either prediction or real-label scoring on this laptop. Fixture tests contain invented rows only. Before any model starts, verify the plan lock, the fixed final-label SHA, every sha256 entry in labels_final_record.json, the record's own pinned hash, the sampling provenance/sample hashes and all candidate pins. Compare runtime package versions to the specification, and engine inference metadata to the frozen record. All failures stop closed.

The first --predict exclusively creates data/outputs/e44.run_lock before inference; it refuses an existing lock, prediction CSV or results JSON. Preserve a failed partial run and its lock; do not delete or overwrite to rerun. Write one wide e44_predictions.csv (id and each candidate's prediction/route), bind its SHA, IDs, frozen inputs, scorer and run provenance in the run lock. --score requires that completed prediction phase, validates its hashes, exclusively claims scoring once and writes e44_results.json. A second prediction or scoring call refuses. No lock/result is created by this implementation task. Local files expose tampering; they cannot stop an owner from deleting artifacts or starting another Kaggle session. Retain the entire Kaggle output and use one execution only.

## Limits carried into results

AI-only labels; no independent human validation. Dropping 78 disagreements selects easier agreed rows and deviates from the original settlement procedure. Default has only 28 rows; support is small and sampling is enriched rather than population representative. Headlines are old; multiple headlines can describe one event; temporal blocks cannot guarantee event independence. Source/publisher mix limits transfer to live news. Project-data overlap checks do not certify absence from unknown upstream model pretraining. Report all these limits regardless of outcome.

## Executable specification

```json
{
  "experiment": "E44b",
  "rows": 822,
  "classes": [
    "Default",
    "Credit downgrade",
    "Covenant breach",
    "Geopolitical",
    "Macroeconomic",
    "M&A",
    "Product news",
    "Earnings",
    "Analyst rating",
    "Legal/Regulatory",
    "Management change",
    "Other"
  ],
  "eligible_classes": [
    "Default",
    "Credit downgrade",
    "Covenant breach",
    "Geopolitical",
    "Macroeconomic",
    "M&A",
    "Product news",
    "Earnings",
    "Analyst rating",
    "Legal/Regulatory",
    "Management change",
    "Other"
  ],
  "secondary_classes": [
    "Default",
    "Credit downgrade",
    "Covenant breach",
    "Geopolitical",
    "Macroeconomic",
    "M&A",
    "Product news",
    "Earnings",
    "Analyst rating",
    "Legal/Regulatory",
    "Management change"
  ],
  "support": {
    "Default": 28,
    "Credit downgrade": 69,
    "Covenant breach": 43,
    "Geopolitical": 34,
    "Macroeconomic": 62,
    "M&A": 52,
    "Product news": 48,
    "Earnings": 57,
    "Analyst rating": 53,
    "Legal/Regulatory": 73,
    "Management change": 58,
    "Other": 245
  },
  "candidates": [
    "A",
    "C",
    "C-noPF",
    "E25-v3"
  ],
  "dropped_candidates": [],
  "student_folder": "private_notes/annotation_uploads/task1/e25_student_output/v3/models/e25-student-bge-small-en-v1.5-int8",
  "work_folder": "private_notes/annotation_uploads/task1/event_test_v2",
  "guide": "private_notes/annotation_uploads/task1/development/GUIDE.md",
  "protocol": "docs/experiments/E44_EVENT_TEST_V2.md",
  "frozen": "data/outputs/e24_frozen.json",
  "labels_sha256": "722986b2547f7274be4b29b9f9b5cb090942adee08cf179390feae7542d1177d",
  "bootstrap": {
    "draws": 1000,
    "seed": 44,
    "block_length": 29,
    "sort": [
      "date",
      "id"
    ],
    "percentiles": [
      2.5,
      97.5
    ]
  },
  "decision": {
    "reference": "C",
    "reference_only": [
      "A"
    ],
    "challengers": [
      "C-noPF",
      "E25-v3"
    ],
    "minimum_primary_gain": 0.02,
    "paired_lower_bound_strictly_above": 0,
    "automatic_engine_change": false
  },
  "runtime": {
    "numpy": "2.5.3",
    "pandas": "3.0.6",
    "scikit-learn": "1.9.1",
    "onnxruntime": "1.30.0",
    "torch": "2.14.1+cpu",
    "transformers": "5.18.0"
  },
  "expected_sha256": {
    "data/outputs/e24_frozen.json": "ac05ff894934ece1fa220bf20c6050ebde38ed841e7e8871b140402dce897bec",
    "docs/DECISIONS.md": "97f716563b1e19ba7a83061cb3dffbae67958f893bf70ab45053de93295b36a7",
    "docs/experiments/E44_EVENT_TEST_V2.md": "d44258e8202f4d9e1a00e80f55b8ed5546ea86e80eb4807ccbe6f9e61d04ed45",
    "models/topic/config.json": "37fa9cd6921b2f5d92331a51e896d0eadd501e391bc05c638278022524ad3266",
    "models/topic/model.onnx": "e942bf1c66896cf809f128eb90e51a146ba32e11381e14564193a670c7282db9",
    "models/topic/source_metrics.json": "2dca2b4609638cb3ff6f954d811b57db11ff84560fe4054015e651287809d80a",
    "models/topic/tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "models/topic/tokenizer_config.json": "8ccd7632bce0dc3e48daaef7c854244d4675d7d381d43e56406f3d6c47c3145e",
    "private_notes/annotation_uploads/task1/e25_student_output/v2/models/e25-student-bge-small-en-v1.5-int8/model.onnx": "a84bb36065ba3b715029f954033c3f3dfbc253e8c6f0b5f64fb8d86300980bbe",
    "private_notes/annotation_uploads/task1/e25_student_output/v3/models/e25-student-bge-small-en-v1.5-int8/config.json": "596a8b3a87448227d15c622c859c55e4b9d35f94aa2da136ab3d119c8ac4d1da",
    "private_notes/annotation_uploads/task1/e25_student_output/v3/models/e25-student-bge-small-en-v1.5-int8/model.onnx": "80b70feb6bc2b436a8e500fb57d9d82642f293fe2e94aeeb99c5d60febb98dd6",
    "private_notes/annotation_uploads/task1/e25_student_output/v3/models/e25-student-bge-small-en-v1.5-int8/source_metrics.json": "d8fbd66da49394d81a339dc69426cd960457279b7ef6332193c833f9d2cc89e3",
    "private_notes/annotation_uploads/task1/e25_student_output/v3/models/e25-student-bge-small-en-v1.5-int8/tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "private_notes/annotation_uploads/task1/e25_student_output/v3/models/e25-student-bge-small-en-v1.5-int8/tokenizer_config.json": "8ccd7632bce0dc3e48daaef7c854244d4675d7d381d43e56406f3d6c47c3145e",
    "private_notes/annotation_uploads/task1/event_test_v2/labels_final_record.json": "249941535a36effd1a1e28866c69b81f21eae11c750b9972574fac59e64a3d46",
    "private_notes/annotation_uploads/task1/event_test_v2/sampling.json": "9999354b34e7453a722c56494eef631ec65531cc1d2470d4b954b0a616cb6b1a",
    "src/riskengine/eval/e24_dev.py": "5f7c6cc096b2a76d1c71dc17aa3dab63ec31a41ddf64d578849c28b452abe036",
    "src/riskengine/eval/e24_gate.py": "4691ba67b52d5ba3e622e9756b55a686171d90f1949d6965466c7ac6353cd7c9",
    "src/riskengine/eval/e24_inference.py": "614f164e76e672e38cb5050d8da56eeb06fa32da998f16d20d337976e3ef2c20",
    "src/riskengine/eval/e24_rules.py": "b7317befcf1fe639c58c91050c5c317ce78626db05a4341dcdad3f804a0a68b4",
    "src/riskengine/eval/e25_candidate.py": "3bf4ff3ac48948be1553ed4997ff6612a3b0f43f50579a4ef9ce8da0d56f5ba7",
    "src/riskengine/nlp/classifier.py": "b433fefb74e2f75ca496bbc34a3be99e873b6dc4fa2a6b3643341e25ef809f1b",
    "src/riskengine/nlp/events.py": "e2f9db573a47ceb2935d3ad9eeab382b6ee0f0474152343eece97974e642e2de",
    "src/riskengine/paths.py": "bb8636a72d1cd67eb1103ef83597064836e5fdcfb219a2f67f054029ad9672a2"
  }
}
```

