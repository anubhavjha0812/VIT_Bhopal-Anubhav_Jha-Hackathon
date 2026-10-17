# Events development baseline — AI reference v6

Completed 2026-10-16T03:30:42.698454+00:00. The unchanged local event classifier scored all 300 development headlines.
AI-derived reference, resolved by Codex after Claude review; no human or expert validation is claimed.

| Set | Rows | Correct | Accuracy | Fixed 12-class macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| D1 | 153 | 71 | 46.4% | 0.3297 |
| D2 | 147 | 67 | 45.6% | 0.2533 |
| Combined development diagnostic | 300 | 138 | 46.0% | 0.3303 |

The primary metric uses all 12 predefined classes with zero_division=0. No classes or rows were dropped. There are no positive Default, Credit downgrade, or Covenant breach examples in the development labels. Their recall cannot be estimated here; their F1 contributions are zero by the fixed metric definition. E23 is the separate 243-row credit reference and was not scored in this run.

Weighted F1 is 0.5470. Always predicting Other gives 64.7% accuracy but only 0.0655 macro-F1, illustrating class imbalance.

The 23 ambiguous development rows are included in the primary score. On the 277 unambiguous rows, macro-F1 is 0.3406 and accuracy is 47.7%; ambiguity alone does not account for the errors.

**Class results**

| Class | Reference rows | Predicted rows | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 0 | 0 | 0.000 | N/A (no positives) | 0.000 |
| Credit downgrade | 0 | 0 | 0.000 | N/A (no positives) | 0.000 |
| Covenant breach | 0 | 0 | 0.000 | N/A (no positives) | 0.000 |
| Geopolitical | 1 | 1 | 0.000 | 0.000 | 0.000 |
| Macroeconomic | 9 | 31 | 0.258 | 0.889 | 0.400 |
| M&A | 4 | 53 | 0.019 | 0.250 | 0.035 |
| Product news | 6 | 60 | 0.100 | 1.000 | 0.182 |
| Earnings | 8 | 11 | 0.273 | 0.375 | 0.316 |
| Analyst rating | 73 | 36 | 0.944 | 0.466 | 0.624 |
| Legal/Regulatory | 4 | 3 | 1.000 | 0.750 | 0.857 |
| Management change | 1 | 1 | 1.000 | 1.000 | 1.000 |
| Other | 194 | 104 | 0.788 | 0.423 | 0.550 |

Management change has one reference row; Legal/Regulatory and M&A have four each, and Geopolitical has one. Their scores do not support reliable per-class quality claims.

**Largest confusions**

| Reference label | Prediction | Rows |
| --- | --- | ---: |
| Other | Product news | 48 |
| Other | M&A | 35 |
| Analyst rating | Other | 19 |
| Other | Macroeconomic | 19 |
| Analyst rating | M&A | 17 |
| Other | Earnings | 8 |
| Analyst rating | Product news | 3 |
| Earnings | Other | 3 |

The model predicted Product news 60 times for six reference product events; 48 Other rows became Product news. It predicted M&A 53 times for four reference M&A events. D1 errors include fund-holding lists treated as acquisitions and explicit analyst downgrades treated as M&A. Analyst-rating recall is 34/73 (46.6%).

These are descriptive development results. D1 row-level errors are saved for rule design. D2 remains for later candidate/threshold choice; its row-level errors were not inspected for rule design in this run.

**Reference decisions fixed before inference**

- Accepted Crew Dragon mission -> Other, ambiguous Y. NASA describes a demonstration flight test rather than a product unveiling: [NASA source](https://www.nasa.gov/image-article/spacex-rocket-prepared-demo-2-mission/).
- Accepted coronavirus-bill EV tax-credit extension -> Macroeconomic, ambiguous Y, applying the guide definition of fiscal stimulus.
- Retained Apple disclosure-rights ruling -> target match Y / non-credit other, with ambiguity. The ruling protects Apple information disclosure, with supplier bankruptcy as its setting: [Reuters report](https://ca.finance.yahoo.com/news/apple-given-time-block-gt-bankruptcy-case-ruling-001316972--sector.html).
- Accepted all 13 analyst-roundup flags. V6 has 11 ambiguous credit and 23 ambiguous development rows, 34 across all 543 annotations. Every row has a label.

**Efficiency and verification**

CPU INT8 ONNX, batch 64, max 64 tokens, one 64-row warm-up. Inference: 5.839 s, 51.4 texts/s. Model 33.94 MB; whole-process peak RSS 688.5 MiB. Load time 16.18 s is separate from inference throughput.

96 existing E24 tests passed; runner ruff check passed. Saved scores were independently recomputed from predictions. Reference/input/model/code hashes, the frozen E24 section and sample backups passed checks before and after inference.

No weights, rules, thresholds, production settings, original annotations or frozen files were changed. No candidate was selected or frozen. Final holdout labels were not read or created; no holdout inference ran.

**Artifacts and next step**

- Current labels: private_notes/annotation_uploads/task1/reference_v6/.
- Scores: data/outputs/e24_ai_dev_baseline_v6.json.
- Predictions: data/outputs/e24_ai_dev_baseline_v6_predictions.csv.
- Per-class/confusion tables and D1 errors: matching _per_class.csv, _confusion.csv and _d1_errors.csv.
- Runner: private_notes/annotation_uploads/task1/events_dev_v6/run_events_dev.py.

Next: design the planned product/earnings rules on D1, compare the fixed candidate/threshold grid on D2, then freeze the selected candidate before annotating the separate 200-news / 100-tweet holdouts. AI reference quality and low event support remain limitations of this development diagnostic.

**Documentation hash changed after the completed run**

A concurrent formatting-only edit to the E24 paragraph in docs/DECISIONS.md changed its locked hash from bfffa0ad37366c8a to 7c769d677b62e159. The model run passed the original frozen checks before and after inference; predictions and labels are unchanged. The live documentation gate now fails, so candidate freeze/reproduction must wait for the original whitespace to be restored. The edited live file was preserved. The exact original document and formatting diff are saved in private_notes/annotation_uploads/task1/events_dev_v6/. Frozen sample/backups were verified using the archived original document.

**Formatting issue resolved**

The original locked E24 whitespace has been restored after confirming that all wording was unchanged. The cause of the earlier whitespace edit is unknown; no attribution to the user or any tool is established. The live E24 hash is again bfffa0ad37366c8a. Frozen sample/backup checks, the v6 reference manifest and input hashes, and model/code hashes all pass. The baseline scores and predictions are unchanged; no model inference was rerun. The pre-repair document and section are preserved in private_notes/annotation_uploads/task1/events_dev_v6/formatting_repair_20261016T033939284174Z.
