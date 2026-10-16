# E24 D1 rule development — 16 Oct 2026

AI-derived v6 reference, D1 only (153 rows). No real D2 or final rows were opened.
Candidate A and production credit/events code are unchanged. No training, calibration, threshold search, selection or freeze.

| Candidate | Before macro-F1 (12) | Final macro-F1 (12) | Before accuracy | Final accuracy | Final supplementary supported-class macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.3171 | 0.3171 | 43.1% | 43.1% | 0.4229 |
| B | 0.3336 | 0.4517 | 55.6% | 85.6% | 0.6022 |
| C | 0.3302 | 0.5110 | 53.6% | 90.8% | 0.6813 |
| D (fixed 0.50) | 0.2820 | 0.5148 | 46.4% | 92.2% | 0.6864 |

The supported-class metric is supplementary over the nine classes present in D1. The fixed 12-class macro-F1 remains primary. D=0.50 was declared before scoring and is a demonstration, not a chosen threshold. The registered D2 grid is untouched. C is shown below as the variant without abstention, not as a selected candidate.

**Batch context limitation**

This D1-only inference differs from the earlier mixed 300-row inference on 12/153 predictions, with identical text, code and weight hashes. A therefore scores 0.3171 here versus 0.3297 in the earlier mixed-batch diagnostic. This is consistent with batch-sensitive INT8 inference; its precise cause has not been isolated. Every candidate and iteration here shares the same cached probabilities; candidate A is exactly identical before/after on that cache.

**Class precision and recall: C before/final**

| Class | Support | Precision before | Precision final | Recall before | Recall final |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 0 | 0.000 | 0.000 | N/A | N/A |
| Credit downgrade | 0 | 0.000 | 0.000 | N/A | N/A |
| Covenant breach | 0 | 0.000 | 0.000 | N/A | N/A |
| Geopolitical | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| Macroeconomic | 5 | 0.235 | 0.267 | 0.800 | 0.800 |
| M&A | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| Product news | 4 | 0.250 | 1.000 | 0.250 | 1.000 |
| Earnings | 5 | 0.214 | 1.000 | 0.600 | 1.000 |
| Analyst rating | 37 | 0.938 | 0.974 | 0.405 | 1.000 |
| Legal/Regulatory | 3 | 1.000 | 1.000 | 0.667 | 0.667 |
| Management change | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| Other | 95 | 0.767 | 0.989 | 0.589 | 0.905 |

**Product and M&A counts**

| Candidate/phase | Product predictions | Product true positives | M&A predictions | M&A true positives |
| --- | ---: | ---: | ---: | ---: |
| A before | 32 | 4 | 26 | 0 |
| B before | 4 | 1 | 26 | 0 |
| C before | 4 | 1 | 25 | 0 |
| D before | 4 | 1 | 17 | 0 |
| A final | 32 | 4 | 26 | 0 |
| B final | 4 | 4 | 0 | 0 |
| C final | 4 | 4 | 0 | 0 |
| D final | 4 | 4 | 0 | 0 |

Reference counts: four Product news and two M&A. The final C product decisions match all four, but C predicts no M&A and misses both genuine M&A examples. Removing false M&A predictions has not solved M&A recall. With four and two positives, recall and apparent perfect precision cannot support reliable generalisation claims.

**Per-rule fixed/broken effects for C**

| Change | Fixed | Broke | Changed an existing error |
| --- | ---: | ---: | ---: |
| product_rule | 6 | 0 | 0 |
| holdings_rule | 17 | 0 | 0 |
| analyst_rule | 23 | 0 | 0 |
| earnings_preview_rule | 6 | 0 | 0 |
| earnings_rule | 5 | 0 | 0 |

Effects are conditional on the other final rules being active and must not be summed as independent gains.
Every affected D1 row, its original/reference/final labels and fixed/broken outcome is saved in e24_d1_rule_ablations.csv. e24_d1_rule_changes.csv compares the original and final candidates; e24_d1_rules_vs_A.csv attributes final routes relative to A. Full per-class scores for every candidate and both phases are in e24_d1_rules_per_class.csv. All fixed-12 confusion matrices are in e24_d1_rules_confusion.csv and the before/final JSON files.

**Remaining errors and stopping point**

Most remaining C errors concern Macroeconomic/Geopolitical boundaries, bank-lending or production commentary and stock moves. These were left outside the requested rules scope. Other errors include both M&A positives, a continuing bullish analyst opinion, and a bank stress-test regulator story. The remaining C rows are in e24_d1_rules_C_remaining_errors.csv.
Only two bounded D1 iterations were used: the requested rule changes, then general edge-case handling for target abbreviations, stock-recovery headlines and truncated holding lists. The intermediate after report is preserved. No plateau was established, and no training was started. Stop here before D2; later D2 selection must use the registered grid and safeguards.

**Verification and reproducibility**

143 E24 tests passed, including the original 96 plus new rule tests. Repository-wide ruff check passed, and formatting checks passed for all changed Python files. The optional repository-wide format check reports four unchanged files (api.py, ingest/gdelt.py, live.py and tests/test_live.py); those unrelated files were left untouched. Production events.py, the gate and all five E23 snapshot files remain byte-identical. The locked E24 section hash remains bfffa0ad37366c8a; frozen sample/backups pass.
Private runner: private_notes/annotation_uploads/task1/e24_rules_work/e24_d1_report.py. The before rule/dev source snapshots are beside it. The probability cache and metadata are e24_d1_model_probabilities.npz/.json.
The source amendment is docs/E24_AMENDMENT.md. A local Python audit of this work parses only D1 inputs and D1 outputs; frozen D2/final data are read only as opaque bytes by integrity checks.
The unchanged model occupies 33.94 MB. Cached comparisons do not benchmark inference speed; the single D1 inference timing and process memory are recorded in e24_d1_model_probabilities.json.
