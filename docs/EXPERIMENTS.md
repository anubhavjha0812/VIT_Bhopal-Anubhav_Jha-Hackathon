# Experiment register

Every experiment, its outcome and where its stored results are. 
The rules for each were fixed before scoring ([DECISIONS.md](DECISIONS.md)); 
numbers here are copied from the stored files,
which are the source of truth. 
Updated after every experiment.

| ID | Question | Outcome (headline numbers) | Verdict | Stored results |
|---|---|---|---|---|
| E1 | Sentiment on a public tweet test (TFNS, 2,388) | engine int8 acc 0.877, macro-F1 0.836 [0.818, 0.853]; FinBERT baseline 0.726 / 0.669 | beats baseline | `data/outputs/eval_results.json`, `docs/RESULTS.md` |
| E2 | Event class on a public test (TFN-topic, 4,117, 4 classes) | topic int8 acc 0.951, macro-F1 0.886; keywords 0.740 / 0.505 | beats baseline | same |
| E3 | Real-headline development reference for events | 300 AI-resolved v6 labels, D1 153 / D2 147; no human/expert validation | development reference available | private_notes/annotation_uploads/task1/reference_v6/, e24_ai_dev_baseline_v6.md |
| E4 | Efficiency next to every accuracy | engine sentiment 34 MB, 56 texts/s on CPU (FinBERT 438 MB, 13/s) | reported | `eval_results.json` |
| E5 | Fine-tune bge-small and FinBERT on TFNS / TFN-topic | bge-small sentiment 0.885 acc; FinBERT 0.893 | bge-small chosen (size/speed) | `eval_results.json`, `models/kaggle_v1/` |
| E6 | Learned impact score v1 | lost to volatility-only baseline (Spearman 0.17 vs 0.27–0.34) | fail | `impact_results.json`, `docs/IMPACT.md` |
| E7 | Teacher (DeBERTa) + distilled student | teacher 0.905 acc (target 0.91 missed); student 0.885, not chosen on tweet dev | engine unchanged | `eval_results.json`, `models/kaggle_v2/` |
| E8 | Impact score v2 | fresh 2019 Spearman 0.294 vs volatility 0.145; fresh 2020 −0.041 vs 0.307 | win 2019, fail 2020 | `impact_v2_results.json`, `docs/IMPACT_V2.md` |
| E9 | int8 ONNX engine model | −0.8 pt accuracy, 4× smaller, 1.24× faster | int8 is default | `eval_results.json` |
| E10 | One encoder, two heads | sentiment 0.886 acc, topic 0.950 acc | reported, not engine | `eval_results.json`, `models/kaggle_v3/` |
| E11 | Switch the engine to our models | demo re-scored, 89.7% same event class | switched | `signals.jsonl.gz` |
| E12 | Bank named only as analyst ≠ subject | 126 / 126 tagged links correct in the demo | rule kept | README "Credit alerts" |
| E13 | Credit rules + Module B gate | alerts 41 (1 real) → 4 (3 real) on the demo (design check) | rule frozen | README "Credit alerts" |
| E14 | Module B checks: real alerts, 12 crises | sign right 2 of 3 alerts, size 18–75× too big; analogs no better than naive ($45.7m vs $47.0m) | negative / neutral | `module_b_alerts_check.csv`, `module_b_crisis_backtest.csv` |
| E15 | State-aware market move, monthly (345 months) | model/naive MAE 1.00 (pass ≤ 0.85), DM p 0.933 | fail | `module_b_monthly_backtest.csv` |
| E16 | Nine learned event classes | Product news F1 0.892; old 3 classes within 2 pts; int8 macro-F1 0.841 | pass | `eval_results.json` |
| E17 | Module A over 10 years (2010 – Jun 2020) + tweets 2021–22 | test IC 0.011 [−0.012, 0.034]; Sharpe diff −0.071 [−0.265, 0.103]; relevance-rule precision 0.849 [0.76, 0.91] | fail: no proven edge | `module_a_history.json`, `relevance_results.json`, `data/relevance/` |
| E18 | Engine sentiment on 4 more public sets, leak-checked | 0 rows seen; PhraseBank AllAgree 0.865, 50Agree 0.685, FiQA 0.573, NWGI 0.406; distilled model better on news (paired) | reported | `eval_results.json`, `e18_*.json`, `e18_predictions.csv.gz`, `leak_check.json` |
| E20 | Choose the engine sentiment model on tweet + news dev | dev: engine 0.864 / 0.404, distilled 0.829 / 0.523 → engine stays; SEntFiN 7,903: 0.745 vs 0.765, paired +0.020 [0.013, 0.027] | engine stays (trade-off) | `e20_results.json`, `e20_predictions.csv.gz` |
| E21 | One sentiment model for tweets + news (trained on Kaggle, confirmed on fresh tests) | dev int8: engine 0.864 / 0.734, new model 0.814 / 0.858 → engine stays. Fresh: FinEntity 851 (human) engine 0.765 vs 0.700 (paired −0.065, lower); 2021–22 tweets 251 (AI-assisted, human-reviewed) 0.452 vs 0.591 (+0.139, higher); TFNS tie | engine stays (wins on primary FinEntity); E21 model kept as a lead for retail tweets; cross-source generalisation = future research | `e21_results.json`, `e21_predictions.csv.gz`, `e21_training_*.json`, `e21_sample.json` |
| M1 | Module B sensitivity: do the formulas move the right way? | 7 checks (13 tests) on the real book: no shock → no loss; loss grows with the episode multiple (3 episodes); multiple 0 removes the episode; more notches → more PD/ECL/loss/RWA; stressed PD rises with z; default → stage 3, bond at recovery; reverse stress lands on 8% | all pass | `tests/test_module_b_sensitivity.py` |
| E22 | Interest-rate shock in Module B (parallel UST10Y shift) | 6 checks pass (incl. exact E14 reproduction) → default `rates = moved`. COVID loss $294m → $194m, CET1 4.22% → 7.05% (yields fell 83 bp); GFC 4.37% → 8.91%; +200 bp alone: CET1 13% → 6.18%. Forecast re-runs (reported only): still no edge over naive | checks pass; funding side not modelled | `e22_rates.json`, `e22_crisis_backtest.csv`, `e22_monthly_backtest.csv`, `tests/test_module_b_rates.py` |
| E22b | Does a credit alert alone carry information? (alert-only prediction vs the stock's market-adjusted 20-day return) | history 2010 – Apr 2020: 79 alerts, 40 independent; 55% negative vs base 47.8%, binomial p = 0.23; mean −1.7% [−4.1%, +0.6%]; predicted size vs reaction Spearman +0.05 | FAIL; many history alerts are false (bank-named securitisations, affirmations) | `e22b_alerts.csv`, `e22b_results.json`, `tests/test_module_b_alert_check.py` |
| E23 | Is the frozen credit-alert gate right? | AI reference 243 pairs: TP 39, FP 44, FN 7, TN 153; sample precision 0.470, recall 0.848; population-weighted recall point estimate 0.758 | report only; gate unchanged | e24_e23_gate_v6.json/.md, e24_e23_gate_v6_false_alerts.csv |
| E24 | Rules for product/M&A overprediction, earnings and analyst routing | D1-only: reproducible per-row A macro-F1 0.3218 / accuracy 0.4379; B 0.4548 / 0.8693; C 0.5156 / 0.9216; D=0.50 demonstration 0.5172 / 0.9281. Product 4/4 correct; M&A 0 predictions, both positives missed. Production A now matches evaluation exactly | D1 development only; D2/final pending, no selection/freeze; production reuses per-row inference | e24_d1_production_alignment_report.md, e24_d1_reproducibility_report.md, e24_d1_rules_reproducible.json, docs/E24_AMENDMENT.md |


## How results are stored
- Every scored run writes a JSON (or CSV) to `data/outputs/` and, 
  where rows are scored, per-row predictions, so any comparison can later be redone as a paired test.

- What was scored is frozen where it matters: dataset revisions and file hashes, model file
  hashes and library versions (`e18_manifest.json`).

- Third-party test files are not committed (licences); they are fetched at pinned revisions and
  checked against the stored hashes.

## E24 D1 rule fixes and E23 report, 16 Oct 2026

- Added simple general rules (details in [E24_AMENDMENT.md](E24_AMENDMENT.md)):
  - fund holding lists are not M&A
  - stock analyst ratings go to Analyst rating
  - earnings previews with no numbers stay Other
  - Product news: real product launches and deals only, not rocket or production launches
- Only D1 was used. No D2 or final rows opened, no candidate picked, no training.
- Scores saved before and after the edit. Candidate A is unchanged.
- Four Product news and two M&A rows are too few to trust recall. Both M&A rows are still missed.
- E23 credit alerts: 44 false alerts, 7 misses. Main false-alert reasons: securitization 21, counterparty 9, no own credit event 8, wrong company 3.
- Sample recall is not the real recall, so a separate estimate is reported.
- Tests: 143 pass. Lint passes. E24 hash `bfffa0ad37366c8a` unchanged.

## E24 same result every time (reproducibility), 16 Oct 2026

- Problem: the model gave different answers when run in a different batch size. In batch 64, 13 of 153 top answers changed (biggest probability change 0.37). Padding also matters.
- Cause: the int8 model rescales values using the whole batch.
- Fix: evaluation now runs one headline at a time. All 153 rows come out identical for every batch size, order and new session.
- New D1 scores (macro-F1 / correct rows of 153): A 0.322 / 67, B 0.455 / 133, C 0.516 / 141, D 0.517 / 142.
- Before the fix (batch 64): A 0.317, B 0.452, C 0.511, D 0.515.
- Product news: 4 predicted, 4 in reference. M&A: 0 predicted, 2 in reference.
- Caches now store the method, versions and file hashes, and refuse to load if anything differs.
- Production at that point still used batch 64 and gave different answers on 13 of 153 rows.
- Tests: 154 pass. No D2 or final rows opened. No commit.
- Reports: `data/outputs/e24_d1_reproducibility_report.md`.

## Production now matches evaluation, 16 Oct 2026

- `nlp/events.py` now uses the same one-headline-at-a-time method as evaluation. No code copied, no new rules in production, candidate A logic unchanged.
- Result on the same 153 D1 headlines: differing answers 13 -> 0. Biggest probability gap 0.37 -> 0 (identical).
- Faster and lighter: 64 -> 81 headlines/s (+26%), memory 705 -> 517 MiB. Numbers are for this laptop and short headlines.
- Added 4 tests. Total 557 pass, 1 expected failure, 1 skipped on purpose (it opens the 300-row CSV with D2 rows).
- No models, rules, thresholds or labels changed. E24 hash unchanged. The old E23 snapshot of `events.py` now differs (allowed edit); the other four snapshot files match.
- Report: `data/outputs/e24_d1_production_alignment_report.md`. No commit.

### E25: two AI teachers labelled the 153 D1 headlines, 2026-10-16
- Teachers: Qwen3-8B and Mistral-Nemo-12B, run on Kaggle.
- They match the reference (v6 / v7) on 76% to 78% of rows.
- They agree with each other on 82% of rows, and those rows are 85% right.
- Mistral is fast enough (about 2 hours for the whole pool), so we keep it.
- Still weak: Product news (too many calls), Default (all wrong).
- Reference is AI-made, not human. File: `data/outputs/e25_d1_agreement.json`.

### E25: teachers labelled the whole training pool, 2026-10-16
- 40,453 headlines labelled by both teachers (Qwen 40 min, Mistral 108 min).
- They agree on 75% of rows. We keep only those: **30,254 rows**.
- Big classes: Other 16,187, Product news 5,465, Legal 2,755, Earnings 1,503.
- Too few rows (under 200): Macroeconomic 104, Default 36, Covenant breach 3.
- Watch: Product news looks too big (18%). Check a sample before training.
- No human labels used. File: `data/outputs/e25_pool_agreement.json`.

### E25: first student model trained, 2026-10-17
- bge-small trained on 20,397 rows (teacher-agreed news + 2,231 TFN tweets for Macro and Geo). 6.5 minutes on Kaggle. Int8 size 33.9 MB.
- Score on its own validation rows: 93.5% right, macro-F1 0.885. These rows are the easy ones both teachers agreed on, so the real score will be lower.
- Problem found: on TFN tweets it calls far too many rows Macro (precision 0.22) and Geo (0.49). Cause: TFN rows were added only for these two classes, so tweet style became a hint for Macro/Geo.
- Next: add TFN rows for every class, retrain, then check on D1. Files: `data/outputs/e25_student_summary.json`.

### E25: student retrained with tweets and news for every class, 2026-10-17
- 22,635 training rows: 18.2k news (teacher-agreed) + 4.5k TFN tweets (600 per class at most). 7 minutes on Kaggle. Int8 33.9 MB.
- Tweets: the Macro problem is fixed. Macro precision 0.22 -> 0.92, Geo 0.49 -> 0.77. Macro-F1 on TFN tweets 0.18 -> 0.82.
- News validation (easy rows): 92.7% right, macro-F1 0.874. Macro has only 6 validation rows, so it is not reliable.
- Not yet checked on D1. Files: `data/outputs/e25_student_summary_v2.json`.

### E25: candidate E checked on D1 only, 2026-10-17
- E reuses C's rules and E13 credit overrides with the v2 student. Both models use the shared per-row inference method. No tuning or retraining.
- Fixed 12-class macro-F1 / accuracy on 153 D1 rows:
  - v6: C 0.515636 / 92.16%; E 0.475546 / 85.62%.
  - v7: C 0.505520 / 91.50%; E 0.466551 / 85.62%.
- Supported-class F1 (supplementary), C -> E: v6 0.687515 -> 0.634062; v7 0.674026 -> 0.622068.
- Against v7, E fixes 6 of C's errors but breaks 15 correct answers. Against v6: 5 fixes, 15 breaks, 1 changed error. E is worse on this D1 check.
- Per-class precision/recall, confusion matrices and every changed row are saved under `data/outputs/e25_candidate_E_*`. Final formatted code reproduces every saved prediction from the hashed probability caches.
- References are AI-derived; v7 is a repeat review by the same resolver. D1 has no positive credit rows, so credit recall cannot be assessed here.
- Production, rules, weights and DECISIONS.md unchanged. No D2/final scoring or candidate selection/freeze. No commit.

### Fresh unlabelled credit sample, 2026-10-17
- Added `eval/credit_heldout.py`, reusing the E25 frame and exclusion helpers. Seed 27 and keyword quotas were fixed before sampling; no model or gate output was used.
- Built 150 FNSPID headlines from 2010-2016 outside the pool/extra: rating-action cues 66, default/bankruptcy 26, covenant/waiver 8, restructuring/distress 20, hard-negative patterns 30. Scarce default/covenant quotas were filled from other credit cues by the fixed fallback rule. These groups are sampling cues, not labels.
- Hard-negative cues: securitization 10, counterparty 2, stock rating 9, wrong-company pattern 9. This enriched sample cannot estimate population prevalence.
- Exclusion checks passed against the existing fixed corpus (including E24 dev through E3, E23, E17, TFN and final texts) and pool/extra. Reserved texts were used only for mechanical duplicate exclusion. No labels were used to choose rows; no model/gate was run on them.
- Sheet: `private_notes/annotation_uploads/task1/credit_heldout/to_label.csv`, exactly `id,text,date,tickers`, no labels or predictions. Manifest: `data/outputs/credit_heldout_manifest.json`. Dry-run and build sample SHA-256 agree: `dc21612f260fb784b27e96f3478a071a0cb028c6141e88662b03c39ee4e52faa`.
- Six synthetic sampler tests added. Regression suite: 591 passed, 1 expected failure, 1 deliberately deselected D2-reading test. Ruff passes. No production/rule/weight/DECISIONS changes, no D2/final scoring, no commit.

### E25: fresh test set R3, student E vs rules C, 2026-10-17
- Rule written first (`docs/E25_RETEST.md`). R3 = 400 new news headlines (2019-20), labelled blind by Claude (AI-made labels, not human).
- Macro-F1 (12 classes): A 0.31, C 0.44, D 0.44, **E 0.47**. Accuracy: C 69.5%, E 73%.
- E beats C by 0.033, which passes the rule (0.02). But the 95% range of the gain is -0.05 to +0.11, so it could be luck.
- D1 gave the opposite (C 0.52, E 0.48). D1 is small and C's rules were built on it. Evidence is mixed.
- Macro, Geo and Management have only 9, 3 and 3 rows in R3, so they cannot be judged. Labels still need a ChatGPT check.
- Files: `data/outputs/e25_retest_scores.json`.

### E25 Macro/Geo adjudication (2026-10-17)
- 963 real FNSPID rows where the two teachers disagreed and one said Macro/Geo. Claude labelled them blind (text only).
- Result: Geo 134, Macro 68 new real rows (student had 296 Geo, 129 Macro). Many rows were company news (Legal 205, Other 486).
- Claude agrees with Qwen 33%, with Mistral 44%. Labels are AI-derived; ChatGPT check still to do. Not used for training yet.
- Synthetic data stays the last option.

### E25 real-source research, 2026-10-17
- Report: `docs/E25_REAL_SOURCES.md`. Five ranked sources with primary links and research-use terms: Fed Board, GOV.UK foreign-office news, ECB, News Category Dataset, AG News. No bulk download or external training set built.
- Old FNSPID Macro/Geo keywords have zero hits beyond the existing extra in the historically cleaned remainder: the 1,212-row union was below both 2,500-row caps. Additive keywords give only 28 raw candidates (13 Macro, 15 Geo; 20 newly matched), before fresh exclusions. These are upper bounds, not labelled or certified-clean examples. Protected examples were not reopened to recount them.
- Small official API queries found 6,588 historical FCO news/press records by 2010-2016 update timestamps; original publication dates and leak status still need checking before ingestion. No labelling in this sources task, synthetic generation, inference, training or frozen-candidate change.

### E25 second AI review of 963 real headlines, 2026-10-17
- ChatGPT read every text against the E24 guide and saved all independent judgments with reasons before opening Claude's row labels. Aggregate counts were known; teacher votes were never inspected. Same ChatGPT resolver as earlier project passes; AI-derived, not a new independent gold reference.
- 679/963 labels agree (70.51%, kappa 0.550); 284 differ. ChatGPT flags 228 ambiguous rows, including 111 agreements. Thus 568 agreed rows have no ChatGPT flag; 395 rows need resolution (284 differences + 111 flagged agreements).
- In the 568-row agreement subset: Macro 35, Geo 17, Other 323, Legal 137, Earnings 56. Claude's proposed 134 Geo + 68 Macro additions are not confirmed wholesale. The largest difference is Geo -> Other (76), then Geo -> Legal (30); many are company actions or regulatory matters with geographic/political context.
- Files: `private_notes/annotation_uploads/task1/macrogeo_adjudication/chatgpt_review_20261017/` (independent labels, paired comparison, unresolved rows, agreement subset, README, record and SHA-256 manifest). No original labels replaced, no training, no model/gate run, no protected-set inspection or frozen-file edit. Data checks pass; no commit.
- Resolution (2026-10-17): two AI labellers agreed on 679 of 963 rows. After resolving the rest with the primary-event rule, only **Geo 19 and Macro 52** confident new rows remain (not 134 / 68). Most "Geo" rows were company news that mention a country. AI-derived labels, no human check.

### E25 official title sampler: source fetch blocked, 2026-10-17
- Added `eval/e25_official_titles.py`: title/date/URL only, 2010-2016, fixed cues and seed 25, at most 400 rows per cue. Cues are not labels. Drops short/non-English titles, exact/near duplicates and reserved texts with the existing exclusion helpers, plus pool/extra, R3 and the credit hold-out. GOV.UK uses original `first_published_at`, not update dates.
- `--dry-run` writes nothing; `--build` caches raw HTTP responses under ignored `data/cache/e25/official/` and writes the CSV and manifest. Requests respect robots and crawl delays. Manifest keeps terms, attribution, hashes and the note that formal titles differ from news.
- **No real sample built:** Fed skipped after its earlier 403; local DNS failed for ECB and GOV.UK. `data/cache/e25/official_titles.csv` has a header only (0 rows). `data/outputs/e25_official_titles_manifest.json` says `blocked`, with leak check `not_run`. A connected build is still needed; live archive parsing is unverified.
- 15 fake-data tests added; 27 sampler/helper tests pass and ruff passes. No labels, training, protected/frozen edits, commit or push.

### E25 official Geo titles (2026-10-17)
- 400 UK foreign-office titles (2010-2016, open licence) via `e25_official_titles.py`. Fed and ECB gave no usable Macro rows, so Macro stays at 52 confident real rows.
- Claude labelled them blind: Geo 318, Other 82 (AI-derived, ChatGPT check pending). Titles are formal ("Foreign Secretary condemns ..."), so mix with Other official-style titles when training.

### E26 market gate: Geo/Macro triggers for Module B (2026-10-17)
- New `src/riskengine/market_gate.py` + endpoint `/module-b/market-alerts`. Fires on Geopolitical or Macroeconomic items only if sentiment is negative, market impact > 7 (severity 7 x (0.6 + 0.4 |sentiment|), +1 news) and a different item of the same class came in the last 24 h. Credit gate and impact.py are unchanged.
- Replay window (Apr-Jun 2020, already seen): 492 Geo/Macro documents, 16 fired (all Macro, all news). Geo never fired: its items there are mostly neutral or positive.
- Quick read of the 16 by Claude (not blind, not pre-registered): about 11 are market-wide (oil price crash, OPEC/IEA demand cuts, recession warning), about 5 are company or sector items the event model called Macro (PC report, Tesla China sales, smartphone shipments). Roughly 70% right; the credit gate is 47%. Exploratory only.
- To do: a fixed trigger-precision/recall test on a labelled event list (TODO L6 step 3).

### E24 final evaluation, run once (2026-10-17)
- Frozen candidate C (rules + engine model) against the unchanged engine A. Labels are all AI-made (ChatGPT, Claude, AI adjudication); no human check. Amendment: `docs/E25_LABELLER_AMENDMENT.md`.
- **News (200):** macro-F1 (12 classes) A 0.309, C 0.376. Gain +0.067, 95% interval +0.029 to +0.107. Pass.
- **Tweets (100):** A 0.142, C 0.132. Gain -0.010, interval -0.031 to +0.009. Pass (inside the tweet margin). 8 of 12 classes have no tweet, so this number is low for every candidate.
- Result file: `data/outputs/e24_final_results.json`. The production engine is not switched yet.

### Separate SEC 8-K credit sampler: code only (2026-10-17)
- Added `eval/e25_sec8k.py`: `--fetch` caches SEC 8-K search JSON for 2010-01-01 to 2020-06-30, pages results, checks every hit's actual item list and counts drops. Uses `SEC_USER_AGENT` with a contact email, below five requests/second. Full search windows split if they reach the result limit; incomplete caches fail.
- `--dry-run` reads SEC cache only and writes nothing. `--build` matches local FNSPID headlines from filing minus 2 through plus 5 days, inclusive, by issuer name or ticker in the text. Reuses the E25 exclusions and leak helpers, plus R3, credit held-out, pool and extra. The standard exclusion loader can retrieve its usual training corpus on the laptop.
- Fixed seed 28; cap 150 unique headlines per item. A headline matching both items appears once, with separate anchor entries. Future output: `private_notes/annotation_uploads/task1/credit_8k/to_label.csv` (only id, text, date, tickers), separate `anchors.csv`, and `data/outputs/e25_sec8k_manifest.json` with counts, hashes and provenance. Item codes are sampling anchors, not labels; the labeller reads only the headline sheet.
- 36 new fixture tests and 71 sampler/linker tests pass; Ruff passes. No network run, real sample build, labelling, gate/model run, protected-file change, commit or push.

### Engine switched to candidate C (2026-10-17)
- `config/default.toml` events = `model_rules`: the engine now runs the frozen E24 candidate C. Stored signals were re-scored with it: 4,110 of 22,943 events changed (mostly Product news to Other, Other to Earnings), sentiment unchanged, credit alerts still 4, market alerts 15 (was 16).
- Module A uses sentiment only, so its results do not change. Module B credit alerts do not change.

### SEC 8-K credit anchors (2026-10-17)
- 1,821 8-K filings (Item 1.03 bankruptcy, Item 2.04 acceleration) from SEC full-text search, 2010 to mid-2020. The SEC items filter is ignored, so items are re-checked on each hit.
- Matched to local FNSPID headlines, only 13 headlines remain (most bankrupt firms are not in our news). Too small for a credit test; a bigger news source is needed (TODO L4 stays open). Not labelled.

### R3: student v3 compared with v2 and C (2026-10-17)

Ran once, offline, on the same 400 R3 headlines and Claude's AI-derived labels. Added `--student <folder>` and `--output-suffix _v3` to `eval/e25_retest_score.py`; the default remains v2. Used the local `e25_student_output/v3/models/e25-student-bge-small-en-v1.5-int8` folder, unchanged per-row inference and rules, D threshold 0.50 and no temperature search.

| Candidate | Macro-F1, fixed 12 classes | Accuracy |
|---|---:|---:|
| C | 0.4363 | 69.50% |
| E with v2 | 0.4690 | 73.00% |
| E with v3 | 0.4495 | 73.25% |

- V3 vs C: +0.0132 macro-F1 and +3.75 percentage points accuracy. The [pre-written R3 rule](E25_RETEST.md) needs at least +0.02 macro-F1 and no F1 drop over 0.05 for a class with at least 10 rows. The class guard passes, but the gain fails: **C stays**. The paired 95% bootstrap range is -0.0515 to +0.0830.
- V3 vs v2: -0.0195 macro-F1, +0.25 percentage points accuracy; 21 predictions changed. R3 remains a development comparison, not another final test. The running engine and frozen candidate are unchanged.
- New files: `data/outputs/e25_retest_scores_v3.json`, `e25_retest_per_class_v3.csv`, `e25_retest_predictions_v3.csv`. All metrics were checked from saved predictions; A/C/D predictions and scores match v2 exactly. V2 files and protected files retain their hashes. 27 scorer/candidate/inference tests pass; Ruff passes. No network, tuning, E24 rerun, commit or push.
