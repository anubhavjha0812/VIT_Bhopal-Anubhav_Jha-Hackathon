# E38c: third candidate of the extended company linker, tested on a new sample

Rules written on 19 Oct 2026, before any code change for this candidate and before any scored run on the new sample. They are not changed after the run.

## Why

The second candidate (E38b, `docs/experiments/LINKER_EXTENDED_V2.md`) raised recall (39.5% to 48.0%) and precision (87.5% to 94.0%) over the 15-company linker on a new 200-headline sample, but it missed the adoption rule: precision 94.0% against 95%, and 16 valid links of the original 15 companies were removed (31 unique removals judged: 15 correct, 16 wrong). Its wrong removals fall in a few groups, seen on the credit sample (design data):

- an analyst-desk or context rule fired on a company that is the actor, not the rater: "<Company> Seeks ...", "<Bank> Says ... (its own legal defence)", "<Bank> cuts mortgage chief role";
- a risky-name guard removed the company although the headline is about its own action (a patent case, a podcast purchase, a bankruptcy ruling in which it is a party);
- a list of several companies with only the first kept, and a branded service (Amazon Web Services) not linked to its parent.

Correct removals to keep: "Apple supplier ... bankruptcy" (the supplier is the subject), a differently named company (Apple Rush Co.), a bank that only supplies market research.

## Candidate changes (linker only; no model, gate or stored output touched)

1. The analyst-desk rule removes a bank or broker only when the headline carries a rating or price-target form (a rating word or a PT/PO/price-target phrase, an upgrade/downgrade/initiate/maintain verb with a target company). A bank that is the actor of an ordinary verb (says, cuts, seeks, sees, rejects) stays linked unless the sentence is a research quote about other named companies.
2. A risky-name guard for the original 15 companies fires only when a look-alike context is present (the fixed context list of E38). Without one, the company link is kept.
3. A branded service named with its parent ("Amazon Web Services") links to the parent; every named company in a list is linked.
4. "<Company> supplier/customer/partner ... <other firm> bankruptcy/default" keeps the other firm as the subject and drops the company, as in E38b (these removals were judged correct).
5. Everything else as in E38b. Candidates 1 and 2 stay reproducible. The live default stays off unless the decision rule is met.

## Test set (fixed before the run)

- `private_notes/annotation_uploads/task1/linker_extended_v3/`: 200 new headlines (seed 40), disjoint from the first two samples, 100 inside and 100 outside the catalog by hidden source symbol, drawn before these rules, with no linker output used.
- Reference: two independent AI labellers from text alone, differences settled from the written guide, then frozen as `labels_final.csv` with its SHA-256 before any run. Web is allowed only to identify a company; every use is logged.
- Metrics, Set B (the 243 original credit-sample pairs, every difference on original tickers listed and judged from the text alone before the run) and the 60 probes as in E38b.

## Decision rule (adopt for `/analyze` and the live feed only if ALL hold)

1. Precision of the candidate on the new sample is at least 0.95.
2. Recall of the candidate is higher than the recall of the 15-company linker.
3. Set B: no wrong removal among the judged differences on the original tickers.
4. Probes: at least 58 of 60 pass.

Otherwise: not adopted, the flag stays off, and the result is kept.

## Honest limits

AI labels from two labellers, no independent human check. The candidate rules were designed after seeing the errors of the first two samples; the third sample was drawn before them. One sample of 200, partly company-conditioned, old US large-cap news. Coverage only: no sentiment or event accuracy claim.

## Result

### Current selection

On 19 Oct 2026, after reading the result, the owner selected **candidate 3** and removed the 95% precision requirement as a blocker to that choice. `config/default.toml` now has `extended_linker=true`; the shared API/live factory explicitly loads `candidate=3`. The sections above retain the original experiment protocol as history. Its adoption rule no longer controls the current selection.

The algorithm, aliases, reference labels, saved scores, run lock and design manifests are unchanged. There was no second scored run. The original 15-company batch/replay linker is unchanged; additional recognised companies get no invented portfolio exposure. Existing live cache rows keep their old scores; new analysis uses candidate 3. Restart an already-running API process to load the new factory and clear its cached linker/feed.

Three added tests in `tests/test_linker_adoption.py` check the real loaded default and candidate version, then API/live parity on invented Apple and Costco headlines with fake scorers and a local temporary feed. The selection changes only the API factory and application config; the original evaluator and algorithm stay frozen. Their historical manifest is not rewritten: its API/config hashes now describe the pre-selection files, while the other 18 sealed files are unchanged.

Selection validation: **1067 passed, 1 expected failure** in the full offline suite (187.08 seconds), with the existing Starlette/httpx warning. All 139 focused config/linker/entity-path checks, Ruff `src tests`, touched-test formatting and changed-line whitespace checks pass. Of 405 baseline file hashes, 397 are unchanged; the eight deliberate source/config/test/report/log edits and one new test file are listed in step 310. All old outputs, labels, design manifests, fixed experiment rules and 99 model-file sizes/mtimes remain unchanged. No scored run, network, model inference, commit or push was made for selection.

### Frozen scored result

Run once on 19 Oct 2026 at 11:59:19 UTC, fully offline. The original automatic adoption rule failed because precision was below 0.95; the flag was still off at that run. Candidate 3 improved precision, recall, F1 and exact accuracy, while wrong-link count rose from 8 to 10. The later owner selection above changes deployment, not this saved experimental outcome.

### New 200-headline sample (Set A)

The settled reference hash is `b3474ff739b2efe731440bf0116e4884224f124763f61018433055e207a4db46`, matching the prompt and freeze record. The two AI labellers agreed on 183 rows; 17 differences were settled from the guide before scoring. There was no independent human check. All 200 IDs/order and normalised-headline uniqueness/disjointness checks passed. The reference contains 308 links, including 153 named entities outside the 99-company catalog.

| Measure | Original 15-company linker | Candidate 3 |
|---|---:|---:|
| Precision per headline | 0.925234 | 0.939276 |
| Recall per headline | 0.377422 | 0.512946 |
| F1 per headline | 0.428856 | 0.560331 |
| Exact entity-set accuracy | 0.335 (67/200) | 0.420 (84/200) |
| Wrong links | 8 | 10 |
| Correct links | 107 | 147 |
| Missed links | 201 | 161 |

These use the unchanged E38 metric: precision averages over headlines with a prediction, recall over headlines with a reference, and F1 over headlines with either. Exact accuracy includes all 200. Micro precision/recall/F1 are 0.930435/0.347403/0.505910 for the original linker and 0.936306/0.477273/0.632258 for candidate 3. The larger candidate finds more correct links and also makes two more wrong links; this is a coverage result, not a sentiment or event-accuracy result.

### Probes and fixed decision

Candidate 3 passes **59/60** separate invented probes. Analyst desks pass 5/6; each of the other nine categories passes 6/6. The failed probe is `analyst_desks-4`, “Morgan Stanley raises Oracle price target”: expected ORCL, predicted no subject. The probe and implementation are left unchanged.

| Fixed check | Result |
|---|---|
| Precision at least 0.95 | Fail: 0.939276 |
| Recall higher than original linker | Pass: 0.512946 vs 0.377422 |
| No wrong original-company removal on Set B | Pass: zero |
| At least 58/60 probes | Pass: 59/60 |

Three checks passed; all four were required by the original rule, so its recorded outcome was **do not adopt**. The owner later removed that rule as a selection blocker and enabled candidate 3, as recorded above. There was one successful `--run`; `data/outputs/linker_extended_v3.json` and its persistent `.run_lock` are retained. No second scoring run was made.

### Seal check before the run

An earlier hash-only check stopped before scoring because E41 had changed `config/default.toml` and `src/riskengine/api.py`. The supplied manifest now records those two changes in a dated amendment and preserves the original as `candidate_manifest_original_before_e41.json`. Neither file is imported by the linker evaluator. Before this run, all 20 amended-seal file hashes, the fixed rules prefix, Set B judgement hash and frozen-label hash matched. No result or run lock existed. The evaluator repeated the seal/reference checks before scoring. This pass did not change either manifest or any sealed file.

### Diagnosis and implementation prepared before test contact

The 16 wrong E38b removals reduce to two code paths: 13 require a positive company-context word within 80 characters and lack one; three bank links are stripped by ordinary research-action verbs. Full original text and rule per row are preserved in `private_notes/annotation_uploads/task1/linker_extended_v3_design/diagnosis.csv` and its hash record. The table abbreviates the headlines only for display.

| Ticker | Design headline (shortened) | Candidate-2 rule responsible |
|---|---|---|
| BAC | Bank of America Says AIG… Reject AIG's Assertions | Generic `Says` classified as research attribution |
| AAPL | Apple Seeks to Move Kodak Patent Case | Missing positive context within 80 characters |
| AAPL | Apple Is Seeking an Executive… Buying Original Podcasts | Missing positive context within 80 characters |
| BAC | BofA cuts mortgage chief role | Generic `cuts` classified as research attribution |
| AAPL | Apple given time to block info in GT bankruptcy case | Missing positive context within 80 characters |
| AMZN | Amazon Web Services Goes Hybrid | Missing positive context within 80 characters |
| AAPL | Apple, Amazon, Google, Facebook outlook | Missing positive context within 80 characters |
| AMZN | Apple, Amazon, Google, Facebook outlook | Missing positive context within 80 characters |
| AAPL | Kodak, Apple patent fight to remain in bankruptcy court | Missing positive context within 80 characters |
| AMZN | Amazon Prime membership without consent… lawsuits | Missing positive context within 80 characters |
| AAPL | Apple Pulls NASDAQ, S&P into the Green | Missing positive context within 80 characters |
| AAPL | Google Says Apple, Microsoft, Oracle Wage Hostile Campaign | Missing positive context within 80 characters |
| AAPL | Apple Fights Kodak… Over Patents | Missing positive context within 80 characters |
| AAPL | Apple Watch Sport will be most popular | Missing positive context within 80 characters |
| AAPL | Apple Asks for Court Approval to Sue Bankrupt Kodak | Missing positive context within 80 characters |
| BAC | BofA Said to Keep Countrywide Bankruptcy as Nuclear Option | Generic `Said` classified as research attribution |

Candidate 3 adds a separate version branch. It gates research attribution on rating/price-target forms or identifiable research commentary, retains ordinary bank actors, uses look-alike guards for original risky names, recognises all catalog names in lists and the Amazon-branded service, and suppresses a company used only to describe a supplier/customer/partner in bankruptcy/default. Differently named legal entities remain separate. The catalog, aliases, original loader, API/config, batch, gates and old outputs stay unchanged. The old invalid-version test now checks 4 instead of 3; candidate 3 is valid. Tests reproduce every saved candidate-1 and candidate-2 prediction/metric.

### Set B completed before new-sample scoring

On the same 243 credit pairs, candidate 3 restores all **16 previously wrong removals**. This is seen design data, not a new performance estimate.

| Original-15 projection | Changed pairs | Removed links | AI-correct removals | AI-wrong removals |
|---|---:|---:|---:|---:|
| Mentions | 6 | 6 | 6 | 0 |
| Subjects | 10 | 10 | 10 | 0 |

There are no added original-company links. Six mention changes are supplier-bankruptcy descriptors or Apple Rush Co.; four further subject changes are bank research/rating attribution. The 16 entries across the two views cover ten unique headline/company removals. Every removal was judged from its text; no human check. `setb3_judgements.csv` and `setb3_judgements_record.json` in the design folder contain all IDs, full texts, added/removed tickers, yes/no judgements and reasons. CSV SHA-256: `c0515ac37ed9582ec1817b58ea8b70671128ae27a3eaf43ca76a726c86103cfa`.

`candidate_before_setb3.json` records source/test hashes before that audit; `candidate_manifest.json` seals the fixed rules, implementation, evaluator, tests, credit/probe sources and design records. The completed run reproduced the pre-run Set B audit on all 243 pairs: six mention differences and ten subject differences, 16 judgement entries covering ten unique removals. All ten were AI-judged correct; zero were wrong, and no original-company links were added. Outside-catalog additions occurred on 26 pairs and were recorded separately. The 16 previously wrong E38b removals had already been restored on this seen design data; that is not a fresh performance estimate.

Preparation files were `src/riskengine/ingest/linking_extended.py`, `src/riskengine/eval/linker_extended_v3.py`, `tests/test_linker_extended_v3.py` and the invalid-version assertion in `tests/test_linker_extended_v2.py`; design records live in `linker_extended_v3_design/`. Those files were not edited for this run. Prompt 26 wrote only its `labels_chatgpt.csv` and `labels_chatgpt_record.json`; CSV SHA-256 `e18aed746fc1c7f16a8822ff22f6288cae3de1b54bf95bba4fcc6e7b8dd1c685`.

This run adds only `data/outputs/linker_extended_v3.json`, `data/outputs/linker_extended_v3.run_lock`, this Result section, the E38c row in `docs/experiments/EXPERIMENTS.md` and step 309 in `STEPS.md`. The result SHA-256 is `693678d5df14df723e1a351a819f475ad2ab15f4cd527035978aab149564febf`.

Validation: the required full offline suite passes, **1064 passed, 1 expected failure**, in 202.44 seconds, with the existing Starlette/httpx warning. Ruff `src tests` and whitespace checks pass. Every metric, Set B count, probe score and decision was independently checked from the stored rows without rerunning a linker. All 20 sealed hashes, final-reference hashes, the fixed rule prefix and all 99 model-file sizes/mtimes remain unchanged. Of 403 baseline file hashes, 399 are unchanged; this pass changed only the three report/log files, while a concurrent `docs/RESULTS.md` edit was preserved without changes by this pass. Old output files are unchanged; the only new outputs are the result JSON and run lock. Snapshot/arithmetic checks are under `/tmp/e38c30_before.json`, `/tmp/e38c30_verification.json` and `/tmp/e38c30_verify.py`.

Offline flags, no dependency sync and a temporary UV cache were used. Full tests ran outside the sandbox because its TestClient stall was already established. No repository-wide formatter ran. No hidden provenance was opened. No network, learned-model/credit-gate evaluation, earlier-result edit, commit or push.
