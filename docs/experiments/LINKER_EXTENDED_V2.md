# E38b: second candidate of the extended company linker, tested on a new sample

Rules written on 19 Oct 2026, before any code change for this candidate and before any scored run on the new sample. They are not changed after the run.

## Why

The first candidate (E38, `docs/experiments/LINKER_EXTENDED.md`) gave a higher recall than the 15-company linker but its precision was 0.76 against the 0.95 rule, and it changed 20 of the 243 original-company links in the credit sample. Its wrong links on the first sample were mostly mechanical: analyst banks linked as subjects (about two thirds), "Price Target" linked as Target (4), and a few single cases. This second candidate fixes these categories. The first sample is now design data (the fixes were chosen after seeing its errors). A new sample decides.

## Candidate changes (linker only; no model, no gate, no stored output touched)

1. **Analyst desk.** A bank or broker named as the one giving a rating, price target, estimate or opinion is not a subject of the headline. This covers the headline forms "<firm> Maintains/Reiterates/Reinstates/Initiates/Upgrades/Downgrades/Raises/Lowers/Cuts/Comments on/Reviews/Sees/Says ... on <X>" and "<X> upgraded/downgraded/cut ... at/by <firm>", and the firm names already listed as research desks plus the large brokers (for example Barclays, BMO, Mizuho, Oppenheimer, RBC, UBS, Credit Suisse, Deutsche Bank). The firm remains a subject when the headline is about its own action or performance (earnings, probe, customers, lease, hack, rate cut) and has no rating verb.
2. **Price-target words.** "Price Target", "target price", "PT", "PO", "price objective" and "target of $..." are never evidence for the company Target.
3. **Original 15 links.** The extended linker keeps every link that the 15-company linker makes, except where a documented context guard for a risky name removes it (for example "apple harvest"). Each such removal on the 243 credit-sample pairs is listed and judged before the run (see decision rule 3).
4. Nothing else changes: aliases, catalog, risky-name guards and analyst-desk lists stay as in E38 except where 1 and 2 above require.

## Test set (fixed before the run)

- `private_notes/annotation_uploads/task1/linker_extended_v2/`: 200 new headlines (seed 38), disjoint from the first sample, 100 inside and 100 outside the catalog by hidden source symbol, written before the candidate rules.
- Reference: `labels_final.csv`, SHA-256 `f7362bbb…ab0`. Two AI labellers worked independently from text alone and agreed on 193 of 200 rows; the 7 differences were settled from the written guide (web only for company identity of two rows; URLs in `disagreements_settled.csv`). No human check. No linker output was seen in labelling.
- Metrics as in E38: per-headline precision, recall, F1 and exact subject-set accuracy on subject links (links minus analyst-only), for the 15-company linker and the new candidate; wrong-link count; out-of-catalog reference share.
- Set B: the 243 original credit-sample pairs, difference between the 15-company and the new linker on the original tickers.
- Probes: the same 60 invented headlines.

## Decision rule (adopt for `/analyze` and the live feed only if ALL hold)

1. Precision of the candidate on the new sample is at least 0.95.
2. Recall of the candidate on the new sample is higher than the recall of the 15-company linker.
3. Set B: every difference on the original tickers is listed with its text and judged by the AI labeller from the text alone, before the run, as "link was not a subject" (correct removal) or "link was a subject" (wrong removal). No wrong removal is allowed.
4. Probes: at least 58 of 60 pass.

Otherwise: not adopted, the live default stays off, and the result is kept.

## Honest limits

- AI labels from two labellers; no independent human check.
- The candidate rules were designed after seeing the errors of the first sample; the new sample was drawn before the rules and no linker output was used to draw it.
- One sample of 200, sampled by hidden source symbol, partly company-conditioned; US large caps and old news; aliases are a design choice.
- No sentiment or event accuracy claim: this is coverage only.

## Result

One offline scored run on 19 October 2026, after all Set B judgements and input/code hashes were sealed. The rules above are unchanged. The new sample is disjoint from E38 after headline normalisation. `labels_final.csv` retains SHA-256 `f7362bbb07d60c2e4db931276f9379b179878c580361e77945ea309786945ab0`.

| New sample: 200 real headlines | Original 15 | Candidate 2 |
|---|---:|---:|
| Per-headline precision | 0.875000 | 0.939732 |
| Per-headline recall | 0.394706 | 0.480343 |
| Per-headline F1 | 0.442829 | 0.520545 |
| Exact subject-set accuracy | 0.365 | 0.410 |
| Wrong headline-company links | 14 | 9 |
| Correct headline-company links | 112 | 140 |
| Pooled link precision | 0.888889 | 0.939597 |
| Pooled link recall | 0.375839 | 0.469799 |
| Pooled link F1 | 0.528302 | 0.626398 |

Precision uses 112 nonempty prediction sets for each linker; recall uses 170 nonempty references. F1 uses 171/170 nonempty unions, with 29/30 empty/empty rows excluded. The 298 reference links include 132 outside the catalog. These numbers compare both linkers on the same new sample. Different samples and label references prevent reading E38 versus E38b scores as a paired improvement estimate.

Set B was reviewed **before any candidate-2 output on this new sample**. Both original-company mention links (the E38 check) and desk-filtered subject links were audited so bank-role changes are visible:

| Original-15 projection on 243 credit pairs | Changed pairs | Removed links | AI: correct removals | AI: wrong removals |
|---|---:|---:|---:|---:|
| Mentions | 20 | 21 | 8 | 13 |
| Subjects | 30 | 31 | 15 | 16 |

No original-company link was added. The same removal can appear in both views; 52 CSV judgement entries cover 31 unique headline/company removals across 30 pairs. On those unique removals, 15 were judged correct and **16 wrong**. These are text-based AI judgements, with no independent human check. The candidate's existing context guards still miss real subjects such as Apple in its own patent litigation. The broader research rule also removes some bank actions, such as BofA cutting its mortgage chief role. These failures were retained without changing the candidate after the audit or scored run.

The complete texts, added/removed tickers, views, yes/no judgements and reasons are in `private_notes/annotation_uploads/task1/linker_extended_v2/setb_judgements.csv`. Its pre-run record is `setb_judgements_record.json`; CSV SHA-256 `fda0361a7db159065181187ab8b54ff851c70cf7f8ccab35a501cc5c6e76deba`. `setb_candidate_before_review.json` records the candidate hash before reviewing these differences; `run_manifest.json` seals candidate/evaluator/tests/reference/credit/probe inputs before scoring. Hidden `provenance.csv` was not opened.

Separate invented probes: **59/60 pass**. Target-price, visa applications, ford/river, gap/coverage, apple/harvest, cashtags, possessives, two-company and other ambiguous/legal categories each pass 6/6; analyst desks pass 5/6. The unchanged “Morgan Stanley raises Oracle price target” probe still misses Oracle because of the unchanged context guard.

**Decision: not adopted.** Precision improves but is below 0.95; recall is higher; the no-wrong-removal condition fails; 59 probes exceed the minimum of 58. `extended_linker=false` stays the live default. Candidate 2 is selected explicitly with `ExtendedLinker.from_csv(candidate=2)`; the default loader remains candidate 1. No alias/catalog/risky-name-guard change or post-run tuning was made. Batch and historical gates remain the original 15-company paths.

Changed/new implementation files: `src/riskengine/ingest/linking_extended.py`, `src/riskengine/eval/linker_extended_v2.py`, `tests/test_linker_extended_v2.py`; this Result section, the E38b register row and step log. Candidate 1 reproduces every saved E38 mention/subject prediction and metric. The new evaluator reuses E38 `metrics`, `subject_links`, `norm`, hashing and exclusive JSON writing; it refuses a second scored run and verifies settled-label and sealed-file hashes before inference. Failed attempts can retry only if no result JSON was written.

Full per-row results, Set B comparison/judgements, probe rows, metric denominators, fixed-rule text, hashes and decision checks: `data/outputs/linker_extended_v2.json`; SHA-256 `167632ad4237a708cb39a7c616840edf17e8990f3261e4f62070c5ea328520ee`. The exclusive `linker_extended_v2.run_lock` is retained. Metrics and counts were independently verified from saved rows without a new linker run. Pre-run rules-prefix SHA-256 remains `7ed454ac7f380119a5dcdf8c4806927edcb73cd075767e95c98ac120a438c88d`.

Validation: focused linker tests 90 passed (59 new cases); full offline suite **972 passed, 1 expected failure**, in 108.54 seconds, with the existing Starlette/httpx warning. Ruff `src tests` and whitespace checks pass. All 213 pre-existing protected file hashes and all 99 model-file sizes/mtimes match, including original/extended catalogs, labels, prior outputs, API/config, gates, original batch and frozen files. Fixed rules-prefix and all sealed code/input/judgement hashes still match. Used offline flags, no dependency sync and a temporary UV cache; tests ran outside the sandbox because its TestClient stall was already established. No network, learned-model/credit-gate evaluation, earlier-result edits, commit or push. No sentiment or event accuracy claim.
