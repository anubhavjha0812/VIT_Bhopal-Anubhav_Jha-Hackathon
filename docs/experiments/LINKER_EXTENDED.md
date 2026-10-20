# E38: extended live company linking

Rules written before candidate implementation, blind sampling/labels or scored runs. No network. Prompt 21's tested local changes are the dependency for this task; the user explicitly authorised proceeding with that working tree. No commit or merge is performed.

## Scope and inputs

- Extend only new `/analyze` and LiveFeed requests. Batch, stored signals, replay, gates, Module A and Module B retain the original 15-company linker. No models, model thresholds or existing results change.
- `prices_fresh.csv` contains 86 symbols: **84 companies and two market proxies**, `^GSPC` and `^VIX`. Exclude the proxies. The extended CSV contains these 84 companies plus the original 15: **99 companies**, not a verified current S&P 100 membership list. The local price list is historical.
- Copy every original CSV field verbatim into `universe_extended.csv`; append `in_index=true` to those rows. New companies have `in_index=false`. Names, sectors and aliases are manually recorded public company names, not scraped or derived from the evaluation labels. Historical names/symbols are not claimed to be current listings.
- Keep `Linker.from_csv()` and the original CSV unchanged. A separate extended loader/subclass reuses that loader and the existing desk/analyst logic. It additionally handles GS, MS, C, WFC and USB desks using the same patterns. Own cashtags continue to override analyst-only tagging.
- The original API `linker()` continues to serve stored gates. A separate live loader is selected by the boolean config flag `extended_linker`, initially false. API entities add `in_index`; outside-index entities have no book/portfolio exposure attached. Existing stored live rows are not rewritten.

## Fixed alias rules

Matching remains case-sensitive for names and case-insensitive for cashtag letters, on word boundaries, as in the original linker. New aliases are legal names and common company short names only. Bare stock symbols that are not also company brands are not aliases. GOOG and FB remain aliases for GOOGL and META through the original cashtag table.

For ambiguous short names, a cashtag or a legal company suffix is sufficient. Otherwise a company-context word must occur within **80 characters** on either side of the match. Legal suffixes are Inc/Incorporated, Corp/Corporation, Co/Company, Ltd/Limited and PLC, with optional punctuation. Full multi-word legal aliases are accepted directly.

Fixed context expression (case-insensitive; word boundaries):

```text
shares?|stocks?|earnings|revenues?|profits?|losses?|dividends?|quarter\w*|fiscal|buybacks?|securities|bonds?|debt|credit|analysts?|ratings?|upgrad\w*|downgrad\w*|acquir\w*|acquisition\w*|merger\w*|ceo|cfo|company|corporation|incorporated|payments?|cards?|cloud|software|vehicles?|trucks?|automaker\w*|retail\w*|stores?|banks?|banking|deposits?|machinery|equipment|iphones?|ipads?|macbooks?
```

The following fixed look-alike phrases block a bare ambiguous alias even if the context window also contains a finance word. A cashtag or legal-name/suffix match still wins. No rule is added or changed after the scored run.

| Ticker / short name | Handling |
|---|---|
| AAPL / Apple | Context/suffix/cashtag required; block apple harvest, orchard, fruit, tree, pie, cider and juice phrases. Original 15-linker behaviour remains unchanged. |
| AMZN / Amazon | Context/suffix/cashtag required; block Amazon river, rainforest, jungle and basin phrases. |
| TGT / Target | Context/suffix/cashtag required; block target price, practice, audience, goal and market phrases. A possessive such as Target's earnings passes through its company context. |
| V / Visa | Context/suffix/cashtag required; block visa application, applicant, immigration, tourist, travel, work permit and visa-free phrases. |
| F / Ford | Context/suffix/cashtag required; block ford the/a river, stream or creek, and river crossing phrases. Ford Motor Company is a full name. |
| ORCL / Oracle | Context/suffix/cashtag required; block oracle of Omaha, ancient oracle and mythical oracle phrases. Oracle Corporation is a full name. |
| COF / Capital One | Context/suffix/cashtag required; Capital One Financial Corporation is a full name. No substring match to capital alone or the number one. |
| CAT / Caterpillar | Context/suffix/cashtag required; block caterpillar larvae, larva, insect, butterfly and cocoon phrases. |
| DOW / Dow | Context/suffix/cashtag required; block Dow Jones, Dow industrial/industrial-average phrases, and bare Dow rises/falls/gains/drops/climbs/slides/jumps expressions. Dow Inc. is a full name. |
| Gap / GPS | Absent from the 84-company price list. Do not add it: gap in coverage and `$GPS` are outside this candidate's catalog. |
| CVX / Chevron | Chevron Corporation and Chevron are accepted company names; no bare CVX alias. |

The context guards apply only to the extended candidate, including ambiguous short names among the original 15. They can change a live match on a non-company phrase such as apple harvest. The original loader and all historical paths preserve their original behaviour. The E23 guard below checks the original-company projection before live adoption.

## Blind sample and reference labels

Unit: one unique normalised headline, not a headline-stock pair. Normalisation is case-folding, punctuation-to-space and whitespace collapse. A headline appears in only one stratum. No linker/model predictions enter sampling or labelling.

The specified caches have different schemas: `history_texts.csv.gz` has only text IDs/text; `headline_scores_fresh.csv.gz` has doc IDs/model outputs but no text or symbols. Read **only its doc_id column**. Recover text/date/source symbols from their existing local raw sources, `universe_news_named.csv.gz` and `fresh_news.csv.gz`, using the existing fresh-news cleaning/ID method. Do not read cached scores. History raw rows must match a cached history text. Fresh raw rows must match a cached fresh doc ID. These joins recover sampling provenance; source symbols are not reference labels.

Merge provenance for duplicate normalised headlines. Canonicalise GOOG→GOOGL and FB→META for stratum membership. Positive stratum: at least one source symbol belongs to the full extended catalog. Negative stratum: none belongs. Sort by normalised headline, sample **100 from each** without replacement using seed **22**, then shuffle with the same seed. Refuse an undersized pool rather than changing the rule. Record the stock-symbol/date/cache mapping separately from the blind file. The sample is frozen before labelling.

`--label` writes a 200-row private blind CSV with `id,text,gold_entities,rationale` and a separate sampling manifest/provenance file. It refuses to overwrite an existing sample. Label every row from **text alone**, before invoking either linker on it. Annotation is by the same AI resolver, with no independent human check. Record model name if shown, otherwise `not shown`, time, input/output hashes and annotation kind `ai_blind_link_reference`.

Reference guide:

- `gold_entities` is a JSON array of company tickers whose company/stock is a subject of the headline. Use `name:<normalised company name>` when its symbol is not known from the text/public company identity; do not guess a ticker or inspect source symbols. Empty array means no identifiable company subject.
- Include explicitly named targets of earnings, products, legal/rating actions, deals and stock/holdings changes. Multiple actual subjects may be labelled. An entity merely supplying an analyst opinion, rating, price target, quote or ownership disclosure is attribution, unless its own action/performance is independently discussed. Do not infer a listed parent from a differently named fund or subsidiary.
- Generic sectors, indices, countries and market commentary are not companies. Headlines about an unnamed company remain empty; do not infer identity from an article not supplied. Ambiguous identity may be described in the one-line rationale.
- Labels may include companies outside the catalog. Report out-of-catalog reference support and the sampling limitations. A source-symbol-negative headline can still name a catalog company.
- Freeze completed labels and their hashes before any scored run. No relabelling or alias/threshold tuning after viewing results.

## Metrics and no-regression checks

Evaluate the **subject** links: `link(text) - analyst_only(text)`, consistently for both old and extended. Also report mention links for inspection, so excluded analyst mentions are visible.

For each headline, TP is the intersection of predicted/reference entity sets. Wrong links are predicted entities outside the reference set, counted as headline-company links. A company occurring twice in one headline counts once.

- Primary precision: mean of per-headline TP/predicted-count over headlines with a nonempty predicted set. Empty prediction has undefined precision and is excluded; report its count. If all predictions are empty, precision is null and adoption fails.
- Primary recall: mean of per-headline TP/reference-count over headlines with a nonempty reference set. Report excluded empty references. If none has a reference, recall is null and adoption fails.
- Per-headline F1: mean of 2TP/(predicted-count + reference-count) over headlines with a nonempty union. Empty/empty rows are counted separately, not rewarded as correct company links.
- Report pooled entity-link precision/recall/F1, TP/FP/FN, exact-set accuracy, wrong-link total and all denominators as secondary checks. No sentiment/event accuracy claim and no population extrapolation.

Set B: use only the existing E23 blind file's ID/headline/target_company fields, **243 pairs**. Do not rerun its gate or use its credit labels. Compare old and extended link sets projected onto the original 15, and the given target's membership. Count differences in either check; expected **0**. Additional outside-index mentions are reported separately and do not count as original-company differences. No E23 texts are read to write probes or aliases.

Synthetic probes: freeze **60 invented rows**, six per category: target-price look-alikes, visa applications, ford/river, gap/coverage, apple/harvest, analyst desks, cashtags, possessives, two-company headlines, and other ambiguous/legal names (Amazon/Caterpillar/Oracle/Dow/Capital One). Store expected subject sets before reading E23 texts or running real evaluation. Report each category's passes/total and failures, clearly separate from real-news metrics. Unit tests may use invented fixtures before the scored run; they are not real-data selection.

Batch guard: before candidate implementation, capture the canonical output SHA-256 of a small invented, fake-scored batch fixture. A test recomputes it with the default batch loader after the extension and must match. Also assert the original loader still has exactly its 15 members and that stored API gates retain it even when extended live linking is selected. Never rescore stored real signals.

## Fixed decision and run procedure

Adopt for API/live only if **all** hold on the frozen sample: extended primary recall is higher than old primary recall; extended primary precision is **at least 0.95**; and Set B has **zero original-company differences**. Otherwise leave `extended_linker=false` by default. Synthetic probes diagnose behaviour and are reported separately; they do not substitute for Set A/B.

`--run` refuses if `data/outputs/linker_extended.json` already exists, takes an exclusive run lock, validates sample/label/probe/source/candidate hashes and writes one result. All selected rows and metrics are stored, including failures. No second scored run or post-result tuning. A pre-output implementation error may be fixed without changing rules, labels or samples. Freeze the rules text, candidate/probe files, source hashes and completed labels before running. Append results below afterwards; retain the pre-run rules hash/text.

Limits: AI labels from the same resolver, one small sample, old US large-cap history, partly company-conditioned source collection, potentially noisy/multiple source symbols, unverified current index membership, manually chosen aliases, and no fresh-model or whole-portfolio accuracy result. These are coverage/precision checks only.

## Result

One offline run on 19 October 2026. All 200 labels were completed from text alone in four batches of 50 before either linker was scored. They are AI-made by the same resolver, with no independent human check. The sample was not changed after scoring.

| Set A: 200 real headlines | Original 15 | Extended 99 |
|---|---:|---:|
| Primary per-headline precision | 0.752381 | 0.763566 |
| Primary per-headline recall | 0.338602 | 0.485536 |
| Per-headline F1 | 0.371238 | 0.504599 |
| Wrong headline-company links | 26 | 33 |
| Correct headline-company links | 85 | 118 |
| Pooled link precision | 0.765766 | 0.781457 |
| Pooled link recall | 0.329457 | 0.457364 |
| Pooled link F1 | 0.460705 | 0.577017 |
| Exact subject-set accuracy | 0.335 | 0.460 |
| Headlines with predictions (precision denominator) | 105 | 129 |
| Headlines with references (recall denominator) | 174 | 174 |

F1 uses 175 nonempty-union headlines; 25 empty/empty rows receive no linking credit. There are 26 empty references. Of 258 reference links, 127 are outside the 99-company catalog. This balanced provenance sample is not balanced by true company subjects and is not a population precision estimate. The local eligible pools contained 53,815 inside and 48,012 outside unique headlines; 100 of each were sampled once with seed 22.

Set B: **20 of 243 pairs change their original-company projection and target membership**, failing the zero-difference condition. The guards remove 18 AAPL and three AMZN mention links across those 20 rows. There are no added original-company links. A further 26 rows have outside-company additions, reported separately. Existing E23 files and the batch/default linker remain unchanged; no gate was rerun.

| Separate synthetic probes | Passed / total |
|---|---:|
| Target-price look-alikes | 6 / 6 |
| Visa applications | 6 / 6 |
| Ford/river | 6 / 6 |
| Gap/coverage | 6 / 6 |
| Apple/harvest | 6 / 6 |
| Analyst desks | 5 / 6 |
| Cashtags | 6 / 6 |
| Possessives | 6 / 6 |
| Two companies | 6 / 6 |
| Other ambiguous/legal names | 6 / 6 |
| Total | 59 / 60 |

The failing invented headline is “Morgan Stanley raises Oracle price target”: the fixed context list does not admit Oracle here, leaving no subject link. No rule, alias, probe, sample or label was amended after this run.

**Decision: do not adopt as the live default.** Recall rises, but precision is below 0.95 and Set B has 20 differences. `extended_linker=false` remains in `config/default.toml`; setting it true explicitly enables the 99-company linker for `/analyze` and LiveFeed only. Each new response entity carries `in_index`; outside-index entities have no attached book/portfolio exposure. The GDELT retrieval query is unchanged: this task expands recognition of incoming headlines, not news-source retrieval coverage. Old live cache entries are not rewritten.

The original batch fixture digest, captured before the candidate existed, is `fe7badc6afaabe08a36adf785c37e1df4f32db8775cf17fc714bd8f26e8bea3c`; it still matches with the live flag either false or true. Stored API gates also retain the 15-company loader when the live flag is true.

Files: the original/extended catalogs and synthetic probes are in `data/reference/`; the live loader is `src/riskengine/ingest/linking_extended.py`; the sample/labels, hidden provenance, sampling record and sealed hashes are in `private_notes/annotation_uploads/task1/linker_extended/`. The evaluator is `src/riskengine/eval/linker_extended.py`; `--label` exports the blind file, `--freeze` seals completed labels and `--run` refuses a repeat. The retained exclusive run lock records the attempt. API/config wiring and fixture tests are in `api.py`, `config.py`, `config/default.toml` and `tests/test_linker_extended.py`.

Pre-run rules SHA-256: `42b6c39454c523889e2aac743a839161af3ca00e201c396ff8b437939c6cc92b`. Complete per-row results, metric denominators, E23 differences, probe failure and input/code hashes: `data/outputs/linker_extended.json`, SHA-256 `ac6c28eefafca0b4918d7d41374e2b4a9250014051255ccae4ae9b66db065e0c`. Saved-row metrics were independently recomputed without rerunning either linker. No sentiment or event accuracy gain is claimed.

Validation: focused fixture/config checks pass (37 cases); full offline suite **913 passed, 1 expected failure**, in 170.22 seconds, with the existing Starlette/httpx deprecation warning. Ruff `src tests` and whitespace checks pass. All 188 pre-existing protected file hashes and all 99 model-file sizes/mtimes match. Tests ran outside the sandbox because FastAPI TestClient's sandbox stall was established during prompt 21; dependency/model downloads remained disabled. No network, learned-model/credit-gate evaluation, old-output edits, commit or push.

## Secondary check: adjudicated labels (19 Oct 2026)

A second AI labeller (a separate pass, labels written before the first labeller's were opened) agreed with the frozen labels on all but 6 of 200 headlines once ticker-versus-name writing was ignored. Four of those six were settled from the written label guide (remove a guessed identity from a truncated headline; remove a partner platform; add two explicitly named listed companies), giving `labels_adjudicated.csv`. The stored linker outputs were re-scored on them without running either linker again (`data/outputs/linker_extended_adjudicated.json`). The frozen-label result above stays the primary result.

| | frozen labels | adjudicated labels |
|---|---:|---:|
| 15-company linker: precision / recall | 0.752 / 0.339 | 0.762 / 0.342 |
| extended linker: precision / recall | 0.764 / 0.486 | 0.771 / 0.488 |
| extended linker: exact subject-set accuracy | 0.460 | 0.465 |

The adoption rule is still not met (precision below 0.95; 20 differences on the original-company check). Both labelers are AI, so this shows consistency, not independent truth.
