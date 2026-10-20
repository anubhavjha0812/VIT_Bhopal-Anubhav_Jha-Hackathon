# E44: a new sealed real-news event test

Fixed before sampling, 19 October 2026. This is a separate test, not a rerun or replacement of E24. E24 candidate C is already frozen and tested; it stays unchanged. No model or rule is scored in this task.

The earlier 12-class guide is `private_notes/annotation_uploads/task1/development/GUIDE.md` (same bytes as `data/gold/e24/GUIDE.md`). Its class definitions and primary-event rule stay unchanged. This protocol uses Claude and ChatGPT, AI adjudication when needed, and identity-only web research as specified below; it supersedes the guide's old no-search and human-adjudication procedure, not its label definitions. No human validation is claimed.

## Pool and exclusions

Use only the local FNSPID full news file, streamed in 50,000-row chunks. Read the five columns in the specification, never article bodies or a model output. Dates must be parseable and within 1999-2023 inclusive. Preserve the title verbatim except outer whitespace. Require 4-80 whitespace-separated words, at least 20 letters, at least 90% of letters ASCII, and no HTML tags or title beginning with a URL. This is a mechanical readability screen, not verified language or editorial quality. No tweets, synthetic text or rewritten titles. FNSPID's recorded licence is CC BY-NC 4.0; keep the sample private and retain source URL/publisher and file hashes in hidden provenance. No download is made.

The exact exclusion list is `exclusion_specs` below. Every listed file is required; missing files stop sampling. Read text columns only. It includes the complete impact/replay/fresh news input files as conservative supersets, E24 dev/finals, E23, R3, both credit samples, all three linker samples, E25 pool/extra/official/student inputs, encoder and sentiment inputs. Also require the validated offline E25 exclusion snapshot using `e25_sec8k.load_cached_exclusions`, whose groups/hash counts are bound to its original manifest; refresh final texts through `e25_pool.exclusions` and `FINAL_FILES`. Do not call the network-capable E24 corpus loader.

Normalise using the existing E18 `leaks.norm`, then collapse whitespace. Remove exact matches to any prior text. Combine the unique normalised reference texts in sorted order into fixed 5,000-text blocks. Apply the existing E25 `remove_seen` and `assert_clean` to these blocks (E18 TF-IDF word 1-2-gram cosine >=0.9). Blocking bounds memory and defines the IDF comparison groups; it is not a claim of identical cosine to fitting the whole reference corpus at once. Recheck on the final written sample. Hash the ordered input-file/column list, every actual input file, exclusion text set, and reference blocks. No exclusion reads labels or scores. Existing files remain untouched.

## Candidate draw, fixed before seeing labels

The ordered regexes below are sampling hints, not event labels. First matching hint wins, in the listed order; no hint gives the Other sampling bucket. This priority reserves rare credit cues before broad company-news cues. Rank each unique normalised title by SHA-256 of seed 44, source path, hint and title key; keep the smallest 3,000 hashes per source/hint, with a fixed metadata tie-break. Count eligible occurrences and unique retained candidates separately; occurrence counts are not unique clean pool support. A bounded reservoir is a disclosed preselection limit, not an exhaustive clean-pool count.

Draw 90 candidates for each credit hint, 60 for each of the other eight hints and 150 from the no-hint bucket: at most 900 rows. This oversamples to aim for at least 30 eventual gold rows in each non-Other class and about 150 Other. Hints cannot guarantee gold balance. Never present hint counts as class support. Filter exact/near prior-text matches, refill only from the same fixed reservoir in rank order, and repeat to stability for at most ten rounds; fail if not stable. Keep fewer when a bucket is short; no cross-bucket fallback, synthetic padding, new keywords or second draw after labels. Within the sample require unique normalised headlines; event-level independence is not guaranteed. Shuffle final rows by a separate seed-44 hash and assign opaque sequential `E44_0001` IDs. Labellers see only `id,text`. Source/date/ticker/publisher/class hint/rank/row number stay in `provenance.csv` with owner-only permissions. This file is hidden by workflow, not cryptographic isolation from the same filesystem user.

`--sample` creates the sample directory once and refuses any existing output directory, including a partial attempt. `--check` verifies the exact input list/content, script/protocol/guide, sample/provenance hashes and alignment, normalised uniqueness, eligibility, hints, and the final exact/near exclusion checks. It never reads annotation files. A hash mismatch stops; no reseal is automatic.

## Independent annotation and freeze

Use fresh separate labeller sessions with no inherited candidate/provenance context. Claude and ChatGPT each read only their prompt, the existing guide and `sample.csv`, in batches of 50, file order. Write only their own `labels_claude.csv` or `labels_chatgpt.csv`: `id,label,ambiguous,rationale`, one valid class, Y/N flag, rationale at most 20 words. Neither may read the other output, provenance, sampling metadata or model results. Record actual model name or not shown, UTC time, interface and prompt/guide/input/output SHA-256. Web is allowed only for company/event identity, with every URL and settled fact in the labeller's separate evidence CSV; no lookup of model predictions or other annotations.

After both complete, a separate authorised annotation resolver may inspect their labels to report raw agreement and Cohen's kappa over all valid paired rows (also report flagged rows separately), then settle disagreements and ambiguous rows from the unchanged guide with recorded reasons and identity URLs. Preserve initial labels. Final provenance must say AI agreement or AI adjudication, never imply human validation. Freeze `labels_final.csv` by SHA-256 in `labels_final_record.json`, bound to sample, guide, both raw outputs, settlement record and protocol, before any model scoring. Report actual gold support and every class below 30; do not refill after seeing labels.

This set is TEST ONLY. No training, tuning, thresholds, prompt selection or model choice uses it. Only labellers and the annotation resolver may read new labels before freeze; developers/evaluators may not. Candidate models and their decision rule need a separate later document and cannot be chosen from these labels. No evaluation command is implemented here.

## Metrics fixed now for later scoring

Primary: macro-F1 over the classes with at least ten gold rows in the frozen full reference; freeze that class list before predictions. Secondary: the same macro-F1 with Other removed, plus accuracy over all valid rows, and all 12 classes' precision/recall/F1/support. If a denominator/list is empty, report unavailable; use zero_division=0 for an absent prediction, and mark support-zero classes unscored. Do not reinterpret this as population prevalence or the old fixed-12 E24 metric.

For 95% intervals use 1,000 moving-block bootstrap draws, seed 44: sort rows by hidden publication date then id; block length ceil(sqrt(n)), sample overlapping contiguous blocks uniformly with replacement, concatenate and truncate to n. Reuse the same resampled indices for every candidate/metric; keep the full-reference eligible class list fixed in every replicate. Report percentile 2.5/97.5 intervals, duplicate-event/old-news/source/AI-label limits, and low-support instability. There are no model results in this task.

## Fixed sampling specification

The JSON below is the executable sampling specification. No field changes after writing this document. Sampling records this whole document's SHA-256; results and status go elsewhere.

Pattern-list SHA-256: `804e4d3f3af422f4b2542ec4f13232053ec765fd866d36230c241dfc2b623338`.

Exclusion input-list SHA-256: `a590141de547a73f75158e4a1693a010119144c6ace649cfb46cab6f29c6e41f`.

```json
{
  "seed": 44,
  "sources": [
    "data/raw/fnspid/All_external.csv"
  ],
  "columns": [
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Publisher"
  ],
  "date_start": "1999-01-01",
  "date_end": "2023-12-31",
  "chunk_rows": 50000,
  "reservoir_per_hint": 3000,
  "near_reference_block": 5000,
  "exclusion_rounds": 10,
  "patterns": {
    "Covenant breach": "\\bcovenants?\\b|\\b(?:waiv\\w*|breach\\w*|violat\\w*).{0,45}\\b(?:loan|debt|credit|lender|terms)\\b|\\b(?:loan|debt|credit).{0,45}\\b(?:amend\\w*|waiv\\w*|breach\\w*|violat\\w*|accelerat\\w*)",
    "Default": "\\bbankrupt\\w*|\\bchapter\\s+(?:7|11|15)\\b|\\binsolven\\w*|\\bmiss\\w*.{0,30}\\b(?:coupon|payment)\\b|\\bdefault\\w*.{0,25}\\b(?:debt|loan|bond|payment)\\b|\\b(?:debt|distressed).{0,25}\\brestructur\\w*",
    "Credit downgrade": "(?=.*\\b(?:moody\\w*|fitch|s\\s*&\\s*p|standard\\s+.{0,5}poor\\w*|dbrs)\\b)(?=.*\\b(?:downgrad\\w*|cut\\w*|lower\\w*|negative|watch\\w*)\\b)",
    "Geopolitical": "\\b(?:war|wars|military|missile\\w*|sanction\\w*|tariff\\w*|trade\\s+(?:war|dispute)|election\\w*|ceasefire|diploma\\w*|nato|brexit|referendum)\\b",
    "Macroeconomic": "\\b(?:fomc|fed|federal\\s+reserve|ecb|central\\s+bank\\w*|inflation|gdp|nonfarm\\w*|payroll\\w*|unemployment|recession|fiscal\\s+stimulus|quantitative\\s+easing|interest\\s+rates?|oil\\s+prices?|currency|currencies|treasury\\s+yields?)\\b",
    "Management change": "\\b(?:ceo|cfo|chief\\s+executive|chief\\s+financial|chairman|chairwoman|executive).{0,60}\\b(?:resign\\w*|retir\\w*|appoint\\w*|replac\\w*|step\\w*\\s+down|leav\\w*|nam\\w*)\\b|\\b(?:appoint\\w*|nam\\w*|replac\\w*).{0,60}\\b(?:ceo|cfo|chairman|chief\\s+executive)\\b",
    "M&A": "\\b(?:acquir\\w*|acquisition\\w*|merger\\w*|merge\\w*|takeover\\w*|buyout\\w*|stake\\s+purchase|asset\\s+sale|spin\\s*off|ipo|initial\\s+public\\s+offering)\\b",
    "Product news": "\\b(?:launch\\w*|unveil\\w*|recall\\w*|introduc\\w*|debut\\w*|releas\\w*|supply\\s+(?:deal|agreement)|product\\s+(?:deal|agreement))\\b",
    "Earnings": "\\b(?:earnings|eps|revenue\\w*|quarterly\\s+results|annual\\s+results|profit\\w*|guidance|dividend\\w*)\\b",
    "Analyst rating": "\\b(?:upgrad\\w*|downgrad\\w*|price\\s+target|target\\s+price|initiates?\\s+coverage|outperform|underperform|buy\\s+rating|sell\\s+rating)\\b",
    "Legal/Regulatory": "\\b(?:lawsuit\\w*|sues?|sued|litigation|regulat\\w*|probe\\w*|investigat\\w*|settlement\\w*|court|antitrust|sec|ftc|fines?|approval)\\b"
  },
  "quotas": {
    "Covenant breach": 90,
    "Default": 90,
    "Credit downgrade": 90,
    "Geopolitical": 60,
    "Macroeconomic": 60,
    "Management change": 60,
    "M&A": 60,
    "Product news": 60,
    "Earnings": 60,
    "Analyst rating": 60,
    "Legal/Regulatory": 60
  },
  "other_quota": 150,
  "baseline_cache": "data/cache/e25/sec8k_exclusions.json",
  "baseline_manifest": "data/outputs/e25_sec8k_manifest.json",
  "exclusion_specs": [
    [
      "data/gold/gold_to_label.csv",
      "headline"
    ],
    [
      "data/gold/credit/credit_to_label.csv",
      "headline"
    ],
    [
      "data/gold/e24/final_news_to_label.csv",
      "text"
    ],
    [
      "data/gold/e24/final_tweets_to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/development/D1_to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/development/D2_to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/retest_set/to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/credit_heldout/to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/credit_8k/to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/credit_8k_full/to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/linker_extended/sample.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/linker_extended_v2/sample.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/linker_extended_v3/sample.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/macrogeo_adjudication/to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/official_geo/to_label.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/e25_student_input/train.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/e25_student_input/valid.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/e25_student_input_v3/train.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/e25_student_input_v3/valid.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/e28_encoder_input/train.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/e28_encoder_input/valid.csv",
      "text"
    ],
    [
      "private_notes/annotation_uploads/task1/e28_encoder_input/eval_texts.csv",
      "text"
    ],
    [
      "data/cache/e25/pool.csv.gz",
      "text"
    ],
    [
      "data/cache/e25/extra.csv.gz",
      "text"
    ],
    [
      "data/cache/e25/official_titles.csv",
      "text"
    ],
    [
      "data/cache/e25/student/train.csv",
      "text"
    ],
    [
      "data/cache/e25/student/valid.csv",
      "text"
    ],
    [
      "data/cache/e25/student/train_v3.csv",
      "text"
    ],
    [
      "data/cache/e25/student/valid_v3.csv",
      "text"
    ],
    [
      "data/raw/fnspid/universe_news_named.csv.gz",
      "Article_title"
    ],
    [
      "data/raw/fnspid/universe_news.csv.gz",
      "Article_title"
    ],
    [
      "data/raw/fnspid/fresh_news.csv.gz",
      "Article_title"
    ],
    [
      "data/cache/history_texts.csv.gz",
      "text"
    ],
    [
      "data/cache/e21_train/train.csv",
      "text"
    ],
    [
      "data/cache/e21_train/dev_news.csv",
      "text"
    ],
    [
      "data/cache/e21_train/dev_tweets.csv",
      "text"
    ],
    [
      "data/eval/phrasebank_50Agree.csv",
      "text"
    ],
    [
      "data/eval/fiqa_train.csv",
      "sentence"
    ],
    [
      "data/eval/fiqa_validation.csv",
      "sentence"
    ],
    [
      "data/eval/fiqa_test.csv",
      "sentence"
    ],
    [
      "data/eval/nwgi_train.csv",
      "news"
    ],
    [
      "data/eval/nwgi_test.csv",
      "news"
    ],
    [
      "data/eval/sentfin_v1.1.csv",
      "Title"
    ],
    [
      "private_notes/annotation_uploads/task1/e27_entity_input/train.csv",
      "headline"
    ],
    [
      "private_notes/annotation_uploads/task1/e27_entity_input/dev.csv",
      "headline"
    ]
  ],
  "guide": "private_notes/annotation_uploads/task1/development/GUIDE.md"
}
```
