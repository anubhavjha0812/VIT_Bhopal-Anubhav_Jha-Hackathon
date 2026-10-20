# E24: Event classes on news the engine was not built on

Rules fixed 2026-10-11, approved v4, before any sample is drawn or label made. Written down before the experiment was run; unchanged here.

## Rules

- **Weaknesses addressed (from my own error analysis):** W1 "Product news" swallows any company
  news; W3 earnings news without $/EPS is missed; W4 low-confidence model guesses still get an
  event class; W5 news and tweets are measured together. **Deferred:** W2 (war and sanctions read
  as Macroeconomic) needs a world-news source and goes to a separate experiment; E24 does not
  change Geopolitical or Macroeconomic. E13 credit rules are unchanged. The other project's
  gold-2 is a seen diagnostic only and decides nothing.
- **Sets:**
  - **Dev:** E3's 300 FNSPID headlines (2019 – 10 Jun 2020), split before labelling into D1 (150,
    rule design and optional temperature) and D2 (150, candidate and threshold choice). The split
    is **prediction-stratified** (by E3's design half and the engine's predicted class, seed 24),
    not true-class stratified: no labels exist at split time. Real label support of D1 and D2 is
    reported after annotation, before any performance number.
  - **Final news (200):** FNSPID 2017-01-01 – 2018-12-31, E17 history kept by the relevance rules,
    role subject, unique headlines. An **earlier-period evaluation**, not a future test: the dev
    period and the model's training data are later.
  - **Final tweets (100):** equinxx 2021–22 unique tweets. A **later-period, cross-source
    evaluation**. News and tweets are never pooled.
  - **Selection:** simple random sample, seed 24, from a stably sorted frame (published time,
    text); no class, prediction or label is used. Frame bias (company-linked news, mostly Other)
    is reported.
  - **Excluded** (exact normalised match or cosine ≥ 0.9): TFN topic train+valid, TFNS
    train+valid, E3, the E23 sample, E22b alerts, all E17 relevance samples, the E21 tweet
    sample, FinEntity, the other project's gold-1, gold-2 and entity check, and near-duplicates
    within the final sample.
- **Taxonomy and metric:** the existing 12 classes (Default, Credit downgrade, Covenant breach,
  Geopolitical, Macroeconomic, M&A, Product news, Earnings, Analyst rating, Legal/Regulatory,
  Management change, Other), fixed now. "Product news" is labelled under a new narrow definition
  (a product event: launch, unveiling, release, recall, product deal), written into the guide
  before annotation; E16/TFN-topic results stay as they are. **Primary metric: macro-F1 over all
  12 classes, `zero_division = 0`, never re-selected by support.** All candidates, **including
  A**, are scored against the same new-definition labels; A's predictions are not changed.
- **Annotation (AI-assisted, human-adjudicated; allowed by the organisers' "AI assistance … with
  full honesty", no rule bans AI for labelling; no AI tool is part of the engine):**
  - Labeller A = ChatGPT, labeller B = Gemini. Recorded: exact model/version, date, full prompt
    (with the guide), settings, interface. Each sees only the blind file and the guide; neither
    sees the other's output or any engine prediction. Their outputs are stored apart, hashed.
  - **Primary-event rule:** the label is the main news fact the headline reports. Precedence only
    for documented overlaps: agency credit action + stock rating → credit class; earnings +
    guidance or dividend → Earnings; regulator decision on a deal → Legal/Regulatory if the
    decision is the news, else M&A; recall + lawsuit → Legal/Regulatory, recall alone → Product
    news; an executive leaving amid a probe → Management change if the departure is the news.
  - A ≠ B, or either marks it ambiguous → a person adjudicates (`human_adjudicated`). A = B →
    `ai_agreed`; a 20% random audit of agreed rows (seed 24) by a person, whose label replaces
    the AI one (`ai_agreed_human_audited`); the audit correction rate is reported with a Wilson
    95% interval. Undecidable → `excluded` (counted). Cohen's kappa of A vs B is reported. The
    labels are described as AI-assisted, not independent human gold.
- **Candidates:** A = current engine; B = A + narrow Product-news mapping (other company news →
  Other); C = B + an earnings precedence rule; D = C + one global abstention threshold. Order:
  E13 credit rules → earnings rule → product rule → model; abstention applies only to model
  predictions (below threshold → Other, counted as a prediction, not dropped). Rules are written
  from these definitions on D1 only. No code, keyword list, weights or labels of the other
  project are used.
- **Calibration and threshold:** temperature scaling (Guo et al., 2017) is fitted on D1 only, and
  only if D1 has at least 5 classes with ≥ 10 labelled rows; else raw probabilities. Threshold
  grid fixed: {0.30, 0.35, …, 0.70}, chosen on D2 only (ties → lowest); a boundary choice is
  reported as a limitation, the grid is never widened. No class-specific tuning for a class with
  < 10 D1 rows.
- **Choice (D2 only):** highest 12-class macro-F1, provided no class with ≥ 10 D2 rows loses more
  than 0.05 F1 against A; within 0.01 → the simpler candidate. The chosen candidate, its rules,
  threshold and temperature are frozen (hashes) before the final set is labelled.
- **Final evaluation (once):** paired bootstrap on the final rows: the same resampled row
  indices for A and the chosen candidate, 1,000 resamples, seed 0, 95% percentile interval of
  Δ = macro-F1(chosen) − macro-F1(A). **Every resample uses the fixed 12-class list with
  `zero_division = 0`, even when a class is absent from that resample.**
  - News: pass if the lower bound > 0; fail if the upper bound < 0; else, or if the interval is
    wider than 0.20, **inconclusive**.
  - Tweets (non-inferiority, margin 0.05): pass if the lower bound > −0.05; fail if the upper
    bound < −0.05; else, or if wider than 0.20, **inconclusive**.
  - The engine switches (`events_rules = on`) only if news **and** tweets pass; otherwise A
    stays. No retest with a larger sample, no change of sample or threshold after the result.
- **Reported, news and tweets apart:** rule coverage by rule, model coverage, abstention rate,
  per-class precision, recall, F1 and support, confusion matrices. Classes with < 5 or no support
  are named; no reliable per-class claim is made for them. TFN-topic and gold-2 scores are
  reported only.
- **Safeguards:** deterministic exclusion (two separate runs must give the same sha256 of the
  blind file, or no freeze); provenance per row (source dataset, file and its sha256, original
  id or timestamp, frame rule); final labels are read only when `e24_frozen.json` exists and the
  blind-file hash matches, and only once; read-only backups of frozen artefacts in
  `private_notes/e24_frozen/` with a sha256 manifest, checked before evaluation. Nothing existing
  is changed; outputs are `e24_*`. Licences: FNSPID CC BY-NC 4.0, equinxx CC0 (tweet text under
  platform terms, gitignored), TFN MIT; Guo et al. (2017) cited.

## Result

- **Question:** Rules for product/M&A overprediction, earnings and analyst routing
- **Outcome:** D1-only: reproducible per-row A macro-F1 0.3218 / accuracy 0.4379; B 0.4548 / 0.8693; C 0.5156 / 0.9216; D=0.50 demonstration 0.5172 / 0.9281. Product 4/4 correct; M&A 0 predictions, both positives missed. Production A now matches evaluation exactly
- **Verdict:** D1 development only; D2/final pending, no selection/freeze; production reuses per-row inference
- **Stored results:** e24_d1_production_alignment_report.md, e24_d1_reproducibility_report.md, e24_d1_rules_reproducible.json, docs/experiments/E24_AMENDMENT.md

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
