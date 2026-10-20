# Helix: design and evaluation decisions

Helix uses a shared text pipeline for two workflows: portfolio rebalancing
(Module A) and bank stress testing (Module B). NLP identifies companies,
sentiment and events; explicit financial calculations turn those signals into
portfolio weights or scenario losses.

This guide explains the current choices as of **20 October 2026** and points to
their evidence. The original [evaluation decisions](../DECISIONS.md) retain the
E1–E24 and M1 protocols, including their dates, thresholds and historical wording.
That file is preserved byte for byte because saved evaluation records include
its SHA-256 hash. This guide adds context without rewriting those records.

## Current implementation

The [configuration](../../config/default.toml) selects the current models and
switches. Saved replay files retain the scores produced when they were built;
changing a default does not rescore them.

| Area | Current choice | Reason and boundary |
|---|---|---|
| Sentiment | E27 company-name/headline pair model; `own` headline model when a company name is missing. | A headline can affect named companies differently. The saved credit replay still uses `own`; the entity-scored replay is a separate comparison. See [E27](../experiments/E27_ENTITY_SENTIMENT.md). |
| Events | `student_rules`: candidate C rules over the E25 v3 student. | Combines learned event probabilities with explicit routing rules. V3 was selected using E44b; its score is a selection result. See the [scoring plan](../experiments/E44_SCORING_PLAN.md) and [saved results](../../data/outputs/e44_results.json). |
| Model serving | Separate INT8 ONNX text models running on CPU. | Keeps inference practical on one host. Model structures and label mappings are in the [model guide](../models/README.md). |
| Impact | Learned v3 move score for batch rows with sufficient price history; formula fallback. Live `/analyze` uses the formula. | Expected price movement and credit-event severity serve different purposes. The credit gate uses formula severity even when a learned move score exists. See [impact v3](../experiments/IMPACT_V3.md) and its [implementation](../../src/riskengine/impact/learned.py). |
| Company linking | Candidate 3 covers 99 company names in live/API analysis; historical batch/replay portfolios cover 15 companies. | More recognised names do not create new portfolio exposures. The owner enabled candidate 3 after its original 95% precision rule failed; that outcome remains recorded. See [linker selection](../experiments/LINKER_EXTENDED_V3.md#current-selection). |
| Alert routing | Credit-class, subject/obligor and formula-severity checks, followed by agency or corroborating-text evidence and precision rules. A separate market gate handles Geopolitical/Macroeconomic events. | Filters unsupported credit alerts and keeps market events separate. Corroboration checks for a different earlier text within 24 hours; it does not establish independent reporting. See the [credit gate](../../src/riskengine/gate.py), [precision rules](../experiments/E36_GATE_PRECISION.md) and [market gate](../../src/riskengine/market_gate.py). |
| Module A | Long-only sentiment tilt, 15% per-name cap, 5 bp turnover costs and an equal-weight comparison. | Makes allocation constraints and trading costs explicit. Signals become usable after their publication day; the reported comparison shows no proven edge. See [portfolio implementation](../../src/riskengine/module_a.py) and [results](RESULTS.md#module-a-portfolio-and-placebo). |
| Module B | Explicit credit, valuation and capital calculations on a hypothetical book; `rates = "moved"`. | Supports inspection of loss assumptions and scenarios. Exposure sizes and losses are demonstration assumptions, not calibrated forecasts. See [book and capital rules](../experiments/E14_MODULE_B_BOOK_SHOCK_CAPITAL.md) and [rate shocks](../experiments/E22_INTEREST_RATE_SHOCK_IN_MODULE_B.md). |
| Application and live state | FastAPI serves the React dashboard and `/api`. Persisted live stress previews are opt-in (`live_decisions = false`). | One local service supports replay and analysis. Live previews keep separate records and do not execute trades. See [application entry point](../../src/riskengine/site.py) and [live decision flow](../experiments/LIVE_DECISION_FLOW.md). |

## Reading the evidence

The historical protocols describe what was planned before an experiment. Later
amendments and saved outputs describe what actually happened. Read them together
when checking a result or reproducing a run; an old plan is not evidence that
every planned procedure was completed.

- **Label provenance:** new project event and credit labels are AI-made or
  AI-adjudicated, with no human check. SEntFiN supplies the human-made labels for
  E27. References to hand labelling or human adjudication in the original plans
  must be read with the [labelling amendment and addendum](../experiments/E25_LABELLER_AMENDMENT.md)
  and [provenance report](RESULTS.md#label-provenance).
- **Event selection:** E44b scores 822 AI-agreed rows from a 900-row sample;
  78 disagreements were dropped. The same set selected v3, so its macro-F1 is
  not an untouched final test. See the [sampling protocol](../experiments/E44_EVENT_TEST_V2.md)
  and [scoring plan](../experiments/E44_SCORING_PLAN.md).
- **Credit validation:** E36b's precision is measured on seen E23 design data.
  The wider 8-K sample has a different composition; its score is not directly
  comparable. See [alert-gate results](RESULTS.md#module-b-alert-gates).
- **Financial interpretation:** Module A has no demonstrated advantage over
  equal weight in the reported comparison. Module B illustrates stress
  calculations on assumed exposures; it does not establish forecast accuracy.

## Original protocol index

These links open the preserved sections of [DECISIONS.md](../DECISIONS.md).
Experiment IDs keep their original numbering; E4 appears at the end of that file.

| Topic | Original sections |
|---|---|
| Public benchmark sets and efficiency | [E1: sentiment](../DECISIONS.md#e1-sentiment-test-set-fixed-2026-10-06), [E2: events](../DECISIONS.md#e2-event-class-test-set-fixed-2026-10-06), [E4: efficiency](../DECISIONS.md#e4-efficiency-fixed-2026-10-06) |
| Training, distillation and CPU deployment | [E5: fine-tuning](../DECISIONS.md#e5-fine-tuning-setup-fixed-2026-10-06-before-any-training-run), [E7: teacher/student](../DECISIONS.md#e7-sentiment-teacher-and-distilled-student-fixed-2026-10-09-before-any-training-run), [E9: INT8](../DECISIONS.md#e9-int8-onnx-version-fixed-2026-10-09-before-any-int8-model-is-scored), [E10: shared encoder](../DECISIONS.md#e10-one-shared-encoder-with-two-heads-fixed-2026-10-09-before-any-training-run), [E11: model switch](../DECISIONS.md#e11-engine-switch-to-our-models-fixed-2026-10-09-before-the-topic-int8-model-is-scored) |
| Early learned-impact designs | [E6: v1](../DECISIONS.md#e6-learned-impact-score-fixed-2026-10-06-before-any-impact-model-is-fitted), [E8: v2](../DECISIONS.md#e8-impact-score-v2-fixed-2026-10-09-before-any-v2-model-is-fitted) |
| Analyst roles and credit gating | [E12: company subject](../DECISIONS.md#e12-a-bank-that-only-gives-a-stock-rating-is-not-the-subject-fixed-2026-10-09-before-the-code), [E13: credit gate](../DECISIONS.md#e13-credit-rules-and-the-module-b-trigger-gate-fixed-2026-10-09-before-the-code) |
| Bank book, capital and sensitivity | [E14: stress rules](../DECISIONS.md#e14-module-b-book-shock-and-capital-rules-fixed-2026-10-10-before-the-code), [M1: sensitivity](../DECISIONS.md#m1-module-b-sensitivity-checks-fixed-2026-10-11-before-the-checks-are-written-or-run), [E22: rate shocks](../DECISIONS.md#e22-interest-rate-shock-in-module-b-fixed-2026-10-11-before-any-code-change-or-run) |
| Market forecasts and alert information | [E15: monthly backtest](../DECISIONS.md#e15-state-aware-market-move-monthly-backtest-fixed-2026-10-10-before-the-code), [E22b: alert-only check](../DECISIONS.md#e22b-does-a-credit-alert-itself-carry-information-fixed-2026-10-11-before-any-alert-is-counted-or-any-return-computed) |
| Event taxonomy and original development sample | [E3: gold-set plan](../DECISIONS.md#e3-gold-set-sampling-fixed-2026-10-06-before-any-credit-class-model-is-trained), [E16: nine learned classes](../DECISIONS.md#e16-nine-learned-event-classes-fixed-2026-10-10-before-any-training-or-scoring), [E24: new-news protocol](../DECISIONS.md#e24-event-classes-on-news-the-engine-was-not-built-on-fixed-2026-10-11-approved-v4-before-any-sample-is-drawn-or-label-made) |
| Longer portfolio history | [E17: Module A](../DECISIONS.md#e17-longer-history-for-module-a-fixed-2026-10-10-before-any-label-or-return-is-seen) |
| Cross-source sentiment evaluation | [E18: additional sets](../DECISIONS.md#e18-more-public-sentiment-test-sets-fixed-2026-10-11-before-any-of-them-is-scored), [E20: model choice](../DECISIONS.md#e20-choosing-the-engine-sentiment-model-on-news-as-well-as-tweets-fixed-2026-10-11-before-any-dev-or-new-test-row-is-scored), [E21: joint model](../DECISIONS.md#e21-one-sentiment-model-for-tweets-and-news-fixed-2026-10-11-before-any-training-labelling-or-scoring) |
| Credit-gate reference sample | [E23: protocol](../DECISIONS.md#e23-credit-event-gold-set-for-the-frozen-alert-gate-fixed-2026-10-11-before-any-sample-is-drawn-or-label-seen) |

## Supporting records

Use the [experiment register](../experiments/EXPERIMENTS.md) for the broader
history, the [results report](RESULTS.md) for comparisons and limitations, and
the [README](../../README.md) for the submission overview and run commands.
Later protocols, amendments and run notes are kept in [experiments](../experiments/).
Saved CSV/JSON artifacts under [data/outputs](../../data/outputs/) retain the
measurements and provenance of their recorded runs.
