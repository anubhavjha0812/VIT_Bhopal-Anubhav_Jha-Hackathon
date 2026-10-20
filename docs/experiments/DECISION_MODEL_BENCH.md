# E39 and E40: calibration of our own probabilities, and a decision-model baseline for the credit gate

Rules written on 19 Oct 2026, before any code and before any scored run. They are not changed after a run. Each experiment has one scored run; a run that wrote a result file is never repeated.

## Why

Our credit-class and impact outputs are probabilities that nobody has checked for calibration, and the credit gate still raises false alerts that rules cannot remove (counterparty events: 9 of the 20 false alerts left after E36b). A small decision model (one yes/no probability per question, no free text) could answer "is this the company's own credit event?". We do not know whether it helps, so it is first measured as a baseline. Nothing here changes the live gate, the frozen model or any stored result.

## E39: calibration benchmark (our own models, no new dependency)

**Data.** Stored probabilities and labels only; no model is run again.
- Credit-class model: `data/outputs/e24_d1_model_probabilities.npz` and `e25_candidate_E_E_probabilities.npz` (153 items, 9 classes) against the D1 reference labels used in E24/E25.
- Credit alert score on the E23 pairs (243) and on the 8-K set (287): the stored impact/fire scores in `e24_e23_gate_v6_rows.csv` and `e25_credit_8k_full_rows.csv`, if a probability-like score is stored. If no score is stored for a set, write "not stored" and skip it; do not re-run models.

**Metrics.** Expected calibration error (ECE) with 10 equal-mass bins, Brier score, negative log-likelihood, accuracy. For the binary alert score also a reliability table.

**Methods (fixed).**
1. Temperature scaling for the multiclass outputs (one parameter; the predicted class does not change, so accuracy is unchanged by construction).
2. Platt scaling (logistic on the logit) for a binary score.
3. Abstain rule: abstain when the top probability is below a threshold; report coverage and accuracy at thresholds 0.5, 0.6, 0.7, 0.8, 0.9.

**No leakage.** Every fitted parameter is out-of-fold: 5-fold cross-fitting, seed 39, stratified by label where possible. Report out-of-fold ECE/Brier/NLL before and after. Never fit and score on the same rows. Fold assignments are stored.

**Decision rule.** Calibration is called "useful" for a set only if out-of-fold ECE falls and out-of-fold NLL does not rise. Otherwise "no gain". Either way the result is kept. Nothing is applied to the live engine from E39; it is a reported metric and, if useful, a display option decided later.

**Limits to state.** 153 items over 9 classes is small; some classes have a few rows; the labels are AI-labelled; ECE on small samples is noisy (report a bootstrap interval, 1000 resamples, seed 39).

## E40: decision-model baseline for the credit gate (Laya, run once on Kaggle)

**Model.** Laya, the open-source typed-decision model (Apache-2.0, 421M parameters), Hugging Face `convaiinnovations/laya`. The checkpoint used is the fine-tuned `typed-decisions` subfolder (ModernBERT-large, 1024 context), chosen before any run because the model card says the root base checkpoint is near chance zero-shot (0.362 on its own typed-decisions test) and that the typed-decisions checkpoint reaches 0.766 there. The `laya` package version is pinned at run time and written into the result file with the files' hashes; the repo shows no revision hash, so the download date is recorded. Loaded with `laya.load("convaiinnovations/laya", subfolder="typed-decisions")`. Heavy compute runs on Kaggle, never on the laptop. If the checkpoint cannot be loaded, stop and report; do not substitute another model.

**Known issues from the model card (stated before the run).** The card says it ships overconfident (raw probabilities need refitting; its own ECE 0.081 after temperature), that yes/no (`noul`) questions can follow the `false:`/`true:` labels instead of the input (issue #156, workaround: a two-option `choice`), and that results are not zero-shot guarantees. Therefore each question is asked in two forms. **Primary form (used for the decision rule): `noul`.** Secondary form (reported, not used for the decision): a two-option `choice` ("yes"/"no") with the same wording. The probabilities are used raw with one threshold set on the 8-K set; no temperature is fitted on E23.

**Questions (fixed text, yes/no probability each).**
- Q1: "Does this headline report a credit event (rating action, default, missed payment, covenant breach, restructuring, bankruptcy or similar) of a company?"
- Q2: "Is the credit event reported one of the company <TICKER NAME> itself, not of a counterparty, a customer, a supplier or a debt issue it does not own?"
- Q3: "Is this headline about a securitisation, trust, note or structured-finance deal rather than the company?"

Question wording is fixed here and is not tuned after seeing results.

**Data.**
- Threshold-setting set: the 8-K credit set (287 headlines, `e25_credit_8k_full_rows.csv`). The one threshold for Q1/Q2 is chosen here only, as the value that maximises F1 for the credit label, then frozen.
- Test set: the E23 243 pairs (`e24_e23_gate_v6_rows.csv`). These are seen/design rows (the E36 rules were designed on them), so the result is design-data evidence only.
- No Laya output is used to change the gate, the rules, the labels or the questions.

**Arms scored on the E23 pairs (report all).**
- A: current gate with the E36b rules (83 → 59 alerts).
- B: Laya alone (Q1 and Q2 above threshold).
- C: gate with E36b rules, then Laya as a filter on the fired alerts only (an alert is kept if Q1 and Q2 are above threshold and Q3 is below 0.5).
- Per arm: alerts, true, false, precision, recall, and for C the list of removed alerts with text.

**Decision rule (promote to an unseen-set test only if ALL hold).**
1. Arm C keeps at least 38 of the 39 true alerts (at most 1 lost).
2. Arm C removes at least 10 of the 20 false alerts left under A.
3. Arm C's precision is above arm A's.
4. Latency on Kaggle is reported; the model runs offline after download.

Otherwise: not promoted, the live gate stays as it is, and the result is kept. Even if all four hold, nothing is adopted from E23 alone: adoption needs a held-out set the rules and thresholds have never seen (not built yet), pre-registered separately.

**Limits to state.** 39 true alerts make rule 1 coarse; one labeller family (AI); the questions were written by us; the model's own benchmark numbers are reported by its author and are not evidence for finance news; no human check.

## What is not allowed

No change to `gate.py`, `gate_rules.py`, models, frozen files, labels, E23/8-K sets, earlier outputs. No second scored run. No claim that the decision model is better until an unseen set says so.

## Result

E39 ran once on 19 Oct 2026, offline, from the two specified caches. `data/outputs/e39_calibration.json` contains the full result and OOF probabilities; `e39_calibration_folds.csv` stores each ID's scoring fold. The persistent run lock is kept. No model was run and no engine setting changed.

The exact D1 reference is `reference_v6/development/D1_labels.csv`: its SHA-256 matches the label hash in the E24 cache metadata. Both caches have the same 153 IDs in the same order and the same nine class names; the student's columns were aligned by name. These are **event-head** probabilities, not credit-alert probabilities. The requested E24 cache used batch 64, whereas student E used per-row inference; these results describe those stored caches, not a regenerated model or the current hybrid C.

| Stored head | Stage | ECE | Brier | NLL | Accuracy |
|---|---|---:|---:|---:|---:|
| E24 event model | Before | 0.388167 | 0.873180 | 2.213898 | 0.431373 |
| E24 event model | OOF temperature | 0.184012 | 0.721695 | 1.649613 | 0.431373 |
| Student E | Before | 0.266349 | 0.630513 | 1.563992 | 0.647059 |
| Student E | OOF temperature | 0.146252 | 0.538949 | 1.198694 | 0.647059 |

Both sets are **useful** under the written rule: ECE falls and NLL does not rise. No class prediction changes. Brier is the mean per-row sum across the nine classes, not divided by nine. Five training-fold temperatures were fitted for each head, with no full-data fit. The fixed optimisation bounds were log(T) [-5,5]. Rare classes cannot appear in every test fold.

ECE 95% paired row-bootstrap intervals (1,000 draws, seed 39): event model before [0.327133, 0.466921], after [0.150718, 0.276155], after−before [-0.243444, -0.115567]; student E before [0.212828, 0.355467], after [0.112747, 0.225774], after−before [-0.178937, -0.036758]. These intervals condition on the fixed OOF predictions: they do not refit temperatures or include overlapping-training-fold uncertainty. There are only 153 AI-labelled rows, no human check, and very small classes; this is a calibration diagnostic, not a new unseen accuracy test.

Abstention cells show **coverage % / retained accuracy %**. A dash means no retained rows.

| Minimum probability | Event before | Event OOF | Student before | Student OOF |
|---|---:|---:|---:|---:|
| 0.5 | 77.8 / 42.0 | 52.3 / 55.0 | 98.0 / 64.7 | 86.3 / 65.9 |
| 0.6 | 73.9 / 44.2 | 44.4 / 61.8 | 92.2 / 64.5 | 81.0 / 65.3 |
| 0.7 | 60.8 / 51.6 | 0 / — | 89.5 / 65.7 | 60.1 / 71.7 |
| 0.8 | 56.2 / 55.8 | 0 / — | 85.0 / 66.2 | 0 / — |
| 0.9 | 52.3 / 55.0 | 0 / — | 78.4 / 66.7 | 0 / — |

E23 and 8-K alert probabilities: **not stored** in either snapshot. E23 impact is severity and its fire values are hard decisions; the 8-K file stores classes. Neither was converted to probability. Binary Platt scaling/reliability were skipped, with no inference rerun.

E40: **run once, not promoted.** Kaggle T4, `laya==0.4.11`, `convaiinnovations/laya` subfolder `typed-decisions`, revision `7b928d82…`; 530 headlines × 3 questions × 2 forms = 3,180 raw rows, offline after download, 0.04 s per headline (three questions). Threshold set on the 8-K set only: 0.1 (noul, F1 0.811; 137 TP, 54 FP, 10 FN). E23 pairs, design data:

| Arm (noul, primary) | Alerts | True kept | False | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A: gate with E36b rules | 59 | 39 | 20 | 0.661 | 0.848 |
| B: Laya alone | 143 | 34 | 109 | 0.238 | 0.739 |
| C: gate, then Laya filter | 36 | 27 | 9 | 0.750 | 0.587 |

Arm C lost 12 true alerts and removed 11 false ones. Rule 1 (lose at most 1 true alert) **failed**; rules 2, 3 and 4 passed. The secondary `choice` form (same threshold, reported only) lost 5 true alerts and removed 6 false ones. **Decision: not promoted**; the live gate is unchanged, and no unseen-set test is registered for it. Raw probabilities were used with no fitted temperature, as the model card says they are overconfident.

Disclosures: (1) The eight multi-issuer 8-K targets were filled by one fixed rule, the first-named candidate company in the headline text, before any model call. (2) The first Kaggle attempt stopped after the model loaded, in the script's own checkpoint-hash step (the cache held only the typed-decisions subfolder); no probability was written. The step was fixed to look only at that subfolder, the dataset was re-versioned, and the job was run again; the script hash in the manifest is the fixed script, and no question, threshold or rule changed. Both attempts' logs are kept (`data/outputs/e40_kaggle_attempt1_failed.log`, `e40_kaggle_run.log` for the second). Files: `e40_decision_model.json`, `e40_laya_raw.csv`, `e40_laya_raw_manifest.json`.
