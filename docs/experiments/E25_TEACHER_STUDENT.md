# E25 (draft): teacher–student event model, candidate E

**Status:** draft, not approved, nothing built or run. Written 2026-10-16.
`docs/DECISIONS.md` is hash-checked by the E24 gate and is not edited; this file is separate on purpose.
Before use, DevxAnu approves it and the open questions in section 9 are answered.

## 1. Why

- Candidate A (current engine) scores 0.33 macro-F1 on the AI-derived development reference (v6). It over-predicts Product news and M&A.
- Targeted rules (B–D plus the E24 amendment) lift D1 macro-F1 to about 0.51, but the gain is in-sample, and M&A recall is unresolved (2 positives).
- The engine's topic model was trained on Twitter Financial News Topic (TFN). TFN has no Default, Credit downgrade or Covenant classes, is tweet-style, and does not use the narrow E24 Product news definition.
- Hand labels in our 12 classes exist only for the test sets (300 dev rows, 243 credit rows). They must not be used for training.
- Idea: label many unlabelled real-news headlines with open-weight LLM teachers that follow the E24 guide, then fine-tune the same small model on those labels. The rules stay on top.

## 2. Scope

- Candidate **E** = E24 rules (B–D with the amendment) + a new student model in place of the current topic model + the same optional abstention.
- Candidate A, production code and `models/topic` are not changed.
- The student keeps the engine's size and speed target (bge-small, int8 ONNX, about 34 MB).
- Credit classes stay with the E13 rules. The student predicts the 9 learned classes (`LEARNED_CLASSES` in `src/riskengine/nlp/events.py`).

## 3. Training pool (no test leakage)

- Built by `src/riskengine/eval/e25_pool.py` (`--dry-run`, `--build`). 
Output `data/cache/e25/pool.csv.gz` (git-ignored, FNSPID licence) and the manifest `data/outputs/e25_pool_manifest.json`.

- Source: 
the E17 history set (`data/cache/history_signals.csv.gz`), 
with the same frame rule as the E24 final news: E17 relevance kept, role subject, unique text. 
This matches how the test headlines were chosen.

- **Time split:** 
titles published 2010-01-01 to 2016-12-31 only (the E17 history starts in 2010). 
The final news is 2017–18 and the dev set is 2019 – Jun 2020, so training text is always older than the test text.


- **Exclusion:** 
exact normalised match or cosine ≥ 0.9 against the E24 exclusion corpus (`riskengine.eval.e24.exclusion_corpus()`) plus the final news and tweet texts: dev 300, E23 sample, E22b alerts, E17 relevance samples, E21 tweets, TFN topic/sentiment train and valid, FinEntity. 
The filter is repeated on the final pool until a pass drops nothing (the cosine depends on the fitted TF-IDF, so a border case can flip), then a guard fails the build if any match is left.
- **Cap:** at most 3,000 headlines per company (seed 25), so AAPL, GOOGL and others cannot dominate.
- **Result of the first build (2026-10-16):** 77,854 unique headlines in the frame, 316 removed as test or excluded-set matches (E23 193, E22b 77, E17 relevance 120, E3 5, TFN 2; sets overlap), 40,453 kept. Companies: 1,151 (AMD) to 3,000. Years: 4,355 to 7,240 per year. None of the final-news texts were in the frame, as the time split intends. Pool sha256 `fb9229c9…58f0a`.
- Leak check done earlier: every dev headline (300 of 300) and 211 of 243 E23 headlines appear in the full FNSPID file, so the filter is required, not optional.

## 4. Teachers

- Two open-weight LLMs from **different families**, run on Kaggle (no laptop compute).
- **Decided (DevxAnu, 2026-10-16):** teacher 1 = Qwen3-8B; teacher 2 = **Mistral-Nemo-12B** (Apache 2.0, different family, run in 4-bit on Kaggle if needed). Backup for teacher 2: Llama-3.1-8B-Instruct (its licence needs attribution; we disclose the teacher anyway). The pair is confirmed by agreement with the v6 reference on D1 only. Phi-4-mini (too small, over-says yes in E17) and OLMo2-7B (said yes to 0% of rows in E17) are not used.
- Check each model's licence on Hugging Face before running; it has not been read for this draft.
- Prompt = the E24 guide (`gold/e24/GUIDE.md`) + class definitions + a few examples. Examples are written by hand from outside every test set.
- Output: a probability over the 12 classes, read from the model's next-token scores for the 12 class names (not free text), plus a one-line reason for audit.
- Prompt tuning uses D1 only: agreement with the v6 D1 reference. At most 2 prompt revisions, each logged. D2, the final sets and E23 are not used.
- Teachers never see test labels or engine predictions.

### Prompt revision log

- **v1 (2026-10-16):** E24 class definitions + 16 invented examples. Dry run on 500 pool rows (Qwen3-8B). A blind spot-check of 100 of those rows by Claude (labels in `private_notes/.../e25_checks/`, a diagnostic, not training data) found 74% agreement (kappa 0.61). Qwen over-called Product news (143 of 500 rows; on the 100-row check 15 of 27 Product news calls were Other, mostly the words launch / release / open used for initiatives, studies, units, exhibits) and called 6 rows Default that report no default (loan losses, "obligations", a plant losing power). Top-class probabilities were near one-hot (mean 0.97), so soft labels carry little extra information.
- **v2 (2026-10-16), revision 1 of at most 2:** tightened G (Product news) and A (Default) and added 6 invented counter-examples. Written from the pool spot-check only; D1 was not used for this revision. D1 agreement with the v6/v7 reference is the first test of v2.

- **v3 (2026-10-16), revision 2 of 2:** from pool rows only (116 Product news, 23 M&A, 19 Management change calls by Qwen v1 on rows outside the spot-check): partnerships, outages, adverts, third-party certifications, investments, bug reports no longer Product news; supply, exploration and partnership deals and merger hearings no longer M&A; pay, remarks and unit renames no longer Management change. D1 result (batch 4): agreement 82.4% vs 81.7% for v2, accuracy on kept rows 84.9% / 85.7% (v6 / v7) vs 84.8% / 85.6%; Qwen macro-F1 0.452 / 0.460 vs 0.436 / 0.443, Mistral 0.454 / 0.490 vs 0.498 / 0.534. No clear gain, so the pool run uses v2 (no more revisions allowed).

### Second-teacher speed check (dry run)

The 500-row dry run runs Qwen3-8B and Mistral-Nemo-12B (the default pair) on the same pool rows. Each teacher writes `load_seconds`, `score_seconds`, `rows_per_second`, `projected_pool_hours` (for the 40,453-row pool) and peak GPU memory to its meta file; a teacher that fails (out of memory, download) writes a `_FAILED.json` and the other still runs.

Rule (fixed before the run): keep **Mistral-Nemo-12B** as teacher 2 unless it fails to load, or its projected pool time is more than 8 hours, or it later agrees clearly worse with the v6 D1 reference than the backup would. Only then run the backup, Llama-3.1-8B-Instruct, with `--teachers llama31_8b` (it is gated: accept its licence on Hugging Face and give Kaggle an `HF_TOKEN` secret first). Speed is measured on scoring only, not on model loading.

## 5. Training labels

- Keep a headline only if **both teachers pick the same top class**. Disagreements are dropped and counted per class.
- Soft label = mean of the two teachers' probabilities. Hard label = the agreed class.
- Report: rows kept, keep rate by class, class balance, and how many rows are Other.
- If a class has fewer than 200 kept rows, report it as under-supported. No oversampling with test rows.
- Optional phase 2 (credit classes): SEC 8-K item codes as documentary labels for Default and Covenant breach (for example Item 1.03 bankruptcy and Item 2.04 triggering events). Verify the item meanings against the SEC form instructions before use. Not part of the first run.

## 6. Student

- Base `BAAI/bge-small-en-v1.5`, E21 recipe: max length 64, batch 32, AdamW lr 5e-5, 4 epochs, 10% warm-up, seeds 42, 43, 44, trained on Kaggle.
- Loss: KL divergence to the soft labels + cross-entropy to the hard label.
- Early stopping on a validation slice taken from the **training pool** (random 5%, same dedupe rules), never on D1, D2 or finals.
- Export: `riskengine.nlp.onnx_int8.export_int8` (as in E21), then the E24 per-row inference method (see the batch determinism fix) so outputs do not depend on batch makeup.
- The best seed is chosen on the training validation slice only.

## 7. Evaluation and choice

- Score E on **D1** only to confirm the rules still work with the student. No rule retuning beyond the two D1 iterations already used.
- Compare A, B, C, D and E on **D2** once, with the registered choice rule: highest 12-class macro-F1, no class with ≥ 10 D2 rows may lose more than 0.05 F1 against A, within 0.01 means the simpler candidate. E is the most complex candidate, so it wins only by a clear margin.
- Report two macro-F1 numbers: fixed 12-class (primary) and over supported classes only (supplementary, as in TODO L3).
- E must be added to the registered candidate list by a written amendment (new file, dated) **before** D2 is scored.
- Freeze the chosen candidate (hashes), then label and score the final holdouts once. Training on D1, D2, E23 or the finals is never allowed.

## 8. Policy and disclosure

- E21 said training used human labels only. E25 changes that for this student only: its training labels come from open-weight LLMs. State this plainly in the report and in the model card.
- Test references stay AI-derived (ChatGPT + Claude, v6) and are reported as such. Teachers are a different family from the test labellers where possible, to reduce shared errors. Report which models were used.
- Check each teacher's licence allows training a derived model before running it.
- NWGI stays out of training.
- No claim of expert or human validation anywhere.

## 9. Open questions for DevxAnu

1. ~~Second teacher~~ decided: Mistral-Nemo-12B (backup Llama-3.1-8B-Instruct).
2. Pool size and run time budget on Kaggle (estimate after a 500-row dry run).
3. Deadline for the candidate freeze, so E is dropped if it is late.
4. Include the 8-K phase 2 for Default and Covenant before the freeze, or after?
5. Approve the E21 policy change in section 8.

## 10. Files to create (when approved)

- `src/riskengine/eval/e25_pool.py`: build the deduplicated time-split pool and its manifest (hashes).
- `notebooks/e25_teacher_label.py`: Kaggle teacher labelling (both models, D1 agreement report).
- `notebooks/e25_train_student.py`: Kaggle student training and int8 export.
- `src/riskengine/eval/e25_candidate.py`: candidate E wired into `eval/e24_dev.py`.
- `tests/test_e25_*.py`: pool exclusion, agreement filter, and the no-test-training guard.
- New outputs named `e25_*` in `data/outputs/`. Log in `STEPS.md` and `docs/experiments/EXPERIMENTS.md`.

## 11. Stop rules

- If teacher–reference agreement on D1 is poor after 2 prompt revisions, stop and keep the hybrid.
- If E does not clearly beat the hybrid on D2, freeze the hybrid.
- If the freeze deadline is near, freeze the hybrid; E can continue as later work.
