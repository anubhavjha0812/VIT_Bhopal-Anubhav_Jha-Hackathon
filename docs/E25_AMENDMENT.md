# E25 amendment: candidate E joins the D2 selection (2026-10-17)

Written before D2 is scored with E. `docs/DECISIONS.md` (E24 hash `bfffa0ad37366c8a`) is not edited.

## What is added

- Candidate **E** = C's rules + the E25 student's probabilities (`src/riskengine/eval/e25_candidate.py`).
- Student = bge-small, int8, v2 (`e25_student_summary_v2.json`), trained only on teacher-agreed pool/extra rows + TFN rows. No D1, D2, R3, E23 or final row was used.

## Evidence so far (development only)

- D1 (153 rows): C 0.516, E 0.476. C's rules were built on D1.
- R3 (400 fresh rows, Claude-only blind labels): C 0.436, E 0.469 (+0.033, CI -0.05 to +0.11). Not counted as proof.

## D2 rule

1. Score A, B, C, D (grid) and E on D2 once. Same macro-F1 (fixed 12 classes), same "no class with ≥10 rows loses >0.05 F1 vs A" rule.
2. E is the most complex, so it is chosen **only if** it beats the best of A-D by **≥ 0.02** macro-F1. Otherwise the simplest candidate within 0.01 of the best A-D score is chosen (old rule).
3. D2 reference = v7 (AI-derived). v6 is reported next to it.
4. One shot. No change of rules, threshold or student after D2 is seen. If E is chosen, the freeze pins the student model hash and the live app must load it; if that is not ready by 2026-10-18, C is frozen.

## Not allowed

- No D2/final label used for training, calibration or tuning of E.
- No second D2 run.
