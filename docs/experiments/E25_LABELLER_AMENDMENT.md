# E25 amendment: Claude is labeller B for the final sets (2026-10-17)

Written before any final label is merged or scored. `docs/DECISIONS.md` (E24 hash `bfffa0ad37366c8a`) is not edited.

- E24 names ChatGPT and Gemini as the two AI labellers. Gemini labels are not available. D1 and D2 were already labelled by Claude and ChatGPT, so labeller B = Claude is used for the final sets too.
- ChatGPT labelled final_news (200) and final_tweets (100) through a Codex CLI agent session (blind prompt `private_notes/codex_prompts/6_final_labels_chatgpt.md`). Claude labelled the same rows afterwards, from the text and date only, **without reading ChatGPT's labels**. The raw files are `data/gold/e24/raw/final_*_{chatgpt,claude}.csv`; records are `data/gold/e24/records/{chatgpt,claude}_final_*.json` (hashes of prompt, guide, input and raw output).
- Code change: the final gate accepts the second record from `gemini` or `claude` (`LABELLER_B` in `e24_gate.py`). Nothing else in the gate, the rules, the candidate, the threshold, the hashes or the metrics changes. The gate file hash recorded at freeze therefore differs; the candidate, rules file, model and sample hashes are untouched.
- Merge, adjudication and audit follow E24: every disagreement and every row either labeller marked ambiguous is adjudicated by a person, plus 20% of the agreed rows (seed 24). The merged columns named `gemini_*` in `e24_labels.py` hold labeller B (Claude).
- Reported plainly: both labellers are AI, the second labeller is from the same family as the system builder's assistant, and the labels are AI-assisted with human adjudication, not independent human gold.

## Addendum (same day): adjudication and audit were done by an AI, not a person

- DevxAnu declined to adjudicate by hand and told Claude to do it. So the 145 rows a person should label (final_news 102, final_tweets 43: ambiguous, disagreement, 20% audit) were labelled by Claude from the text with the guide, **without looking at either AI's label** (only text and reason were shown). Files: `private_notes/annotation_uploads/task1/final_adjudication/final_*_ai_adjudication.csv`.
- New label sources so nothing is called human: `ai_adjudicated` (replaces `human_adjudicated`) and `ai_agreed_ai_audited` (replaces `ai_agreed_human_audited`). `merge_labels(..., adjudicator="ai")`, `LABEL_SOURCES` and the "touched rows" sensitivity in `e24_gate.py` accept them.
- Result: every final label is AI-made: 3 AI passes on the contested rows, 2 on the rest. There is **no human check** of the final sets; report them as AI-derived. The "human correction rate" of the audit is not a human measure here.
- Adjudicator and labeller B are both Claude; ChatGPT is the only labeller outside that family. Said plainly in the report.
