# E12: A bank that only gives a stock rating is not the subject

Rules fixed 2026-10-09, before the code. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** a hand check of the 41 high-impact demo texts found JPM and BAC linked to news about
  other companies ("iRobot $IRBT Downgraded by JPMorgan Chase & Co. to Underweight" → JPM).
  These would trigger Module B on the wrong bank.
- **Rule:** JPM or BAC (our two universe companies that publish stock ratings) is tagged
  `analyst` instead of `subject` for a text when all three hold: (1) the text is about a stock rating (downgrade, upgrade,
  rating, price target / PT, overweight, underweight, outperform, underperform, neutral,
  equal-weight, market/sector perform, "to buy/sell/hold", coverage); (2) the bank is named as
  the analyst: at the start followed by a rating action, after "by" / "at", or as "<bank>
  analysts"; (3) the bank's own cashtag is not in the text. Applied at scoring time; ingest is
  unchanged, so the impact training data is unchanged. `analyst` rows are kept and shown (audit
  trail) but Module A, the alert count and Module B use `subject` rows only.
- **Seen before fixing:** the rule was drafted while reading demo texts, and a prototype was run
  on the demo signals (128 links removed; 25 sampled, all correct; 2 Disney links wrongly
  removed, which is why the rule is limited to the two banks). So the demo numbers are a design
  check, not a test.
- **Check:** every `analyst`-tagged link on the demo is read by hand and reported as "X of N correct";
  30 remaining JPM/BAC rating texts are read for misses. Rules first; a learned "which company
  is this news about" model may replace them later and is compared against them.

## Result

- **Question:** Bank named only as analyst ≠ subject
- **Outcome:** 126 / 126 tagged links correct in the demo
- **Verdict:** rule kept
- **Stored results:** README "Credit alerts"

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
