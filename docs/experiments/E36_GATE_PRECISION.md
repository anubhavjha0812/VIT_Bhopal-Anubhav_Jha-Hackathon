# E36: fewer false credit alerts through two category rules in the gate

Rules written on 19 Oct 2026, before any code and before any scored run. They are not changed after the run.

## Question

E23 scored the frozen credit gate on 243 reference pairs: 83 fired alerts, 39 right, 44 false (precision 0.470, recall 0.848). Part of the 44 are mechanical: a rating action on a securitisation deal that carries the bank's name, or an affirmation. Can two generic category rules remove them without removing a true alert?

False alerts by reason in E23: securitization 21, counterparty 9, no own credit event 8, wrong company 3, affirmation 1, upgrade or positive outlook 1, low materiality 1.

## What is tested (the gate only removes alerts, never adds one, never changes an event class)

Rules apply to a row only after the old gate has fired it. A removed row gets `fired = False` and `reason = "blocked: <rule>"`.

1. **Securitization / structured finance** (a rating of a deal or a servicer, not of the company's own debt). The text matches at least one of: a deal id such as `2002-CIBC4`, `2009-RR1` (year, hyphen, a code that contains a letter); `Series` followed by a year; the phrases Commercial Mortgage, Mortgage Trust, Mortgage Finance, Mortgage Securities, Pass-Through, Asset-Backed, Collateralized; the words Tranche, Servicer, Servicing; the acronyms CMBS, RMBS, CLO, ABS (capital letters only). It is not blocked if the text also names a corporate rating: IDR, Issuer Default Rating, senior unsecured, long-term rating.
2. **Affirmation / non-adverse action**, only for the class "Credit downgrade". Blocked if the text has "affirm" (affirms, affirmed), or "upgrade" (upgrades, upgraded), or "outlook" with "stable" or "positive", or "removed from watch/review negative", and has none of the words downgrade, cut, lower(s/ed), slash, withdraw. A text with both an affirmation and a downgrade word stays.

The old gate is reproduced exactly with the flag off (`precision_rules=False`, the default until the decision below is taken).

## Not tested on purpose (known errors that stay)

- Counterparty and other-obligor cases ("a supplier's bankruptcy", "shares rise after another firm's bankruptcy"). Regular expressions do not read who the bankrupt party is, and they would not carry over to new phrasing. This is left for a later model test.
- Wrong-company agency actions on sovereign, municipal or foundation bonds, mortgage servicing ratings that the words above miss, low materiality.
- Recall rules (credit events the detector misses) are a different task and are not touched.

## Honest limits

- The 44 false alerts were seen before these rules were written, so a precision gain on E23 is design-data, in-sample. Every table that uses E23 says so.
- Fresh evidence:
  - a removal audit: run the gate over the stored history and replay signals with the rules on, take the alerts the rules remove that are not among the 243 E23 pairs, sample up to 60 (seed 36), and judge each one blind to the rule (AI-labelled, stated in the result);
  - the credit text set from SEC 8-K filings (287 headlines): the rules must not block credit-true rows;
  - the replay window (demo alerts) is checked for removed alerts.
- No rule contains a headline, a ticker or a company name. The patterns describe a category.
- Clusters of repeated headlines about one deal are possible; the audit counts rows, not independent events.

## Decision rule

Turn the rules on by default only if all hold:

1. E23 design data: at least 20 of the 44 false alerts are removed and none of the 39 true alerts.
2. 8-K set: at most 2 of the 147 credit-true rows (final label in the three credit classes) are blocked (recall change at most 0.02).
3. Replay window: none of the three real alerts is removed.
4. Removal audit: at least 80% of the audited removed alerts are correctly removed.

Otherwise: "no adoption", the flag stays off, and the code and result are kept.

## Result

Run on 19 Oct 2026, once (`data/outputs/e36_gate_precision.json`). Rules were not changed.

E23 design data (seen before the rules; in-sample):

| | alerts | true | false | precision | recall |
|---|---:|---:|---:|---:|---:|
| gate as frozen | 83 | 39 | 44 | 0.470 | 0.848 |
| with the two rules | 58 | 38 | 20 | 0.655 | 0.826 |

- False alerts removed: 24 of 44 (23 by the securitization rule, 1 by the affirmation rule).
- True alerts removed: 1 of 39: "Fitch Affirms AMD's IDR at 'B-'; Outlook Revised to Negative". The affirmation rule read "Affirms" as non-adverse, but a revision of the outlook to negative is adverse news.
- 8-K credit text set: 0 of 147 credit-true rows blocked. Replay window: 0 of the 4 alerts removed.
- Fresh removal audit: not possible. The old gate fires 79 alerts in the history and 4 in the replay window, and all 83 are the E23 alert stratum, so there are no removed alerts outside the seen set.

Decision: **no adoption**. Check 1 ("none of the 39 true alerts removed") failed, and check 4 (audit) could not be run. `precision_rules` stays off by default; the stored numbers are unchanged.

What this shows and does not show: the securitization rule alone would have removed 23 false alerts and no true alert on this seen data (precision 0.65 by simple count), and it blocks nothing in the credit-true 8-K texts. That is a diagnostic, not a result: it was found after seeing the outcome, and there is no unseen alert set left to confirm it. A fresh labelled alert set would be needed before the rule is turned on.

Counterparty and other-obligor cases were not attempted (see above).

## E36b: amendment after the E36 run

The affirmation rule in E36 removed one true alert, an outlook revised to negative. After seeing that, the rule was amended: a text that says the outlook is revised, lowered or changed to negative counts as adverse and is not blocked. This was decided after the E36 result and with the same E23 rows in view, so it is design data, not a test. No unseen alert set exists to confirm it: all 83 alerts of the gate are E23 rows. The securitization rule is unchanged.

The same checks were run once more (`data/outputs/e36b_gate_precision.json`):

| | alerts | true | false | precision | recall |
|---|---:|---:|---:|---:|---:|
| gate without the rules | 83 | 39 | 44 | 0.470 | 0.848 |
| with the rules (E36b) | 59 | 39 | 20 | 0.661 | 0.848 |

- 24 of 44 false alerts removed, none of the 39 true alerts; 0 of 147 credit-true 8-K rows blocked; 0 of 4 replay alerts removed.
- `gate()` now applies the rules by default (`precision_rules=True`). The E23 sampling and the Module B history check still call it with `precision_rules=False`, because their stored results were made without the rules.
- Counterparty and other-obligor alerts (9 in E23) are still not handled.

