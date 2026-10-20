# E16: Nine learned event classes

Rules fixed 2026-10-10, before any training or scoring. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** with the E2 table the model learns only Geopolitical, Macroeconomic, M&A and Other.
  "Product launch" is never predicted (the topic data has no such class: 134 launch texts in the
  demo window by keywords, 0 by the model), and earnings, analyst, legal and personnel news all
  fall into Other. The TFN topics already separate these, so the mapping is widened.
- **Classes:** 9 learned + 3 credit rule classes = 12. "Product launch" is renamed
  "Product news" (the TFN topic is any company or product news, not only launches).
- **Mapping** (fixed now, never changed after seeing results; all 20 TFN topics):

  | TFN-topic label | Our class |
  |---|---|
  | Politics | Geopolitical |
  | Fed \| Central Banks, Macro, Treasuries \| Corporate Debt, Energy \| Oil, Gold \| Metals \| Materials, Currencies | Macroeconomic |
  | M&A \| Investments, IPO | M&A |
  | Company \| Product News | Product news |
  | Earnings, Dividend | Earnings |
  | Analyst Update | Analyst rating |
  | Legal \| Regulation | Legal/Regulatory |
  | Personnel Change | Management change |
  | General News \| Opinion, Markets, Stock Commentary, Stock Movement, Financials | Other |

- **Credit classes:** Default, Credit downgrade and Covenant breach stay the E13 rules (frozen)
  and override the model, as now. They are scored later on the gold set.
- **Gold set (E3):** not labelled yet, so its label list grows from 8 to these 12 classes. The
  sample itself is unchanged.
- **Impact severity** (E6 formula; old classes unchanged, fixed before anything is re-scored):
  Product news 4 (as Product launch), Earnings 5, Analyst rating 4 (an opinion on the stock, not
  on credit), Legal/Regulatory 6 (fines and lawsuits reach cash, like Geopolitical and Macro),
  Management change 5. Only credit classes can trigger Module B, so these move impact and
  Module A only.
- **Keyword baseline:** rules for the new classes, written before scoring. Order (first match
  wins): Default, Credit downgrade, Covenant breach, Analyst rating (the E13 stock-rating rule),
  M&A, Legal/Regulatory (lawsuit, sues, SEC, antitrust, probe, fine, court, settlement),
  Management change (CEO, CFO or chief executive with steps down, resigns, appoints or names),
  Earnings (earnings, EPS, beats or misses estimates, quarterly results, dividend, guidance),
  Geopolitical, Macroeconomic, Product news (the old launch rule), else Other.
- **Training:** the E5 recipe (bge-small, max length 64, batch 32, AdamW lr 5e-5, 4 epochs, 10%
  warm-up, seed 42, the E5 dev split) on `topic_train.csv` with the E10 leak filter (texts that
  match an E2 test text, or repeat with different labels, are dropped). Two candidates:
  (A) a 9-class head trained on the mapped labels; (B) a 20-topic head, mapped after prediction.
  The higher dev 9-class macro-F1 wins; within 0.01, (A).
- **int8:** the winner is exported as in E9 and used only if its test accuracy is within 1 point
  of the full-precision model (as E11). Both are reported.
- **Test:** the E2 test set (4,117) under this mapping, scored once per model: keyword baseline,
  the current engine model (E11 int8, 20 topics remapped; it was trained with the 999 leaked
  texts found in E10, so its scores flatter it), the new model and its int8 version. Metrics:
  accuracy, macro-F1 over the 9 classes, per-class precision, recall and F1, with 95% bootstrap
  intervals (1,000 resamples, seed 0) for accuracy, macro-F1 and each class's F1.
- **Pass rule** (on the model the engine would use): (a) Product news F1 ≥ 0.85, and
  (b) Geopolitical, Macroeconomic and M&A each lose at most 2 F1 points against the current
  engine model under this same mapping. The 9-class macro-F1 is reported as it comes; it is not
  comparable to the 4-class E2 figures.
- **Engine:** if E16 passes, the new model replaces the E11 topic model. If it fails, the engine
  keeps the E11 model with this 9-class mapping. Either way the result goes into the README.

## Result

- **Question:** Nine learned event classes
- **Outcome:** Product news F1 0.892; old 3 classes within 2 pts; int8 macro-F1 0.841
- **Verdict:** pass
- **Stored results:** `eval_results.json`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
