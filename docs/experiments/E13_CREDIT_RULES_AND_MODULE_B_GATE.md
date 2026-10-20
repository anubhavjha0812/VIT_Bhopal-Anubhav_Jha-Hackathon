# E13: Credit rules and the Module B trigger gate

Rules fixed 2026-10-09, before the code. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** the credit keyword rules fire on stock-analyst rating changes and on "default search
  engine", and miss "cuts … to junk". Of 41 demo texts with impact > 7, 1 is a real credit
  event; two more real ones (Moody's–Boeing, S&P–Disney) scored 7 and were missed.
- **Analyst rating:** a rating action (downgrade, upgrade, initiate, reiterate, maintain, cut,
  raise, lower) with a stock-rating word (buy, sell, hold, neutral, overweight, underweight,
  outperform, underperform, equal-weight, market/sector perform, price target / PT) and no
  agency or credit word is never a credit event. The class stays the topic model's (8 classes
  unchanged); the text is flagged `analyst_rating`.
- **Credit downgrade:** a cut word (downgrade, cut, lower, slash) with a rating agency (Moody's,
  Fitch, DBRS, Standard & Poor's, S&P but not the S&P index) or a credit word (junk, credit
  rating, IDR, investment grade, high yield, bond, debt rating, CreditWatch); or an agency with
  "outlook … negative".
- **Default:** bankrupt, Chapter 11, insolvent, missed coupon/interest/debt payment, "defaulted",
  "defaults on". Plain "default" only with a debt word in the same text (debt, loan, bond,
  coupon, payment, creditor, lender, obligation).
- **Covenant breach:** the existing rule. These rules replace the keyword credit override in the
  engine only; the keyword baseline (E2) and the impact training data are unchanged.
- **Impact:** +1 when a rating agency is named (an official rating action, like the existing +1
  for news). Chosen after seeing that the two missed real events scored 7.
- **E12 addition:** "<bank> analysts / strategists / economists" makes the bank an analyst even
  without rating words ("Bank of America analysts see oil at $40").
- **Credit gate:** a (text, company) row triggers Module B only if (1) the company is
  the subject (E12); (2) it is the obligor: named in the text and, if the text has cashtags,
  the first cashtag is its own; (3) the event is a credit class; (4) impact > 7 (case study);
  (5) a rating agency is named, or a different text on the same company and credit class
  arrived in the previous 24 hours.
- **Check:** the demo numbers (naive vs gated alerts, every gated alert read by hand, the 3 real
  events above as "should fire") are a design check, since those texts were read while writing
  this rule. The test is the gold set (E3), labelled blind: per-class precision and recall for
  the credit classes, keywords vs these rules, scored once. A learned credit model may be
  compared against these rules later.
- **Amended after the first demo check (2026-10-09, before the gold test):** the check showed
  "Google pays Apple to be the default search engine … such payments …" as Default and missed
  "Moody’s" written with a curly apostrophe. So: plain "default" needs a debt word within three
  words of it; hedged bankruptcy ("won't go bankrupt", "may file for bankruptcy", "brink of
  bankruptcy") is not Default, as the gold labelling guide already says; agency names accept
  either apostrophe. The demo is re-scored once more after this.

## Result

- **Question:** Credit rules + Module B gate
- **Outcome:** alerts 41 (1 real) → 4 (3 real) on the demo (design check)
- **Verdict:** rule frozen
- **Stored results:** README "Credit alerts"

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
