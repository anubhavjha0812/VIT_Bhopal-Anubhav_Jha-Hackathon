# E23: Credit-event gold set for the frozen alert gate

Rules fixed 2026-10-11, before any sample is drawn or label seen. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** E22b failed, and reading its alerts showed many that are not credit events for the
  named company (bank-named securitisations, affirmations, a counterparty, an upgrade). The gate
  (E12, E13) has never been measured against human labels. E23 measures it; it extends E3 (whose
  300 class-labelling headlines stay as they are and are excluded from this sample).
- **Gate snapshot:** before export, sha256 of `src/riskengine/gate.py`, `nlp/events.py`,
  `nlp/impact.py`, `ingest/linking.py`, `ingest/relevance.py` and `impact_trigger` are written
  to `data/outputs/e23_gate_snapshot.json`. The gate is not changed before it is evaluated.
- **Unit:** a (headline, company) pair, as the gate sees it.
- **Pool:** the E22b universe: E17 history headlines 1 Jan 2010 – 8 Apr 2020 kept by the E17
  relevance rules with role `subject`, scored as in E22b (E13 credit rules, impact formula with
  the stored sentiment, source news, gate with trigger 7), plus the replay-window signals
  (9 Apr – 10 Jun 2020, news and tweets) through the same gate. Pairs whose headline is one of
  the 300 E3 headlines are removed.
- **Strata (seed 23):**
  - **S1, gate fired:** every fired pair (a census).
  - **S2, credit rule but gate silent:** 60 at random from pairs the E13 rules call a credit
    class where the gate did not fire.
  - **S3, hard negatives:** 60 at random from pairs the rules do *not* call credit whose
    headline contains a credit-flavoured word or pattern: affirm, upgrade, outlook, rating,
    Moody's, Fitch, S&P, DBRS, securiti, CMBS, mortgage trust, "Classes", a deal code such as
    `2006-FL2`, bankrupt, default, liquidity, refinanc, covenant, waiver, junk, bond, debt,
    lender, counterparty.
  - **S4, random:** 40 at random from all other pairs (to find credit events the rules miss).
  Each stratum's population size is stored, so recall can be weighted back to the pool.
- **Blind file:** `data/gold/credit/credit_to_label.csv` shows headline id, headline, date and
  target company only; no event class, impact, gate result or stratum. System outputs sit in
  `data/gold/credit/sample_meta.csv`. AI suggestions, if ever added, go in a separate
  `ai_suggestion` column and never count as labels; rows stay `unreviewed` until a person
  reviews them.
- **Labels:** defined in `data/gold/credit/GUIDE.md` (written before labelling): target match
  correct, credit-event relevant, subtype (default, downgrade, upgrade, covenant, liquidity,
  refinancing, affirmation, counterparty, securitization, other), materiality, whether the gate
  should fire, rationale, adjudication status. Only rows with status `reviewed` or
  `adjudicated` are scored; `unreviewed` and `excluded` are counted.
- **Metrics (once labels are reviewed, scored once):**
  1. **Gate:** precision on S1 with a Wilson 95% interval; recall weighted by stratum
     population sizes, with a stratified bootstrap interval (1,000 resamples, seed 0).
  2. **Credit-relevance rules (E13):** precision and weighted recall of "rules say credit"
     against "credit-event relevant".
  3. **Target matching:** precision of the linked (headline, company) pairs, overall and by
     stratum. Recall of matching is **not measurable** here: only linked pairs are sampled.
  4. Subtype confusion matrix: rule class (default / downgrade / covenant / none) vs human
     subtype; the gate's false-positive share by human subtype.
  5. Splits by source (history news; replay news; replay tweets) and period (2010–2016;
     2017 – 8 Apr 2020; replay window), shown only with n ≥ 10, all with n.
- **No pass rule:** E23 measures; it does not decide. Any change to the gate is a later,
  separately pre-registered experiment, tuned on a part of the data that E23 does not use for
  its final numbers. A small sample is reported as such.
- **Amendment (2026-10-11, after the export, before any label exists; the author's decision):**
  AI-made labels count as the ground truth. Rows labelled by AI carry `adjudication_status =
  ai_labelled` (or `reviewed` if the author then checked them) and are scored. Every E23 result
  states who labelled it (the models' names, and whether a person reviewed the rows), as E21
  did. The AI tools only label this test set; none is part of the engine.

## Result

- **Question:** Is the frozen credit-alert gate right?
- **Outcome:** AI reference 243 pairs: TP 39, FP 44, FN 7, TN 153; sample precision 0.470, recall 0.848; population-weighted recall point estimate 0.758
- **Verdict:** report only; gate unchanged
- **Stored results:** e24_e23_gate_v6.json/.md, e24_e23_gate_v6_false_alerts.csv

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
