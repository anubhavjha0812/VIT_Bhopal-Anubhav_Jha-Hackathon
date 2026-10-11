# Credit-event gold set: how to label (DECISIONS E23)

Open `credit_to_label.csv`. Each row is one headline **and one company** (`target_company`).
Judge the headline only for that company. Read the headline alone: no search, no prices, no
model output. Save your work as `credit_labelled.csv` with the same rows and columns.

## Columns you fill

| Column | Values | Meaning |
|---|---|---|
| `target_company_match_correct` | `Y` / `N` | Is the headline about this company (or its own debt)? `N` if the company is only named: a fund, trust, deal, client, rival, bank acting as broker or lender to someone else. |
| `credit_event_relevant` | `Y` / `N` | Does the headline report something that changes **this company's** credit risk: its ability or willingness to pay its debt, or an agency's view of it? A headline can be `N` here even when it is about the company (e.g. an earnings beat). |
| `event_subtype` | one of the list below | What kind of credit news it is. Use `other` when it is not credit news at all. |
| `materiality` | `low` / `medium` / `high` | How much it should move a lender's view of **this company**, for credit-relevant rows (leave blank otherwise). |
| `existing_gate_should_fire` | `Y` / `N` | Should a credit alert for **this company** fire? Rule below. |
| `annotator_rationale` | free text | One short line, especially when unsure. |
| `adjudication_status` | `unreviewed` / `ai_labelled` / `reviewed` / `adjudicated` / `excluded` | `ai_labelled` if an AI tool made the labels; `reviewed` once a person labelled or checked the row; `adjudicated` after a second look settles a doubt; `excluded` if the headline cannot be judged. `ai_labelled`, `reviewed` and `adjudicated` rows are scored (E23 amendment). |

## Event subtypes

| Subtype | Use it when the headline reports… |
|---|---|
| `default` | a missed coupon or loan payment, bankruptcy or Chapter 11 filing, distressed exchange, insolvency |
| `downgrade` | a credit rating agency (Moody's, S&P Global Ratings, Fitch, DBRS) cutting the company's or its own debt's rating, a negative outlook, or CreditWatch/review for downgrade |
| `upgrade` | an agency raising the rating or outlook |
| `covenant` | a covenant breach, waiver or amendment request to lenders |
| `liquidity` | a cash squeeze: drawing down credit lines, emergency funding, going-concern doubt |
| `refinancing` | raising or rolling over debt (bond sale, new loan, extension), distressed or not |
| `affirmation` | an agency confirming an unchanged rating ("Fitch Affirms …") |
| `counterparty` | the company is a lender, insurer, trustee, servicer or counterparty to **someone else's** credit event |
| `securitization` | a deal, trust or class of notes that only carries the company's name (e.g. "J.P. Morgan 2006-FL2", CMBS classes) |
| `other` | anything that is not credit news for this company (stock ratings, earnings, products, markets) |

## Materiality (credit-relevant rows only)

- `high`: default, bankruptcy, a cut to or below junk, a multi-notch cut, a covenant breach.
- `medium`: a one-notch cut, a negative outlook or watch, a liquidity strain, distressed refinancing.
- `low`: routine refinancing, affirmations, upgrades, small or one-off items.

## Should the gate fire? (`existing_gate_should_fire`)

`Y` only if **all** hold: the target match is correct (`Y`); the headline is credit-relevant
(`Y`); the subtype is `default`, `downgrade`, `covenant` or `liquidity` (bad credit news); and
materiality is `medium` or `high`. Otherwise `N`. Judge what *should* happen, not what you think
the system did (you are not shown it).

## Hard cases

| Headline (shape) | Labels |
|---|---|
| "Fitch Downgrades 3 Classes of JPMorgan 2006-FL2" for JPM | match `N`, relevant `N`, `securitization`, fire `N` |
| "Fitch Affirms Bank of America's IDR at 'A'; Outlook Stable" for BAC | match `Y`, relevant `Y`, `affirmation`, `low`, fire `N` |
| "Fitch Affirms X at 'BBB'; Outlook Revised to Negative" | match `Y`, relevant `Y`, `downgrade` (the outlook cut is the news), `medium`, fire `Y` |
| "Bank of America Says MBIA Defaulted on …" for BAC | match `N` for BAC's own credit, relevant `N`, `counterparty`, fire `N` |
| "S&P Raises Netflix to BB- from B+" | match `Y`, relevant `Y`, `upgrade`, `low`, fire `N` |
| "Moody's cuts Boeing to Baa2" for BA | match `Y`, relevant `Y`, `downgrade`, `medium`, fire `Y` |
| "Goldman downgrades Disney to Neutral" for DIS | match `Y`, relevant `N` (a stock rating), `other`, fire `N` |
| "JPMorgan downgrades Ford to junk" for JPM | match `N` (JPM is the broker), relevant `N`, `other`, fire `N` |
| "Could Tesla go bankrupt?" for TSLA | match `Y`, relevant `N` (opinion), `other`, fire `N`; note it |
| "Apple sells $7bn of bonds" for AAPL | match `Y`, relevant `Y`, `refinancing`, `low`, fire `N` |
| "Boeing draws down entire $13.8bn credit line" for BA | match `Y`, relevant `Y`, `liquidity`, `high`, fire `Y` |
| "Moody's cuts India's rating … $BAC" for BAC | match `N`, relevant `N`, `other`, fire `N` |

A genuine credit event changes **this company's** ability to pay or an agency's view of it. A
headline that only mentions a bank, a rating, debt or "default" in another sense is not one.

## Adjudication

1. Label every row (`reviewed`). Mark doubtful ones in `annotator_rationale` with `?`.
2. Go back over the `?` rows once, after a break, without looking at any system output; settle
   each and set `adjudicated`, or `excluded` if it still cannot be judged.
3. If a second person is available, they label a random 50 rows blind; agreement (Cohen's
   kappa) is reported. Disagreements are settled by discussion and marked `adjudicated`.
4. Never change a label after seeing the evaluation; if a real mistake is found later, fix it,
   say so in the report, and give both numbers.
5. AI labels count as labels (E23 amendment, the author's decision): mark such rows
   `ai_labelled`, or `reviewed` if a person checked them, and write in `annotator_rationale` or
   the report which tools labelled them. Results always say who labelled the rows.
