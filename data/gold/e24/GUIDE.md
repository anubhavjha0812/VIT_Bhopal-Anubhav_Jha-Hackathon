# E24 event-class annotation guide

Rules: `docs/DECISIONS.md`, E24 (approved v4). This guide only spells out that entry; it does
not change it. Its sha256 is recorded with every label file.

## What you label

Each row is one headline or tweet (`item_id`, `text`, `date`, `companies`). Give it **exactly one**
of the 12 classes below, an ambiguity flag (`Y` or `N`) and a short rationale that quotes or
points to the words in the text that support the label. Judge the text alone: no search, no
prices, no other labels, no model output. Do not infer an event the text does not report.

## The 12 classes

| Class | Use it when the text reports… | Not this |
|---|---|---|
| `Default` | a missed coupon or loan payment, a bankruptcy or Chapter 11 filing, insolvency, a distressed debt restructuring | "could go bankrupt" opinions |
| `Credit downgrade` | a credit rating agency (Moody's, S&P Global Ratings, Fitch, DBRS) cutting a company's or its debt's rating, or a negative outlook or watch | a stock analyst's rating (→ `Analyst rating`) |
| `Covenant breach` | breaking, or asking lenders to waive or amend, loan terms | ordinary new loans |
| `Geopolitical` | wars, military action, sanctions, tariffs and trade disputes, elections and political shocks, diplomacy | an economic indicator that merely mentions a country |
| `Macroeconomic` | central banks and rates, inflation, jobs, GDP, recession, fiscal stimulus, bond and currency markets, oil and commodity prices | one company's results |
| `M&A` | a merger, acquisition, takeover bid, buyout, stake purchase, asset sale, spin-off or IPO | a fund buying shares as an investment holding |
| `Product news` | **narrow:** a product event of the company: a launch, unveiling, release, recall, or a product deal (a deal to supply or buy a product) | other company news (strategy, stock moves, executives' remarks, plants, hiring): → `Other` unless another class fits |
| `Earnings` | quarterly or annual results, revenue, profit or loss figures, beats or misses, guidance, dividends | a preview or an opinion on coming results with no figures (→ `Other`) |
| `Analyst rating` | a stock analyst's rating or price-target change (upgrade, downgrade, initiate, target cut) | a credit agency's action (→ a credit class) |
| `Legal/Regulatory` | lawsuits, regulators (SEC, FTC, EU, …), fines, probes, court rulings, settlements, approvals | |
| `Management change` | a CEO, CFO, chair or other executive appointed, leaving or replaced | |
| `Other` | everything else: price moves, market commentary, opinion, fund holdings, lists, company news that is not a product event | |

## Primary-event rule

The label is the **main news fact** the text reports: what happened, not its likely effect or
its background. A text about a fall in Tesla shares "after the recall" is a stock move, so
`Other`, unless the recall itself is reported as the news.

## Overlaps with a fixed answer (the only precedence rules)

| Both appear | Label |
|---|---|
| a credit agency's action and a stock rating | the credit class |
| earnings and guidance or a dividend | `Earnings` |
| a regulator's decision on a deal | `Legal/Regulatory` if the decision is the news, else `M&A` |
| a recall and a lawsuit | `Legal/Regulatory`; a recall alone → `Product news` |
| an executive leaving amid a probe | `Management change` if the departure is the news |

Any other overlap: apply the primary-event rule. There is no other order of classes.

## Ambiguity and exclusion

- Set `ambiguous = Y` when two classes fit equally well after the rules above, or the text is too
  short or garbled to tell. Still give the best single label and say why in the rationale.
- A person adjudicates every row where the two AI labellers disagree or either marked it
  ambiguous, and 20% of the agreed rows at random (seed 24).
- A row is `excluded` only by the person, and only when it cannot be judged at all: not English,
  empty, or unreadable. Excluded rows are counted and reported; they are not scored.

## Label sources

| `label_source` | Meaning |
|---|---|
| `ai_agreed` | ChatGPT and Gemini gave the same label, neither marked it ambiguous, not in the audit |
| `ai_agreed_human_audited` | as above, in the 20% audit; the person's label is used |
| `human_adjudicated` | the labellers disagreed or one marked it ambiguous; the person's label is used |
| `excluded` | the person could not judge it |

The labels are AI-assisted with human adjudication and audit. They are not independent human gold.
