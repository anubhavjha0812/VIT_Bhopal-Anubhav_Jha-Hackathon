# E24 D1 rule amendment — 2026-10-16

Authorized by the user's request to follow the Claude-authored INSTRUCTIONS.MD.
The original registered E24 section and its hash bfffa0ad37366c8a remain unchanged.

Candidate A, production events.py, the credit gate, model weights, labels and final samples
are unchanged. This amendment applies only to evaluation candidates B, C and D.

The D1 baseline has 153 rows: four Product news, two M&A, five Earnings, 37 Analyst rating
and 95 Other, with small support for the other classes. D1 errors include fund-holding
lists predicted as acquisitions, stock-analyst changes predicted as M&A or Other, and
general company/production/space-flight news predicted as Product news.

## Authorized changes

- B–D route the general investor holding-list template to Other before model predictions.
  Explicit acquisition/deal language prevents this route from masking a corporate deal.
- B–D use the existing is_analyst_rating detector before the model, supplemented by generic
  recommendation-action and roundup syntax. The unchanged-rating-only case remains Other,
  as the guide requires a change. A headline led by a stock move does not take this route.
- Product rules require a product event. Space missions, launch of production and
  administrative/financial releases are excluded. Product supply/sales agreements and
  externally offered beta services follow the existing product-event definition.
- C–D route unquantified earnings previews to Other, including when the raw model says
  Earnings. Actual results and company operating guidance retain their existing class.

Order: unchanged E13 credit rules, holdings, analyst, earnings preview/results (C–D),
product, then the model/registered product-to-Other mapping and D abstention.
New rule decisions are not subject to the model abstention threshold.

Only D1 text and labels are used for development. There is no D1 temperature fitting,
threshold search, model training, production switch, candidate selection or freeze.
For the requested D1 diagnostic, D uses a predeclared 0.50 threshold with no temperature.
This is not a selected threshold; the registered D2 grid is reserved for later selection.

One fixed D1 probability cache is shared by all before/after candidates. The stand-alone
D1 batches changed 12 predictions relative to the previous mixed 300-row run with the
same texts and weights. This batch-context sensitivity is recorded as a limitation;
rule gains are measured against the identical cache, not the earlier mixed-batch score.

The v6 labels are AI-derived and AI-resolved, with no human/expert validation. Product
and M&A recall estimates on four and two D1 positives are not reliable general estimates.
D1 improvements are development diagnostics, not evidence of held-out generalisation.
D2 and final rows are not opened or scored in this work. Frozen files are read only as
opaque bytes by the existing integrity checks.
