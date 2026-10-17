# Frozen E23 gate — v6 AI reference

Report only: stored frozen fired decisions versus existing_gate_should_fire on 243 headline–company pairs. AI-derived labels, no human/expert validation. Gate source and snapshot were not changed.

| Stratum | Rows | TP | False alerts | Missed alerts | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall sample | 243 | 39 | 44 | 7 | 153 | 47.0% | 84.8% |
| S1 | 83 | 39 | 44 | 0 | 0 | 47.0% | 100.0% |
| S2 | 60 | 0 | 0 | 7 | 53 | N/A | 0.0% |
| S3 | 60 | 0 | 0 | 0 | 60 | N/A | N/A |
| S4 | 40 | 0 | 0 | 0 | 40 | N/A | N/A |

Overall exact sample precision is 39/83 (47.0%) and recall is 39/46 (84.8%): 44 false alerts and seven missed alerts. S1 contains all fired examples; S2–S4 contain silent examples, so their conditional precision/recall should not be interpreted as separate deployment performance.

**False alerts by primary reason**

| Reason | Count |
| --- | ---: |
| securitization | 21 |
| affirmation | 1 |
| counterparty | 9 |
| stock_rating | 0 |
| wrong_company | 3 |
| no_own_credit_event | 8 |
| non_adverse_or_low_materiality | 1 |
| upgrade_or_positive_outlook | 1 |

All 44 false alerts are listed with text, stratum, gate reason and label rationale in e24_e23_gate_v6_false_alerts.csv. All seven missed alerts are in e24_e23_gate_v6_missed_alerts.csv. The complete 243-row comparison is e24_e23_gate_v6_rows.csv.

The sample deliberately oversamples gate/rule positives. Inverse sampling-fraction point estimates give population precision 47.0% and recall 75.8%. These are estimates, without confidence intervals. Zero sampled positives in S3/S4 do not establish zero missed events in those large strata.
All five E23 snapshot file hashes pass. No gate, event classifier, credit label or snapshot was edited. No final/D2 rows were read.
