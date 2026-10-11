# Experiment register

Every experiment, its outcome and where its stored results are. The rules for each were fixed
before scoring ([DECISIONS.md](DECISIONS.md)); numbers here are copied from the stored files,
which are the source of truth. Updated after every experiment.

| ID | Question | Outcome (headline numbers) | Verdict | Stored results |
|---|---|---|---|---|
| E1 | Sentiment on a public tweet test (TFNS, 2,388) | engine int8 acc 0.877, macro-F1 0.836 [0.818, 0.853]; FinBERT baseline 0.726 / 0.669 | beats baseline | `data/outputs/eval_results.json`, `docs/RESULTS.md` |
| E2 | Event class on a public test (TFN-topic, 4,117, 4 classes) | topic int8 acc 0.951, macro-F1 0.886; keywords 0.740 / 0.505 | beats baseline | same |
| E3 | Hand-labelled gold set for credit classes | sample drawn; not labelled yet | pending | `data/gold/` |
| E4 | Efficiency next to every accuracy | engine sentiment 34 MB, 56 texts/s on CPU (FinBERT 438 MB, 13/s) | reported | `eval_results.json` |
| E5 | Fine-tune bge-small and FinBERT on TFNS / TFN-topic | bge-small sentiment 0.885 acc; FinBERT 0.893 | bge-small chosen (size/speed) | `eval_results.json`, `models/kaggle_v1/` |
| E6 | Learned impact score v1 | lost to volatility-only baseline (Spearman 0.17 vs 0.27–0.34) | fail | `impact_results.json`, `docs/IMPACT.md` |
| E7 | Teacher (DeBERTa) + distilled student | teacher 0.905 acc (target 0.91 missed); student 0.885, not chosen on tweet dev | engine unchanged | `eval_results.json`, `models/kaggle_v2/` |
| E8 | Impact score v2 | fresh 2019 Spearman 0.294 vs volatility 0.145; fresh 2020 −0.041 vs 0.307 | win 2019, fail 2020 | `impact_v2_results.json`, `docs/IMPACT_V2.md` |
| E9 | int8 ONNX engine model | −0.8 pt accuracy, 4× smaller, 1.24× faster | int8 is default | `eval_results.json` |
| E10 | One encoder, two heads | sentiment 0.886 acc, topic 0.950 acc | reported, not engine | `eval_results.json`, `models/kaggle_v3/` |
| E11 | Switch the engine to our models | demo re-scored, 89.7% same event class | switched | `signals.jsonl.gz` |
| E12 | Bank named only as analyst ≠ subject | 126 / 126 tagged links correct in the demo | rule kept | README "Credit alerts" |
| E13 | Credit rules + Module B gate | alerts 41 (1 real) → 4 (3 real) on the demo (design check) | rule frozen | README "Credit alerts" |
| E14 | Module B checks: real alerts, 12 crises | sign right 2 of 3 alerts, size 18–75× too big; analogs no better than naive ($45.7m vs $47.0m) | negative / neutral | `module_b_alerts_check.csv`, `module_b_crisis_backtest.csv` |
| E15 | State-aware market move, monthly (345 months) | model/naive MAE 1.00 (pass ≤ 0.85), DM p 0.933 | fail | `module_b_monthly_backtest.csv` |
| E16 | Nine learned event classes | Product news F1 0.892; old 3 classes within 2 pts; int8 macro-F1 0.841 | pass | `eval_results.json` |
| E17 | Module A over 10 years (2010 – Jun 2020) + tweets 2021–22 | test IC 0.011 [−0.012, 0.034]; Sharpe diff −0.071 [−0.265, 0.103]; relevance-rule precision 0.849 [0.76, 0.91] | fail: no proven edge | `module_a_history.json`, `relevance_results.json`, `data/relevance/` |
| E18 | Engine sentiment on 4 more public sets, leak-checked | 0 rows seen; PhraseBank AllAgree 0.865, 50Agree 0.685, FiQA 0.573, NWGI 0.406; distilled model better on news (paired) | reported | `eval_results.json`, `e18_*.json`, `e18_predictions.csv.gz`, `leak_check.json` |
| E20 | Choose the engine sentiment model on tweet + news dev | dev: engine 0.864 / 0.404, distilled 0.829 / 0.523 → engine stays; SEntFiN 7,903: 0.745 vs 0.765, paired +0.020 [0.013, 0.027] | engine stays (trade-off) | `e20_results.json`, `e20_predictions.csv.gz` |
| E21 | One sentiment model for tweets + news (trained on Kaggle, confirmed on fresh tests) | dev int8: engine 0.864 / 0.734, new model 0.814 / 0.858 → engine stays. Fresh: FinEntity 851 (human) engine 0.765 vs 0.700 (paired −0.065, lower); 2021–22 tweets 251 (AI-assisted, human-reviewed) 0.452 vs 0.591 (+0.139, higher); TFNS tie | engine stays (wins on primary FinEntity); E21 model kept as a lead for retail tweets; cross-source generalisation = future research | `e21_results.json`, `e21_predictions.csv.gz`, `e21_training_*.json`, `e21_sample.json` |
| M1 | Module B sensitivity: do the formulas move the right way? | 7 checks (13 tests) on the real book: no shock → no loss; loss grows with the episode multiple (3 episodes); multiple 0 removes the episode; more notches → more PD/ECL/loss/RWA; stressed PD rises with z; default → stage 3, bond at recovery; reverse stress lands on 8% | all pass | `tests/test_module_b_sensitivity.py` |
| E22 | Interest-rate shock in Module B (parallel UST10Y shift) | 6 checks pass (incl. exact E14 reproduction) → default `rates = moved`. COVID loss $294m → $194m, CET1 4.22% → 7.05% (yields fell 83 bp); GFC 4.37% → 8.91%; +200 bp alone: CET1 13% → 6.18%. Forecast re-runs (reported only): still no edge over naive | checks pass; funding side not modelled | `e22_rates.json`, `e22_crisis_backtest.csv`, `e22_monthly_backtest.csv`, `tests/test_module_b_rates.py` |
| E22b | Does a credit alert alone carry information? (alert-only prediction vs the stock's market-adjusted 20-day return) | history 2010 – Apr 2020: 79 alerts, 40 independent; 55% negative vs base 47.8%, binomial p = 0.23; mean −1.7% [−4.1%, +0.6%]; predicted size vs reaction Spearman +0.05 | FAIL; many history alerts are false (bank-named securitisations, affirmations) | `e22b_alerts.csv`, `e22b_results.json`, `tests/test_module_b_alert_check.py` |
| E23 | Credit-event gold set: is the frozen alert gate right? | gate snapshot hashed; 243 (headline, company) pairs exported blind: all 83 fired (S1) + 60 rule-but-silent (of 107) + 60 hard negatives (of 5,278) + 40 random (of 121,267); E3's 300 headlines excluded. Labels pending (by hand, GUIDE.md) | pending | `data/gold/credit/`, `e23_gate_snapshot.json`, `src/riskengine/eval/credit_gold.py` |
| E24 | Event classes on news the engine was not built on (Product-news mapping, earnings rule, abstention) | pre-registered (v4); final frozen: 200 FNSPID 2017–18 news (earlier-period) + 100 equinxx 2021–22 tweets (later-period, cross-source), deterministic (3 identical runs), read-only backups; E3 split D1 153 / D2 147. No labels yet | pending | `e24_final_sample.json`, `data/gold/e24/`, `private_notes/e24_frozen/` |

## How results are stored

- Every scored run writes a JSON (or CSV) to `data/outputs/` and, where rows are scored,
  per-row predictions, so any comparison can later be redone as a paired test.
- What was scored is frozen where it matters: dataset revisions and file hashes, model file
  hashes and library versions (`e18_manifest.json`).
- Third-party test files are not committed (licences); they are fetched at pinned revisions and
  checked against the stored hashes.
