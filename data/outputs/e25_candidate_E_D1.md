# Candidate E: D1 only

E reuses C's unchanged rules with the v2 student. Production and models/topic are unchanged.
Both models use the same per-row CPU inference method. No calibration, threshold search,
retraining, D2 selection or final inference occurred. Reference labels are AI-derived.

| Reference | Candidate | Fixed 12-class macro-F1 | Supported-class macro-F1 (supplementary) | Accuracy |
|---|---|---:|---:|---:|
| v6 | C | 0.515636 | 0.687515 | 92.16% (141/153) |
| v6 | E | 0.475546 | 0.634062 | 85.62% (131/153) |
| v7 | C | 0.505520 | 0.674026 | 91.50% (140/153) |
| v7 | E | 0.466551 | 0.622068 | 85.62% (131/153) |

| Reference | E fixes vs C | E breaks vs C | Changes one error to another |
|---|---:|---:|---:|
| v6 | 5 | 15 | 1 |
| v7 | 6 | 15 | 0 |

Full precision/recall/support and fixed-12 confusion matrices are in the matching CSVs
and JSON. Every changed row is in changes_vs_C.csv; all 153 predictions are preserved.
D1 is a development diagnostic. Small per-class support and AI reference uncertainty limit
generalization claims; absent credit classes have no measurable D1 recall. This report does
not select E or change production. The existing D2 selector/freeze still handles A–D only;
adding E to that procedure requires the separate authorized selection step.
