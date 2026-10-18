# E29: can R-Drop lift the sentiment teacher from 0.905 to 0.91? (registered before the run)

Idea (arXiv 2106.14448): run each batch twice with different dropout and add a symmetric KL term between the two outputs. Same teacher (DeBERTa-v3-base), same data and split as E7, seeds 42-44, 4 epochs, lr 2e-5.

## Rules, fixed now
- Loss = mean of the two cross-entropies + 1.0 x symmetric KL (alpha 1.0, not tuned).
- Per seed the best epoch is chosen on the TFNS dev split (macro-F1). Then ONE of two options is chosen on dev macro-F1 only: the best single seed, or the average of the three seeds' probabilities.
- The TFNS test (2,388 rows) is scored once, for the chosen option only. Success: accuracy at least 0.91. The E7 teacher scored 0.905 [0.892, 0.916] on this test, which has been viewed; this is a new model, not a new test set, and the result is reported either way.
- No tuning after the test. A failure is logged as a failure.

## Result (one test run, `data/outputs/e29_teacher_rdrop.json`)
Dev macro-F1: seed 42 0.8530, seed 43 0.8523, seed 44 0.8489, 3-seed ensemble 0.8519; chosen on dev: seed 42. TFNS test (2,388): accuracy **0.896** [0.884, 0.908], macro-F1 0.875. The E7 teacher scored 0.905; the 0.91 target is missed, and R-Drop (alpha 1.0) did not help. Not tuned further. Each seed trained about 10 minutes on a Kaggle GPU.
