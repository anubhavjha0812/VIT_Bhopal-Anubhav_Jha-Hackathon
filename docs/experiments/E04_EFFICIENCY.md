# E4: Efficiency

Rules fixed 2026-10-06. Written down before the experiment was run; unchanged here.

## Rules

Next to every accuracy figure: CPU throughput (texts per second, batch 64, after a warm-up
batch), model size on disk, and peak resident memory of the scoring process.

## Result

- **Question:** Efficiency next to every accuracy
- **Outcome:** engine sentiment 34 MB, 56 texts/s on CPU (FinBERT 438 MB, 13/s)
- **Verdict:** reported
- **Stored results:** `eval_results.json`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
