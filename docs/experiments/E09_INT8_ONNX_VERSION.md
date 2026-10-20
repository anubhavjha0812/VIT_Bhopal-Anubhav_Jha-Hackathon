# E9: int8 ONNX version

Rules fixed 2026-10-09, before any int8 model is scored. Written down before the experiment was run; unchanged here.

## Rules

- **Which model:** the sentiment model chosen as engine default under E7 (or the E5 one if E7
  does not replace it).
- **How:** export to ONNX (opset 17), then onnxruntime dynamic quantisation, int8 weights
  (`nlp/onnx_int8.py`). No calibration data, no further training.
- **Test:** scored once on E1 with the E7 metrics and E4 efficiency. Size = the `model.onnx` file
  (for torch models: weights in memory, which match their fp32 file size).
- **Engine default:** int8 replaces the torch model only if its E1 accuracy is within 1 point of
  the torch model's. This is the one choice made on test scores; both are reported.

## Result

- **Question:** int8 ONNX engine model
- **Outcome:** −0.8 pt accuracy, 4× smaller, 1.24× faster
- **Verdict:** int8 is default
- **Stored results:** `eval_results.json`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
