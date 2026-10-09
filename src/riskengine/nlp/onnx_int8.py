"""Export a fine-tuned classifier to ONNX and quantise its weights to int8 (DECISIONS E9).

uv run python -m riskengine.nlp.onnx_int8 models/sentiment [--out models/sentiment-...-int8]

The output folder holds model.onnx (int8), the tokenizer and config.json, and loads through
TextClassifier like any other model folder.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

OPSET = 17


def export_int8(src: Path, out: Path) -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).float().eval()
    model.config.return_dict = False
    sample = tok(["a sample headline"], return_tensors="pt")
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample]
    axes = {k: {0: "batch", 1: "tokens"} for k in names}

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        torch.onnx.export(
            model,
            tuple(sample[k] for k in names),
            str(fp32),
            input_names=names,
            output_names=["logits"],
            dynamic_axes={**axes, "logits": {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
        )
        quantize_dynamic(str(fp32), str(out / "model.onnx"), weight_type=QuantType.QInt8)
    tok.save_pretrained(out)
    model.config.return_dict = True
    model.config.save_pretrained(out)
    if (src / "metrics.json").exists():
        shutil.copy(src / "metrics.json", out / "source_metrics.json")
    return out / "model.onnx"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="fine-tuned model folder")
    p.add_argument("--out", type=Path, help="default: <src resolved>-int8")
    args = p.parse_args()
    src = args.src.resolve()
    out = args.out or src.with_name(src.name + "-int8")
    path = export_int8(src, out)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
