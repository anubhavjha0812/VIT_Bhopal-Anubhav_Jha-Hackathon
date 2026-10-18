"""E27 entity-aware sentiment on Kaggle (docs/E27_ENTITY_SENTIMENT.md). Never reads test.csv.

Input (private dataset): train.csv and dev.csv with columns sno, headline, entity, label
(negative | neutral | positive), conflict, n_entities. The model reads (entity, headline) as a
text pair. Loss = cross-entropy with inverse-sqrt class weights. Seeds 42-44; the best epoch per
seed and the best seed are chosen by dev macro-F1 only. Output: int8 ONNX + summary.json.
"""

import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BACKBONE = "BAAI/bge-small-en-v1.5"
CLASSES = ["negative", "neutral", "positive"]
MAX_LEN, BATCH, LR, EPOCHS, WARMUP = 64, 32, 5e-5, 4, 0.1
SEEDS = [42, 43, 44]
OPSET = 17
OUT = Path("/kaggle/working")


def find(name: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, enc, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(enc["input_ids"]), 256):
        b = {k: v[i : i + 256].to(device) for k, v in enc.items()}
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**b).logits.float().argmax(-1).cpu())
    return torch.cat(out).numpy()


def encode(tok, df):
    entities, headlines = list(df["entity"].astype(str)), list(df["headline"].astype(str))
    return tok(entities, headlines, padding="max_length", truncation=True,
               max_length=MAX_LEN, return_tensors="pt")  # fmt: skip


def macro(y, p) -> float:
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def run(seed, data, device):
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=3, id2label=dict(enumerate(CLASSES)),
        label2id={c: i for i, c in enumerate(CLASSES)},
    ).to(device)  # fmt: skip
    enc, y, cw = data["enc"], data["y"], data["class_w"]
    n = len(y)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * ((n + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    g = torch.Generator().manual_seed(seed)
    best, history, state = None, [], None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            j = order[i : i + BATCH]
            b = {k: v[j].to(device) for k, v in enc.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**b).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, y[j].to(device), weight=cw.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        pv = predict(model, data["dev_enc"], device)
        score = macro(data["dev_y"], pv)
        history.append({"epoch": epoch, "dev_macro_f1": round(score, 4),
                        "dev_acc": round(float((pv == data["dev_y"]).mean()), 4)})  # fmt: skip
        print("seed", seed, history[-1], flush=True)
        if best is None or score > best["dev_macro_f1"]:
            best = history[-1] | {}
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"seed": seed, "best": best, "history": history}


def export_int8(model_dir: Path, out: Path) -> Path:
    """Same export as riskengine.nlp.onnx_int8.export_int8 (ONNX opset 17, int8 weights)."""
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"],
                       check=True)  # fmt: skip
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().eval()
    model.config.return_dict = False
    sample = tok(["a sample headline"], return_tensors="pt")
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample]
    axes = {k: {0: "batch", 1: "tokens"} for k in names}
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        torch.onnx.export(
            model, tuple(sample[k] for k in names), str(fp32), input_names=names,
            output_names=["logits"], dynamic_axes={**axes, "logits": {0: "batch"}},
            opset_version=OPSET, dynamo=False,
        )  # fmt: skip
        quantize_dynamic(str(fp32), str(out / "model.onnx"), weight_type=QuantType.QInt8)
    tok.save_pretrained(out)
    model.config.return_dict = True
    model.config.save_pretrained(out)
    if (model_dir / "metrics.json").exists():
        shutil.copy(model_dir / "metrics.json", out / "source_metrics.json")
    return out / "model.onnx"


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    train, dev = pd.read_csv(find("train.csv")), pd.read_csv(find("dev.csv"))
    lab = {c: i for i, c in enumerate(CLASSES)}
    y = torch.tensor(train["label"].map(lab).to_numpy())
    counts = np.bincount(y.numpy(), minlength=3).astype(np.float64)
    cw = 1 / np.sqrt(np.maximum(counts, 1))
    cw = torch.tensor(cw * len(y) / cw[y.numpy()].sum(), dtype=torch.float)
    data = {"enc": encode(tok, train), "y": y, "class_w": cw, "dev_enc": encode(tok, dev),
            "dev_y": dev["label"].map(lab).to_numpy()}  # fmt: skip
    runs, best_model, best_run = [], None, None
    for seed in SEEDS:
        model, info = run(seed, data, device)
        runs.append(info)
        if best_run is None or info["best"]["dev_macro_f1"] > best_run["best"]["dev_macro_f1"]:
            best_run, best_model = info, model.cpu()
        else:
            del model
        torch.cuda.empty_cache()
    fp32 = OUT / "models" / "e27-entity-bge-small-en-v1.5"
    fp32.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(fp32)
    tok.save_pretrained(fp32)
    (fp32 / "metrics.json").write_text(json.dumps(best_run, indent=2))
    int8 = export_int8(fp32, OUT / "models" / "e27-entity-bge-small-en-v1.5-int8")
    summary = {"chosen": best_run, "runs": runs, "train_rows": len(train), "dev_rows": len(dev),
               "int8_mb": round(int8.stat().st_size / 1e6, 1),
               "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"chosen": best_run["seed"], "best": best_run["best"],
                      "minutes": summary["minutes"]}))  # fmt: skip


if __name__ == "__main__":
    main()
