"""E28 S, standalone Kaggle script; same blind inputs and NPZ/JSON contract as G.

SetFit is installed only when main is explicitly run on Kaggle. Imports in tests do not
install packages, download encoders or train. Use the same private dataset as G.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss

CLASSES = [
    "M&A",
    "Geopolitical",
    "Macroeconomic",
    "Product news",
    "Earnings",
    "Analyst rating",
    "Legal/Regulatory",
    "Management change",
    "Other",
]
ENCODERS = ["BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"]
SEED, MAX_LENGTH = 28, 64
OUT = Path("/kaggle/working/e28_encoder")


def sample_train(frame):
    if not frame["label"].isin(CLASSES).all() or set(frame["label"]) != set(CLASSES):
        raise ValueError("training must contain exactly nine learned classes")
    return pd.concat(
        [
            frame[frame["label"] == label].sample(
                min(64, int(frame["label"].eq(label).sum())), random_state=SEED
            )
            for label in CLASSES
        ],
        ignore_index=True,
    )


def scale(p, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.log(np.clip(p, 1e-12, 1)) / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def nll(labels, p):
    return float(
        log_loss([CLASSES.index(label) for label in labels], p, labels=list(range(len(CLASSES))))
    )


def main():
    try:
        import setfit
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setfit>=1.1,<2"], check=True)
        import setfit
    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    paths = {}
    for name in ("train", "valid", "eval_texts"):
        hits = sorted(Path("/kaggle/input").rglob(name + ".csv"))
        if len(hits) != 1:
            raise ValueError(f"need exactly one {name}.csv")
        paths[name] = hits[0]
    frames = {
        name: pd.read_csv(path, dtype=str, keep_default_na=False) for name, path in paths.items()
    }
    evaluation, valid = frames["eval_texts"], frames["valid"]
    if set(evaluation) != {"id", "text", "set"} or evaluation.empty:
        raise ValueError("evaluation must be the blind id,text,set sheet")
    if not evaluation["set"].isin(["D1", "D2", "R3"]).all():
        raise ValueError("E24 finals are forbidden")
    if evaluation.duplicated(["set", "id"]).any() or not valid["label"].isin(CLASSES).all():
        raise ValueError("invalid evaluation ids or validation labels")
    train = sample_train(frames["train"])
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    for encoder in ENCODERS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        prefix = "query: " if encoder.startswith("intfloat/") else ""
        model = SetFitModel.from_pretrained(
            encoder, head_params={"C": 1.0, "max_iter": 1000, "random_state": SEED}
        )
        model.model_body.max_seq_length = MAX_LENGTH
        model.model_body.to("cuda" if torch.cuda.is_available() else "cpu")
        dataset = Dataset.from_dict(
            {
                "text": [prefix + text for text in train["text"]],
                "label": [CLASSES.index(label) for label in train["label"]],
            }
        )
        args = TrainingArguments(
            batch_size=(16, 16), num_epochs=(1, 1), num_iterations=20, seed=SEED, report_to="none"
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        start = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - start
        order = [list(model.model_head.classes_).index(i) for i in range(len(CLASSES))]
        start = time.perf_counter()
        valid_p = np.asarray(model.predict_proba([prefix + t for t in valid["text"]]))[:, order]
        p = np.asarray(model.predict_proba([prefix + t for t in evaluation["text"]]))[:, order]
        seconds = time.perf_counter() - start
        temperature = float(
            minimize_scalar(
                lambda t, valid_p=valid_p: nll(valid["label"], scale(valid_p, t)),
                bounds=(0.05, 20),
                method="bounded",
            ).x
        )
        p, calibrated = scale(p, temperature), scale(valid_p, temperature)
        folder = OUT / ("S_" + encoder.split("/")[-1])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "probabilities.npz"
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError("preserve the earlier S export")
        np.savez_compressed(
            path,
            ids=evaluation["id"].to_numpy(dtype=str),
            sets=evaluation["set"].to_numpy(dtype=str),
            labels=np.array(CLASSES),
            text_sha256=np.array(
                [hashlib.sha256(t.encode()).hexdigest() for t in evaluation["text"]]
            ),
            probabilities=p,
        )
        metadata = {
            "variant": "S",
            "encoder": encoder,
            "seed": SEED,
            "class_order": CLASSES,
            "sample_cap_per_class": 64,
            "train_counts": train["label"].value_counts().to_dict(),
            "contrastive_epochs": 1,
            "head_C": 1.0,
            "max_length": MAX_LENGTH,
            "temperature": temperature,
            "calibration_split": "valid",
            "calibrated": True,
            "valid_macro_f1": float(
                f1_score(
                    valid["label"],
                    np.array(CLASSES)[calibrated.argmax(1)],
                    labels=CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "valid_log_loss_calibrated": nll(valid["label"], calibrated),
            "input_sha256": hashes,
            "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "train_seconds": train_seconds,
            "inference_seconds": seconds,
            "seconds_per_1000_texts": seconds * 1000 / (len(valid) + len(evaluation)),
            "machine": str(model.model_body.device),
            "setfit_version": setfit.__version__,
            "scope": "Exploratory development only; no fresh test or E24 finals.",
            "labels": "AI-derived training labels; evaluation input has no labels.",
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
