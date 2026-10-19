"""E40: self-contained Kaggle GPU job, never run this model on the laptop.

Inputs: original 8-K/E23 snapshots and a separately confirmed 8-K target map.
Only ID, headline and target company enter a question; labels/pred_C never do.
See E40_README.md for upload/run/download steps. No threshold is fitted here.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

os.environ["USE_TF"] = "0"

LAYA_VERSION = "0.4.11"
MODEL_ID = "convaiinnovations/laya"
SUBFOLDER = "typed-decisions"
QUESTIONS = {
    "Q1": "Does this headline report a credit event (rating action, default, missed payment, "
    "covenant breach, restructuring, bankruptcy or similar) of a company?",
    "Q2": "Is the credit event reported one of the company <TICKER NAME> itself, not of a "
    "counterparty, a customer, a supplier or a debt issue it does not own?",
    "Q3": "Is this headline about a securitisation, trust, note or structured-finance deal "
    "rather than the company?",
}
FORMS = ("noul", "choice")
RAW_COLUMNS = (
    "dataset",
    "item_id",
    "target_company",
    "text_sha256",
    "form",
    "question",
    "p_yes",
    "p_no",
    "call_seconds",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_digest():
    return text_digest(json.dumps(QUESTIONS, sort_keys=True))


def csv_rows(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {required}")
        return list(reader)


def load_inputs(credit8k, e23, targets):
    eight = csv_rows(credit8k, ("item_id", "text"))
    pairs = csv_rows(e23, ("headline_id", "headline", "target_company"))
    mapping_rows = csv_rows(targets, ("item_id", "target_company"))
    mapping = {r["item_id"]: r["target_company"].strip() for r in mapping_rows}
    eight_ids = {r["item_id"] for r in eight}
    if len(mapping) != len(mapping_rows) or set(mapping) != eight_ids:
        raise ValueError("Target map must cover every 8-K ID exactly once")
    missing = sorted(k for k, v in mapping.items() if not v)
    if missing:
        raise ValueError(f"Confirm missing 8-K targets before any model load: {missing}")
    result = [
        {
            "dataset": "8K",
            "item_id": r["item_id"],
            "text": r["text"],
            "target_company": mapping[r["item_id"]],
        }
        for r in eight
    ] + [
        {
            "dataset": "E23",
            "item_id": r["headline_id"],
            "text": r["headline"],
            "target_company": r["target_company"].strip(),
        }
        for r in pairs
    ]
    if len(eight) != 287 or len(pairs) != 243:
        raise ValueError("Fixed benchmark requires 287 8-K rows and 243 E23 pairs")
    keys = [(r["dataset"], r["item_id"]) for r in result]
    if len(set(keys)) != len(keys) or any(not r["text"] or not r["target_company"] for r in result):
        raise ValueError("Duplicate ID, missing text or missing target company")
    return result


def make_questions(target_company, form):
    if form not in FORMS or not target_company.strip():
        raise ValueError("Unknown form or missing target")
    questions = {}
    for key, template in QUESTIONS.items():
        q = {"type": form, "instructions": template.replace("<TICKER NAME>", target_company)}
        if form == "choice":
            # Neutral option IDs; descriptions carry the yes/no meaning.
            q["criteria"] = {"A": "yes", "B": "no"}
        questions[key] = q
    return questions


def extract_probabilities(answer, form):
    if answer.get("type") != form:
        raise ValueError("SDK returned an unexpected question type")
    if form == "noul":
        yes = float(answer["noul"])
        no = 1.0 - yes
    else:
        yes, no = (float(answer["probabilities"][k]) for k in ("A", "B"))
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in (yes, no)):
        raise ValueError("Invalid SDK probabilities")
    if abs(yes + no - 1) > 0.0002:
        raise ValueError("SDK yes/no probabilities do not sum to one")
    return yes, no


@contextmanager
def offline_network():
    """Deny IPv4/IPv6 sockets during every scored call, including SDK fallbacks."""
    original = socket.socket.connect

    def blocked(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("E40 inference must run offline after checkpoint download")
        return original(sock, address)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.socket.connect = original


def require_kaggle():
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Kaggle GPU only; do not run the Laya model on this laptop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credit8k", type=Path, required=True)
    parser.add_argument("--e23", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    require_kaggle()
    rows = load_inputs(args.credit8k, args.e23, args.targets)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw_path, manifest_path = output / "e40_laya_raw.csv", output / "e40_laya_raw_manifest.json"
    lock = output / "e40_laya.run_lock"
    if any(p.exists() for p in (raw_path, manifest_path, lock)):
        raise FileExistsError("One scored Kaggle run only; an output or lock already exists")
    with lock.open("x") as stream:
        stream.write(datetime.now(UTC).isoformat())
    # Install only on Kaggle; imports/tests on the laptop never reach this point.
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"laya=={LAYA_VERSION}"])
    import laya
    import torch
    from huggingface_hub import snapshot_download

    if importlib.metadata.version("laya") != LAYA_VERSION or not torch.cuda.is_available():
        raise RuntimeError("Pinned Laya version and a CUDA GPU are required")
    torch.manual_seed(40)
    started = datetime.now(UTC).isoformat()
    load_start = time.perf_counter()
    # This exact checkpoint is fixed. Any load failure propagates; no substitution.
    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions", device="cuda")
    load_seconds = time.perf_counter() - load_start
    if agent.model_id_or_path != MODEL_ID or agent.subfolder != SUBFOLDER:
        raise RuntimeError("Loaded model ID/subfolder differs from the registered checkpoint")
    # Only the fixed subfolder was downloaded, so ask the cache for that subfolder alone.
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=agent.revision,
            allow_patterns=[f"{SUBFOLDER}/*"],
            local_files_only=True,
        )
    )
    checkpoint = snapshot / SUBFOLDER
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError("Fixed typed-decisions weights not present; no fallback allowed")
    checkpoint_files = {
        str(p.relative_to(checkpoint)): digest(p)
        for p in sorted(checkpoint.rglob("*"))
        if p.is_file()
    }
    # Raw option probabilities: no shipped or newly fitted temperature is applied.
    shipped = {
        "temperature": list(agent.temperature),
        "temperature_by_options": dict(agent.temperature_by_options),
    }
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    agent.binning_map = None
    timings = {form: [] for form in FORMS}
    inference_start = time.perf_counter()
    with offline_network(), raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for i, row in enumerate(rows):
            for form in FORMS:
                questions = make_questions(row["target_company"], form)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                result = agent.predict(row["text"], questions)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                timings[form].append(elapsed)
                if getattr(agent, "cpu_fallback_count", 0):
                    raise RuntimeError("CPU fallback occurred; stop this GPU benchmark")
                usage = result.get("usage", {})
                if (
                    usage.get("truncated")
                    or usage.get("truncated_questions")
                    or usage.get("options")
                ):
                    raise RuntimeError(f"Truncated/collapsed input or options at {row['item_id']}")
                for question in QUESTIONS:
                    yes, no = extract_probabilities(result["answers"][question], form)
                    writer.writerow(
                        {
                            "dataset": row["dataset"],
                            "item_id": row["item_id"],
                            "target_company": row["target_company"],
                            "text_sha256": text_digest(row["text"]),
                            "form": form,
                            "question": question,
                            "p_yes": yes,
                            "p_no": no,
                            "call_seconds": elapsed,
                        }
                    )
            stream.flush()
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{len(rows)} headlines, both forms", flush=True)
    total_seconds = time.perf_counter() - inference_start
    package_dir = Path(laya.__file__).parent
    manifest = {
        "experiment": "E40",
        "package": "laya",
        "package_version": LAYA_VERSION,
        "model_id": MODEL_ID,
        "subfolder": SUBFOLDER,
        "resolved_revision": agent.revision,
        "download_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_files_sha256": checkpoint_files,
        "package_source_sha256": {
            str(p.relative_to(package_dir)): digest(p) for p in sorted(package_dir.rglob("*.py"))
        },
        "script_sha256": digest(Path(__file__)),
        "questions_sha256": question_digest(),
        "questions": QUESTIONS,
        "choice_mapping": {"A": "yes", "B": "no"},
        "primary_form": "noul",
        "secondary_form": "choice, reported only",
        "probabilities": "Raw SDK option probabilities with T=1 and no language overrides. "
        "The SDK rounds to four decimals; noul p_no is 1-p_yes. No fit.",
        "shipped_calibration_not_applied": shipped,
        "effective_temperature": [1.0, 1.0, 1.0],
        "inputs_sha256": {
            "8K": digest(args.credit8k),
            "E23": digest(args.e23),
            "targets": digest(args.targets),
        },
        "columns_used": {
            "8K": ["item_id", "text"],
            "E23": ["headline_id", "headline", "target_company"],
            "targets": ["item_id", "target_company"],
        },
        "rows_by_dataset": {"8K": 287, "E23": 243},
        "raw_rows": len(rows) * 6,
        "raw_sha256": digest(raw_path),
        "offline_after_download": True,
        "offline_enforcement": "IPv4/IPv6 socket.connect denied during all scored predictions",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "load_seconds": load_seconds,
        "inference_wall_seconds": total_seconds,
        "timing": {
            form: {
                "calls": len(times),
                "total_seconds": sum(times),
                "mean_seconds_per_headline_three_questions": sum(times) / len(times),
                "call_seconds": times,
            }
            for form, times in timings.items()
        },
        "timing_note": "One three-question call per headline/form; CSV repeats that call time "
        "on Q1/Q2/Q3. Includes the first call, excludes install/load/hash time.",
        "labels_used_by_model": False,
        "threshold_fitted_on_kaggle": False,
    }
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Download {raw_path.name} and {manifest_path.name}. No accuracy scored here.")


if __name__ == "__main__":
    main()
