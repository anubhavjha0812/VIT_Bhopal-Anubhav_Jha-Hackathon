"""Build the private offline Kaggle upload without reading label contents or running models."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from riskengine.eval.e44_score import (  # noqa: E402
    PLAN,
    PLAN_LOCK,
    load_plan,
    local,
    read_json,
    sha256,
    verify_inputs,
)


def bundle_paths(root: Path, spec: dict) -> list[Path]:
    paths = {local(root, name) for name in spec["expected_sha256"]}
    paths.update((root / "src/riskengine").rglob("*.py"))
    paths.update((root / "notebooks/e44_score").glob("*"))
    paths.update(root / name for name in (PLAN, PLAN_LOCK, "pyproject.toml"))
    work = local(root, spec["work_folder"])
    for name in read_json(work / "labels_final_record.json")["sha256"]:
        paths.add(local(root, spec[name]) if name in {"guide", "protocol"} else work / name)
    paths.update(work / name for name in read_json(work / "sampling.json")["files_sha256"])
    return sorted(path for path in paths if path.is_file())


def prepare(root: Path, output: Path) -> dict:
    spec = load_plan(root)
    verify_inputs(root, spec)
    if any(
        (root / "data/outputs" / name).exists()
        for name in ("e44_predictions.csv", "e44_results.json", "e44.run_lock")
    ):
        raise FileExistsError("E44 already attempted; do not package a fresh rerun")
    files = bundle_paths(root, spec)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("write the private upload outside the repository, e.g. /tmp")
    manifest = {
        "plan_sha256": sha256(root / PLAN),
        "files_sha256": {str(path.relative_to(root)): sha256(path) for path in files},
    }
    with (
        output.open("xb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive,
    ):
        for path in files:
            archive.write(path, str(path.relative_to(root)))
        archive.writestr("e44_bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
        archive.writestr(
            "e44_runtime_requirements.txt",
            "".join(f"{name}=={version}\n" for name, version in spec["runtime"].items()),
        )
    return {
        "archive": str(output),
        "sha256": sha256(output),
        "files": len(files),
        "predictions_or_scoring_run": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(REPO, args.output), indent=2))
