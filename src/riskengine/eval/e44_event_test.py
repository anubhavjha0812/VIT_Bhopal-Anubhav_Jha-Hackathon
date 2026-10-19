"""E44 local, text-only sampling. No inference, annotation or evaluation command."""

import argparse
import hashlib
import heapq
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskengine.eval.e25_pool import FINAL_FILES, assert_clean, exclusions, remove_seen
from riskengine.eval.e25_sec8k import load_cached_exclusions, sha_file
from riskengine.eval.leaks import norm
from riskengine.paths import ROOT

RULES = ROOT / "docs/others/E44_EVENT_TEST_V2.md"
WORK = ROOT / "private_notes/annotation_uploads/task1/event_test_v2"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def key(text: str) -> str:
    return " ".join(norm(text).split())


def local(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("E44 inputs must be local files inside the repository")
    return path


def specification(path: Path) -> dict:
    return json.loads(path.read_text().split("```json\n", 1)[1].split("```", 1)[0])


def exclusion_inputs(root: Path, spec: dict) -> tuple[set[str], dict, dict]:
    """Validate the offline E25 snapshot and refresh text-only local inputs."""
    cache, baseline = local(root, spec["baseline_cache"]), local(root, spec["baseline_manifest"])
    corpus = load_cached_exclusions(cache, baseline)
    finals = {name: root / path.relative_to(ROOT) for name, path in FINAL_FILES.items()}
    corpus = exclusions(corpus, finals)
    seen = {key(text) for values in corpus.values() for text in values if key(text)}
    records = []
    for relative, column in spec["exclusion_specs"]:
        path = local(root, relative)
        texts = pd.read_csv(path, usecols=[column], dtype=str, keep_default_na=False)[column]
        seen.update(k for text in texts for k in [key(text)] if k)
        records.append(
            {"path": relative, "column": column, "rows": len(texts), "sha256": sha_file(path)}
        )
    ordered = sorted(seen)
    size = spec["near_reference_block"]
    blocks = {f"prior_{i // size:04d}": ordered[i : i + size] for i in range(0, len(ordered), size)}
    report = {
        "baseline_cache": {"path": spec["baseline_cache"], "sha256": sha_file(cache)},
        "baseline_manifest": {"path": spec["baseline_manifest"], "sha256": sha_file(baseline)},
        "input_list_sha256": digest(spec["exclusion_specs"]),
        "files": records,
        "unique_normalised_texts": len(ordered),
        "texts_sha256": digest(ordered),
        "reference_blocks": {
            name: {"rows": len(values), "sha256": digest(values)} for name, values in blocks.items()
        },
    }
    return seen, blocks, report


def readable(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return (
        4 <= len(text.split()) <= 80
        and len(letters) >= 20
        and sum(char.isascii() for char in letters) >= 0.9 * len(letters)
        and not re.search(r"<[^>]+>", text)
        and not re.match(r"https?://", text, re.I)
    )


class Reservoir:
    """Keep a deterministic bounded set per source/hint, independent of chunk boundaries."""

    def __init__(self, spec: dict, seen: set[str]):
        self.spec, self.seen = spec, seen
        # The credit cue has two positive suffix lookaheads. A match anywhere on
        # a line is also a match at its start; avoid retrying every character.
        self.patterns = [
            (
                name,
                re.compile(
                    "^" + pattern
                    if name == "Credit downgrade" and pattern.startswith("(?=.*")
                    else pattern,
                    re.I | re.M,
                ),
            )
            for name, pattern in spec["patterns"].items()
        ]
        self.rows: dict[tuple, dict] = {}
        self.heaps: dict[tuple, list] = {}
        self.occurrences = Counter()
        self.stats = Counter()

    def hint(self, text: str) -> str:
        return next((name for name, pattern in self.patterns if pattern.search(text)), "Other")

    def add(self, chunk: pd.DataFrame, source: str, source_hash: str, offset: int) -> None:
        dates = pd.to_datetime(chunk.Date, format="mixed", utc=True, errors="coerce")
        for number, (row, date) in enumerate(
            zip(chunk.itertuples(index=False), dates, strict=True), offset + 1
        ):
            self.stats["source_rows"] += 1
            text = str(row.Article_title).strip()
            if (
                pd.isna(date)
                or not self.spec["date_start"] <= str(date.date()) <= self.spec["date_end"]
            ):
                self.stats["invalid_or_outside_date"] += 1
                continue
            if not readable(text):
                self.stats["readability_drops"] += 1
                continue
            normal = key(text)
            if normal in self.seen:
                self.stats["exact_prior_occurrence_drops"] += 1
                continue
            hint = self.hint(text)
            self.occurrences[hint] += 1
            rank = digest([self.spec["seed"], source, hint, normal])
            bucket = (source, hint)
            rows, heap = self.rows.setdefault(bucket, {}), self.heaps.setdefault(bucket, [])
            record = {
                "text": text,
                "key": normal,
                "hint": hint,
                "draw_rank": rank,
                "date": str(date.date()),
                "tickers": str(row.Stock_symbol),
                "url": str(row.Url),
                "publisher": str(row.Publisher),
                "source_file": source,
                "source_sha256": source_hash,
                "source_row": number,
            }
            if normal in rows:
                # Stable original source-row metadata, even with duplicated source titles.
                if self.tie(record) < self.tie(rows[normal]):
                    rows[normal] = record
                continue
            if len(heap) >= self.spec["reservoir_per_hint"]:
                if int(rank, 16) >= -heap[0][0]:
                    continue
                _, removed = heapq.heappop(heap)
                del rows[removed]
            rows[normal] = record
            heapq.heappush(heap, (-int(rank, 16), normal))

    @staticmethod
    def tie(row: dict) -> tuple:
        return tuple(
            row[c] for c in ["source_file", "date", "tickers", "url", "publisher", "source_row"]
        )

    def frame(self) -> pd.DataFrame:
        rows = [row for bucket in self.rows.values() for row in bucket.values()]
        if not rows:
            raise ValueError("No eligible unseen real-news candidates")
        frame = pd.DataFrame(rows).sort_values(["draw_rank", "source_file", "key"], kind="stable")
        return frame.drop_duplicates("key").reset_index(drop=True)


def draw_clean(frame: pd.DataFrame, spec: dict, blocks: dict) -> tuple[pd.DataFrame, dict]:
    quotas = {**spec["quotas"], "Other": spec["other_quota"]}
    blocked = set()
    audits = []
    for _ in range(spec["exclusion_rounds"]):
        available = frame[~frame.key.isin(blocked)]
        picked = pd.concat(
            [available[available.hint.eq(hint)].head(n) for hint, n in quotas.items()],
            ignore_index=True,
        )
        if picked.empty:
            raise ValueError("No candidates survive prior-text exclusions")
        kept, audit = remove_seen(picked, blocks)
        audits.append(audit)
        removed = set(picked.key) - set(kept.key)
        if not removed:
            assert_clean(kept, blocks)
            return kept, {
                "passes": audits,
                "near_dropped_unique": len(blocked),
                "short_hint_buckets": {
                    hint: {"target": n, "drawn": int(kept.hint.eq(hint).sum())}
                    for hint, n in quotas.items()
                    if int(kept.hint.eq(hint).sum()) < n
                },
            }
        blocked.update(removed)
    raise RuntimeError("E44 exclusions did not stabilise; no sample is written")


def blind_files(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["shuffle"] = frame.key.map(lambda normal: digest([seed, "blind_shuffle", normal]))
    frame = (
        frame.sort_values(["shuffle", "key"], kind="stable")
        .drop(columns="shuffle")
        .reset_index(drop=True)
    )
    frame.insert(0, "id", [f"E44_{i + 1:04d}" for i in range(len(frame))])
    return frame[["id", "text"]], frame.drop(columns="text")


def sample_once(root: Path = ROOT, protocol: Path = RULES, work: Path = WORK) -> dict:
    if work.exists():
        raise FileExistsError("E44 sample directory already exists; repeat draw refused")
    spec = specification(protocol)
    seen, blocks, excluded = exclusion_inputs(root, spec)
    initial = {
        "protocol": sha_file(protocol),
        "guide": sha_file(local(root, spec["guide"])),
        "script": sha_file(Path(__file__)),
    }
    work.mkdir(parents=True, exist_ok=False)
    (work / "sample.lock").write_text(datetime.now(UTC).isoformat() + "\n")
    reservoir = Reservoir(spec, seen)
    sources = {}
    for relative in spec["sources"]:
        path, offset = local(root, relative), 0
        source_hash = sha_file(path)
        for index, chunk in enumerate(
            pd.read_csv(
                path,
                usecols=spec["columns"],
                dtype=str,
                keep_default_na=False,
                chunksize=spec["chunk_rows"],
            )
        ):
            reservoir.add(chunk[spec["columns"]], relative, source_hash, offset)
            offset += len(chunk)
            if (index + 1) % 10 == 0:
                print(f"E44: {offset:,} source rows streamed", file=sys.stderr, flush=True)
        if sha_file(path) != source_hash:
            raise ValueError("News source changed while streaming")
        sources[relative] = {"sha256": source_hash, "rows": offset}
    frame = reservoir.frame()
    selected, audit = draw_clean(frame, spec, blocks)
    blind, provenance = blind_files(selected, spec["seed"])
    assert_clean(blind, blocks)
    if initial != {
        "protocol": sha_file(protocol),
        "guide": sha_file(local(root, spec["guide"])),
        "script": sha_file(Path(__file__)),
    }:
        raise ValueError("E44 protocol, guide or script changed during sampling")
    blind.to_csv(work / "sample.csv", index=False, mode="x")
    provenance.to_csv(work / "provenance.csv", index=False, mode="x")
    (work / "provenance.csv").chmod(0o600)
    manifest = {
        "experiment": "E44",
        "status": "rules fixed, set drawn, not labelled",
        "utc_time": datetime.now(UTC).isoformat(),
        "seed": spec["seed"],
        "rule_file": str(protocol.relative_to(root)),
        "rule_sha256": initial["protocol"],
        "guide_file": spec["guide"],
        "guide_sha256": initial["guide"],
        "script_sha256": initial["script"],
        "pattern_list_sha256": digest(spec["patterns"]),
        "hint_order": list(spec["patterns"]),
        "sources": sources,
        "exclusions": excluded,
        "scan_counts": dict(reservoir.stats),
        "eligible_occurrences_by_hint_before_near_check": dict(reservoir.occurrences),
        "unique_reservoir_candidates_by_hint": frame.hint.value_counts().to_dict(),
        "sampled_hint_counts": provenance.hint.value_counts().to_dict(),
        "sample_rows": len(blind),
        "exclusion_audit": audit,
        "gold_support": "not labelled; hints are not class labels",
        "models_run": False,
        "files_sha256": {
            name: sha_file(work / name) for name in ["sample.csv", "provenance.csv", "sample.lock"]
        },
    }
    with (work / "sampling.json").open("x") as out:
        json.dump(manifest, out, indent=2)
        out.write("\n")
    (work / "sampling.json").chmod(0o600)
    return summary(manifest)


def summary(manifest: dict) -> dict:
    return {
        name: manifest[name]
        for name in [
            "status",
            "sample_rows",
            "sampled_hint_counts",
            "gold_support",
            "rule_sha256",
            "pattern_list_sha256",
            "files_sha256",
        ]
    } | {
        "short_hint_buckets": manifest["exclusion_audit"]["short_hint_buckets"],
        "exclusion_list_sha256": manifest["exclusions"]["input_list_sha256"],
        "exclusion_texts_sha256": manifest["exclusions"]["texts_sha256"],
    }


def check(root: Path = ROOT, protocol: Path = RULES, work: Path = WORK) -> dict:
    manifest = json.loads((work / "sampling.json").read_text())
    spec = specification(protocol)
    if (
        manifest["rule_sha256"] != sha_file(protocol)
        or manifest["guide_sha256"] != sha_file(local(root, spec["guide"]))
        or manifest["script_sha256"] != sha_file(Path(__file__))
    ):
        raise ValueError("E44 protocol, guide or sampler hash changed")
    for name, expected in manifest["files_sha256"].items():
        if sha_file(work / name) != expected:
            raise ValueError(f"E44 frozen sample hash changed: {name}")
    for relative, info in manifest["sources"].items():
        if sha_file(local(root, relative)) != info["sha256"]:
            raise ValueError("E44 source hash changed")
    seen, blocks, excluded = exclusion_inputs(root, spec)
    if excluded != manifest["exclusions"]:
        raise ValueError("E44 prior-text exclusions changed")
    blind = pd.read_csv(work / "sample.csv", dtype=str, keep_default_na=False)
    provenance = pd.read_csv(work / "provenance.csv", dtype=str, keep_default_na=False)
    expected_ids = [f"E44_{i + 1:04d}" for i in range(len(blind))]
    if (
        list(blind.columns) != ["id", "text"]
        or blind.id.tolist() != expected_ids
        or provenance.id.tolist() != expected_ids
        or len(blind) != manifest["sample_rows"]
    ):
        raise ValueError("E44 blind schema, IDs/order or provenance alignment changed")
    keys = blind.text.map(key)
    if keys.duplicated().any() or set(keys) & seen or keys.tolist() != provenance.key.tolist():
        raise ValueError("E44 duplicate or previously seen text")
    if not blind.text.map(readable).all():
        raise ValueError("E44 unreadable candidate")
    hints = Reservoir(spec, set())
    if blind.text.map(hints.hint).tolist() != provenance.hint.tolist():
        raise ValueError("E44 sampling hint changed")
    if provenance.hint.value_counts().to_dict() != manifest["sampled_hint_counts"]:
        raise ValueError("E44 hint counts changed")
    quotas = {**spec["quotas"], "Other": spec["other_quota"]}
    for hint, count in provenance.hint.value_counts().items():
        if hint not in quotas or count > quotas[hint]:
            raise ValueError("E44 hint quota exceeded")
    dates = pd.to_datetime(provenance.date, utc=True, errors="coerce")
    if (
        dates.isna().any()
        or not provenance.date.between(spec["date_start"], spec["date_end"]).all()
    ):
        raise ValueError("E44 date eligibility changed")
    assert_clean(blind, blocks)
    return {"check": "passed", **summary(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(sample_once() if args.sample else check(), indent=2))


if __name__ == "__main__":
    main()
