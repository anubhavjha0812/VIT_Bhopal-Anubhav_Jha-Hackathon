"""Read-only decision log: what the engine proposed and when, in a hash chain.

Each line holds one record (a fired credit alert or the Module A target weights of a day) and the
hash of the line before it, so an edited, removed or reordered line is found by `verify`. It records
the rule-based output; nobody approves or changes anything. It does not prove the records are
true, only that they were not changed afterwards.

uv run python -m riskengine.ledger --build    # rebuild from the stored signals (deterministic)
uv run python -m riskengine.ledger --verify
uv run python -m riskengine.ledger --export   # CSV next to the ledger
"""

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from riskengine.paths import DATA, ROOT

LEDGER = DATA / "outputs" / "decision_ledger.jsonl"
GENESIS = "0" * 64


@contextmanager
def locked(path: Path):
    """Serialise cooperating local processes; released by the OS after a crash."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def append(record: dict, *, path: Path, versions: dict | None = None) -> dict:
    """Append/fsync a live attempt, leaving the historical ledger untouched.

    A payload's attempt_id makes interrupted append retries idempotent. Refuse a
    partial or invalid existing chain rather than silently repairing its evidence.
    """
    if path.resolve() == LEDGER.resolve():
        raise ValueError("Use a separate live ledger; the historical ledger is protected")
    if set(record) != {"kind", "at", "payload"} or not isinstance(record["payload"], dict):
        raise ValueError("Expected kind, at and a dictionary payload")
    with locked(path.with_suffix(".lock")):
        if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            raise ValueError("Incomplete ledger line; decision work stopped")
        existing = read(path) if path.exists() else []
        if verify(existing) != (True, None):
            raise ValueError("Invalid ledger chain; decision work stopped")
        attempt = record.get("payload", {}).get("attempt_id")
        if attempt:
            for entry in existing:
                if entry["payload"].get("attempt_id") == attempt:
                    if any(entry[k] != record[k] for k in ("kind", "at", "payload")):
                        raise ValueError("Conflicting ledger attempt_id")
                    return entry
        prefix = [] if existing else chain([], versions or {})
        previous = (existing or prefix)[-1]
        entry = {"seq": previous["seq"] + 1, **record, "prev": previous["hash"]}
        entry["hash"] = _digest(entry)
        lines = [
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in [*prefix, entry]
        ]
        with path.open("a", encoding="utf-8") as stream:
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        return entry


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def chain(records: list[dict], versions: dict) -> list[dict]:
    """Records (each with `kind`, `at`, `payload`) as a chain that starts with a version entry."""
    entries, prev = [], GENESIS
    items = [{"kind": "versions", "at": "", "payload": versions}, *records]
    for seq, rec in enumerate(items):
        entry = {"seq": seq, **rec, "prev": prev}
        entry["hash"] = prev = _digest(entry)
        entries.append(entry)
    return entries


def write(entries: list[dict], path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))


def read(path: Path = LEDGER) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(entries: list[dict]) -> tuple[bool, int | None]:
    """(True, None) if the chain is intact, else (False, seq of the first bad line)."""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i or e.get("prev") != prev or e.get("hash") != _digest(e):
            return False, i
        prev = e["hash"]
    return True, None


def export(entries: list[dict], path: Path) -> None:
    rows = [
        {"seq": e["seq"], "kind": e["kind"], "at": e["at"], "hash": e["hash"],
         "payload": json.dumps(e["payload"], sort_keys=True)}
        for e in entries
    ]  # fmt: skip
    pd.DataFrame(rows).to_csv(path, index=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alert_records(fired: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in fired.sort_values(["published_at", "doc_id"]).iterrows():
        out.append(
            {
                "kind": "credit_alert",
                "at": pd.Timestamp(r["published_at"]).isoformat(),
                "payload": {
                    "doc_id": str(r["doc_id"]),
                    "ticker": r["ticker"],
                    "event": r["event"],
                    "impact": int(r["impact_formula"] if "impact_formula" in r else r["impact"]),
                    "reason": r["reason"],
                    "text": r["text"],
                },
            }
        )
    return out


def weight_records(weights: pd.DataFrame, rebalancer: str) -> list[dict]:
    return [
        {
            "kind": "module_a_weights",
            "at": pd.Timestamp(day).date().isoformat(),
            "payload": {
                "rebalancer": rebalancer,
                "weights": {t: round(float(w), 6) for t, w in row.items()},
            },
        }
        for day, row in weights.iterrows()
    ]


def build(path: Path = LEDGER) -> list[dict]:
    """Rebuild the ledger from the stored signals, the gate and the Module A backtest."""
    from riskengine.config import load_config
    from riskengine.gate import gate
    from riskengine.ingest.linking import Linker
    from riskengine.ingest.prices import load_prices
    from riskengine.module_a import run_module_a
    from riskengine.pipeline import SIGNALS_PATH, load_signals

    cfg, signals = load_config(), load_signals()
    fired = gate(signals, cfg.impact_trigger, Linker.from_csv())
    fired = fired[fired["fired"]]
    runs = run_module_a(signals, load_prices(), cfg)
    versions = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "config_file_sha256": _sha256(ROOT / "config" / "default.toml"),
        "signals_sha256": _sha256(Path(SIGNALS_PATH)),
    }
    records = alert_records(fired) + weight_records(runs[cfg.rebalancer].weights, cfg.rebalancer)
    records.sort(key=lambda r: (r["at"], r["kind"]))
    entries = chain(records, versions)
    write(entries, path)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    if args.build:
        entries = build()
        print(f"wrote {len(entries)} entries to {LEDGER.relative_to(ROOT)}")
    if args.verify:
        ok, bad = verify(read())
        print("chain intact" if ok else f"chain broken at line {bad}")
        raise SystemExit(0 if ok else 1)
    if args.export:
        export(read(), LEDGER.with_suffix(".csv"))


if __name__ == "__main__":
    main()
