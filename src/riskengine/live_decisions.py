"""Opt-in scored-live-row -> unchanged gate -> stress preview -> local journal.

No inference, execution, history rewriting or Module A weight changes occur here.
"""

import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from functools import cache, lru_cache
from pathlib import Path
from statistics import median

import pandas as pd

from riskengine import ledger
from riskengine.config import Config
from riskengine.gate import CONFIRM_WINDOW, gate
from riskengine.ingest.linking import Linker
from riskengine.paths import DATA, ROOT

LIVE = DATA / "live"
BOOK = DATA / "reference" / "book.csv"
TERMINAL = {"decided", "skipped"}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@lru_cache(maxsize=256)
def _file_digest(path: str, size: int, modified: int) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path: Path) -> str:
    stat = path.stat()
    return _file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@cache
def code_version() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
                text=True,
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def model_files(config: Config) -> dict[str, Path]:
    """Local weights and their inference assets; never fetch a remote checkpoint."""
    from riskengine.nlp.events import OWN_TOPIC
    from riskengine.nlp.sentiment import ENTITY_SENTIMENT, OWN_SENTIMENT

    if config.sentiment == "finbert":
        raise ValueError("Local FinBERT model-file provenance unavailable; no download attempted")
    folders = {"sentiment": OWN_SENTIMENT}
    if config.sentiment == "entity":
        folders["entity"] = ENTITY_SENTIMENT
    if config.events != "keywords":
        folders["events"] = OWN_TOPIC
    result = {}
    for kind, folder in folders.items():
        weights = folder / "model.onnx"
        if not weights.is_file():
            raise FileNotFoundError(f"Missing local {kind} weights: {weights}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in {".json", ".onnx", ".txt"}:
                result[f"{kind}/{path.name}"] = path
    return result


def stress_preview(row: dict, config: Config, book_path: Path) -> dict:
    """Use existing pricing once; no Monte Carlo, model, order or book write."""
    from riskengine.module_b import report
    from riskengine.module_b.shock import BOOK_DATE, DEFAULT_EPISODE, parse_notches

    book = pd.read_csv(book_path)
    ticker = row["ticker"]
    if ticker not in set(book["ticker"]):
        raise ValueError(f"No exposure for {ticker} in the stored book")
    impact = row.get("impact_formula", row["impact"])
    notches = parse_notches(row["text"], row["event"])
    multiple = report.alert_multiple(impact, config.shock)
    valuation = report.revalue(book, ticker, notches, DEFAULT_EPISODE, multiple, config.rates)
    return {
        "dry_run": True,
        "execution": "preview only",
        "module_a": "no change",
        "book_date": BOOK_DATE,
        "scenario": {
            "ticker": ticker,
            "notches": notches,
            "episode": DEFAULT_EPISODE,
            "multiple": multiple,
            "rates": config.rates,
        },
        "capital": report._capital(valuation),
        "obligor_loss_usd": float(
            valuation.positions.loc[valuation.positions["ticker"] == ticker, "loss"].sum()
        ),
        "limits": "Stored hypothetical book and historical what-if episode, not a forecast.",
    }


def company_rows(scored: dict) -> list[dict]:
    """Explode supplied company scores; never relink, rescore or recalculate impact."""
    entities = scored.get("entities")
    if not entities:
        entities = [
            {
                "ticker": ticker,
                "role": "analyst"
                if ticker in scored.get("analysts", [])
                else scored.get("role", "subject"),
            }
            for ticker in dict.fromkeys(
                [scored["ticker"]]
                if scored.get("ticker")
                else [*scored.get("tickers", []), *scored.get("analysts", [])]
            )
        ] or [{"ticker": None, "role": "subject"}]
    rows = []
    for entity in entities:
        row = {
            k: scored.get(k)
            for k in (
                "doc_id",
                "text",
                "published_at",
                "source",
                "event",
                "impact",
                "impact_formula",
                "sentiment",
                "sentiment_path",
                "model",
                "ingested_at",
            )
            if k in scored
        }
        row.update(
            {k: entity[k] for k in ("ticker", "role", "sentiment", "sentiment_path") if k in entity}
        )
        row.setdefault("role", "subject")
        rows.append(row)
    return rows


def publication_minute(row: dict) -> str:
    stamp = pd.Timestamp(row["published_at"])
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("published_at must include a timezone")
    return stamp.tz_convert("UTC").floor("min").isoformat()


def decision_key(row: dict, provenance: dict) -> str:
    try:
        minute = publication_minute(row)
    except (KeyError, TypeError, ValueError):
        minute = f"invalid:{row.get('published_at')}"
    return digest(
        {
            "text": " ".join(str(row.get("text", "")).casefold().split()),
            "ticker": row.get("ticker"),
            "event": row.get("event"),
            "published_minute": minute,
            **{k: provenance.get(k) for k in ("config_hash", "model_hash", "book_hash")},
        }
    )


class LiveDecisions:
    def __init__(
        self,
        config: Config,
        linker: Linker,
        *,
        path: Path = LIVE / "decisions.jsonl",
        ledger_path: Path = LIVE / "decision_ledger.jsonl",
        book_path: Path = BOOK,
        files: Mapping[str, Path] | None = None,
        preview: Callable[[dict, Config, Path], dict] = stress_preview,
    ):
        if path.with_suffix(".lock").resolve() == ledger_path.with_suffix(".lock").resolve():
            raise ValueError("Decision index and live ledger need separate paths")
        if any(p.resolve() == ledger.LEDGER.resolve() for p in (path, ledger_path)):
            raise ValueError("The historical ledger is protected")
        self.config, self.linker = config, linker
        self.path, self.ledger_path, self.book_path = path, ledger_path, book_path
        self.files, self.preview = files, preview
        self.last_error: str | None = None
        self.processing_failures = 0

    def provenance(self) -> dict:
        config = json.loads(canonical(asdict(self.config)))
        out = {
            "config": config,
            "config_hash": digest(config),
            "code_version": code_version(),
            "model_hash": None,
            "model_files_sha256": {},
            "book_hash": None,
            "model_hash_scope": "Configured local files at decision time; no rescoring",
            "missing": [],
        }
        try:
            files = self.files if self.files is not None else model_files(self.config)
            if not files:
                raise ValueError("No model files supplied")
            out["model_files"] = {k: str(p.resolve()) for k, p in sorted(files.items())}
            out["model_files_sha256"] = {k: file_digest(p) for k, p in sorted(files.items())}
            out["model_hash"] = digest(out["model_files_sha256"])
        except (OSError, ValueError) as exc:
            out["missing"].append(f"model-file hash: {exc}")
        try:
            out["book_hash"] = file_digest(self.book_path)
        except OSError as exc:
            out["missing"].append(f"book hash: {exc}")
        sources = [
            Path(__file__),
            ROOT / "src/riskengine/live.py",
            ROOT / "src/riskengine/api.py",
            ROOT / "src/riskengine/ledger.py",
            ROOT / "src/riskengine/gate.py",
            ROOT / "src/riskengine/gate_rules.py",
            *sorted((ROOT / "src/riskengine/module_b").glob("*.py")),
        ]
        out["code_hash"] = digest({str(p.relative_to(ROOT)): file_digest(p) for p in sources})
        return out

    def _append_index(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _restore(self, repair: bool) -> tuple[dict, list[dict]]:
        if self.ledger_path.exists() and self.ledger_path.stat().st_size:
            if not self.ledger_path.read_bytes().endswith(b"\n"):
                raise ValueError("Incomplete live ledger line; decision work stopped")
        entries = ledger.read(self.ledger_path) if self.ledger_path.exists() else []
        if ledger.verify(entries) != (True, None):
            raise ValueError("Live ledger verification failed")
        committed = {}
        for entry in entries:
            if entry["kind"] == "live_decision":
                rec = {**entry["payload"], "ledger_hash": entry["hash"]}
                committed[rec["attempt_id"]] = rec
        indexed = []
        if self.path.exists():
            if self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n"):
                raise ValueError("Incomplete decisions line; decision work stopped")
            indexed = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        seen = set()
        for rec in indexed:
            attempt_id = rec["attempt_id"]
            if attempt_id in seen or rec != committed.get(attempt_id):
                raise ValueError("Decision index differs from its live ledger")
            seen.add(attempt_id)
        if repair:
            for attempt_id, rec in committed.items():
                if attempt_id not in seen:
                    self._append_index(rec)
        latest = {}
        for rec in committed.values():
            latest[rec["key"]] = rec
        return latest, list(committed.values())

    @staticmethod
    def _validate(row: dict) -> None:
        publication_minute(row)
        for field in ("text", "event", "source", "model"):
            if not row.get(field):
                raise ValueError(f"Missing scored {field}")
        impact = row.get("impact_formula", row.get("impact"))
        if (
            not isinstance(impact, (int, float))
            or not math.isfinite(impact)
            or not 1 <= impact <= 10
        ):
            raise ValueError("Missing/invalid scored impact")

    def process(self, scored_rows: Sequence[dict]) -> list[dict]:
        if not self.config.live_decisions:
            return []
        try:
            with ledger.locked(self.path.with_suffix(".lock")):
                return self._process(scored_rows)
        except Exception as exc:
            self.last_error = str(exc)
            self.processing_failures += 1
            return []  # ingestion must continue even if the decision journal is unavailable

    def _process(self, scored_rows: Sequence[dict]) -> list[dict]:
        latest, _ = self._restore(repair=True)
        provenance = self.provenance()
        sources = {}
        for row in scored_rows:
            try:
                source = json.loads(canonical(row))
            except (TypeError, ValueError):
                source = {
                    **{
                        k: str(row.get(k, ""))
                        for k in ("text", "published_at", "event", "source", "model")
                    },
                    "tickers": row.get("tickers", []),
                    "impact": None,
                    "invalid_input_repr": repr(row),
                }
            sources[digest(source)] = source
        for rec in latest.values():
            if rec["status"] == "error":
                sources.setdefault(rec["input_hash"], rec["input"])
        candidates, valid = {}, []
        for input_hash, source in sources.items():
            for row in company_rows(source):
                key = decision_key(row, provenance)
                error = None
                try:
                    self._validate(row)
                    valid.append({**row, "ticker": row.get("ticker") or ""})
                except (KeyError, TypeError, ValueError) as exc:
                    error = str(exc)
                candidates.setdefault(key, (row, source, input_hash, error))
        # Preserve corroborating evidence for a preview retry after the feed cache evicts it.
        for rec in latest.values():
            if rec["status"] == "error":
                valid.extend(rec.get("retry_context", []))
        gate_results = {}
        if all(key in latest and latest[key]["status"] in TERMINAL for key in candidates):
            self.last_error = None
            return []
        if valid:
            frame = pd.DataFrame(valid)
            frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, format="mixed")
            frame = frame.sort_values("published_at", kind="stable")
            if "impact_formula" in frame:
                frame["impact_formula"] = frame["impact_formula"].fillna(frame["impact"])
            try:
                gated = gate(frame, self.config.impact_trigger, self.linker, precision_rules=True)
                for rec in gated.to_dict("records"):
                    rec["ticker"] = rec["ticker"] or None
                    gate_results[decision_key(rec, provenance)] = rec
            except Exception as exc:
                for key in candidates:
                    row, source, input_hash, error = candidates[key]
                    candidates[key] = row, source, input_hash, error or f"gate: {exc}"
        written = []
        for key, (row, source, input_hash, error) in candidates.items():
            previous = latest.get(key)
            if previous and previous["status"] in TERMINAL:
                continue
            attempt = previous["attempt"] + 1 if previous else 1
            now = datetime.now(UTC)
            rec = {
                "key": key,
                "attempt": attempt,
                "attempt_id": f"{key}:{attempt}",
                "input_hash": input_hash,
                "input": source,
                "company": row,
                **provenance,
                "at": now.isoformat(),
                "status": "error",
                "error": None,
                "preview": None,
                "module_a": "no change",
                "execution": "preview only",
                "gate": {"fired": None, "reason": "not evaluated"},
            }
            try:
                if error:
                    raise ValueError(error)
                result = gate_results[key]
                rec["gate"] = {"fired": bool(result["fired"]), "reason": result["reason"]}
                if provenance["missing"]:
                    raise ValueError("; ".join(provenance["missing"]))
                if result["fired"]:
                    rec["preview"] = self.preview(row, self.config, self.book_path)
                    if (
                        not isinstance(rec["preview"], dict)
                        or rec["preview"].get("dry_run") is not True
                    ):
                        raise ValueError("Preview must be a dry-run object")
                    canonical(rec["preview"])  # reject non-finite/unserialisable preview output
                    if file_digest(self.book_path) != provenance["book_hash"]:
                        raise ValueError("Book changed during preview; result discarded")
                    rec["status"] = "decided"
                else:
                    rec["status"] = "skipped"
            except Exception as exc:
                rec["error"] = str(exc)
                rec["preview"] = None
            if rec["status"] == "error" and rec["gate"]["fired"] is True:
                stamp = pd.Timestamp(row["published_at"])
                evidence = frame[
                    (frame["ticker"] == row["ticker"])
                    & (frame["event"] == row["event"])
                    & (frame["published_at"] < stamp)
                    & (frame["published_at"] >= stamp - CONFIRM_WINDOW)
                ]
                fields = [
                    k
                    for k in (
                        "text",
                        "ticker",
                        "role",
                        "event",
                        "published_at",
                        "source",
                        "model",
                        "impact",
                        "impact_formula",
                    )
                    if k in evidence
                ]
                rec["retry_context"] = json.loads(canonical(evidence[fields].to_dict("records")))
            finished = datetime.now(UTC)
            rec["started_at"] = rec["at"]
            rec["at"] = finished.isoformat()
            try:
                ingested = pd.Timestamp(source["ingested_at"])
                if ingested.tzinfo is None or pd.isna(ingested):
                    raise ValueError("ingested_at lacks timezone")
                lag = (finished - ingested.to_pydatetime()).total_seconds()
                if lag < 0:
                    raise ValueError("ingested_at is in the future")
                rec["ingest_to_decision_seconds"] = lag
                rec["lag_missing_reason"] = None
            except (KeyError, TypeError, ValueError) as exc:
                rec["ingest_to_decision_seconds"] = None
                rec["lag_missing_reason"] = str(exc)
            entry = ledger.append(
                {"kind": "live_decision", "at": rec["at"], "payload": rec},
                path=self.ledger_path,
                versions=provenance,
            )
            rec = {**rec, "ledger_hash": entry["hash"]}
            self._append_index(rec)
            written.append(rec)
        self.last_error = None
        return written

    def health(self, scored_rows: Sequence[dict]) -> dict:
        if not self.config.live_decisions:
            return {
                "enabled": False,
                "backlog": None,
                "error_count": None,
                "missing_reason": "live_decisions is off; no decision work performed",
            }
        try:
            latest, attempts = self._restore(repair=False)
            provenance = self.provenance()
            pending = {
                decision_key(row, provenance)
                for source in scored_rows
                for row in company_rows(source)
            }
            pending = {key for key in pending if key not in latest}
            active = [
                rec
                for rec in latest.values()
                if all(
                    rec.get(k) == provenance.get(k)
                    for k in ("config_hash", "model_hash", "book_hash")
                )
            ]
            errors = sum(rec["status"] == "error" for rec in active)
            recent = sorted(latest.values(), key=lambda row: row["at"])[-100:]
            lags = [
                r["ingest_to_decision_seconds"]
                for r in recent
                if r["status"] in TERMINAL and r["ingest_to_decision_seconds"] is not None
            ]
            return {
                "enabled": True,
                "status_counts": dict(Counter(r["status"] for r in latest.values())),
                "error_count": errors,
                "error_count_scope": "Latest errors for the current config/model/book hashes",
                "error_attempt_count": sum(r["status"] == "error" for r in attempts),
                "processing_failures": self.processing_failures,
                "last_error": self.last_error,
                "backlog": len(pending) + errors,
                "pending": len(pending),
                "lag": {
                    "window": 100,
                    "count": len(lags),
                    "median_seconds": median(lags) if lags else None,
                    "max_seconds": max(lags) if lags else None,
                    "missing_reason": None
                    if lags
                    else "No terminal decision with valid ingest time",
                },
                "recent": [
                    {
                        k: rec[k]
                        for k in (
                            "key",
                            "status",
                            "at",
                            "gate",
                            "preview",
                            "module_a",
                            "ledger_hash",
                        )
                    }
                    for rec in recent[-10:]
                ],
                "journal_missing_reason": "No decisions yet" if not attempts else None,
            }
        except Exception as exc:
            return {
                "enabled": True,
                "error_count": None,
                "backlog": None,
                "last_error": str(exc),
                "missing_reason": "Cannot read/verify live journal",
            }
