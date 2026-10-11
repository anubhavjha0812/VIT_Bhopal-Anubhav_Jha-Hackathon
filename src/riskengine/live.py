"""Background live news ingestion, isolated from historical signals and backtests."""

import hashlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from riskengine.ingest.gdelt import GdeltClient
from riskengine.paths import DATA

log = logging.getLogger(__name__)


class LiveFeed:
    def __init__(
        self, client: GdeltClient, analyze: Callable[[str], dict],
        path: Path = DATA / "live" / "gdelt_signals.jsonl", interval: float = 300,
    ):
        self.client, self.analyze, self.path = client, analyze, path
        self.interval = interval
        self.rows: deque[dict] = deque(maxlen=500)
        self.seen: dict[str, None] = {}
        self.lock = threading.Lock()
        self.poll_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_attempt = 0.0
        self.status = {
            "polling": False, "last_poll": None, "last_success": None,
            "error": None, "warning": None, "active_source": None, "polls": 0,
        }
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        self.remember(row)
                        self.retain(row)
                    except (ValueError, KeyError, TypeError):
                        continue

    @staticmethod
    def keys(row: dict) -> tuple[str, str]:
        return row["url"], " ".join(row["text"].casefold().split())

    def remember(self, row: dict) -> None:
        for key in self.keys(row):
            self.seen[key] = None
        while len(self.seen) > 10000:
            del self.seen[next(iter(self.seen))]

    def retain(self, row: dict) -> None:
        self.rows = deque(
            sorted([*self.rows, row], key=lambda item: item["published_at"])[-500:],
            maxlen=500,
        )

    def start(self) -> None:
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.run, name="gdelt-live", daemon=True)
            self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=2)

    def refresh(self) -> None:
        self.start()
        self.wake.set()

    def poll_once(self) -> None:
        if not self.poll_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                self.last_attempt = time.monotonic()
                self.status.update(polling=True, last_poll=datetime.now(UTC).isoformat())
                self.status["polls"] += 1
            rows, source, warning = self.client.fetch()
            added = 0
            for row in sorted(rows, key=lambda item: item["published_at"], reverse=True):
                if self.stop_event.is_set():
                    break
                if any(key in self.seen for key in self.keys(row)):
                    continue
                result = self.analyze(row["text"])
                scored = {
                    **row, **result,
                    "doc_id": hashlib.sha256(row["url"].encode()).hexdigest()[:24],
                    "ingested_at": datetime.now(UTC).isoformat(),
                }
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(scored, ensure_ascii=False) + "\n")
                with self.lock:
                    self.retain(scored)
                    self.remember(scored)
                added += 1
                if added == 50:
                    break
            with self.lock:
                self.status.update(
                    last_success=datetime.now(UTC).isoformat(), error=None,
                    active_source=source, warning=warning,
                )
        except Exception as exc:
            log.warning("Live GDELT poll failed: %s", exc)
            with self.lock:
                self.status["error"] = str(exc)
        finally:
            with self.lock:
                self.status["polling"] = False
            self.poll_lock.release()

    def run(self) -> None:
        while not self.stop_event.is_set():
            cooldown = max(0, 60 - (time.monotonic() - self.last_attempt))
            if self.stop_event.wait(cooldown):
                return
            self.wake.clear()
            self.poll_once()
            self.wake.wait(self.interval)

    def snapshot(self, ticker: str | None = None, min_impact: int = 1) -> dict:
        with self.lock:
            rows = sorted(self.rows, key=lambda row: row["published_at"], reverse=True)
            matching = [
                row for row in rows if row["impact"] >= min_impact
                and (not ticker or ticker.upper() in row["tickers"])
            ]
            return {
                **self.status, "poll_interval_seconds": self.interval,
                "next_poll_in": max(
                    0, round(self.interval - (time.monotonic() - self.last_attempt))
                ),
                "total": len(rows), "high_impact": sum(row["impact"] > 7 for row in rows),
                "tickers": sorted({ticker for row in rows for ticker in row["tickers"]}),
                "signals": matching,
            }
