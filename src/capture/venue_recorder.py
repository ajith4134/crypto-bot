"""Consumes venue frames and routes them to writers, ledger and gap trackers.

Governing principle (see spec): data is never dropped silently, and never
modified to "fix" it. A malformed frame is still written verbatim and flagged.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import AsyncIterator, Callable

from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, SEVERITY_INFO,
)
from capture.raw_writer import RawWriter
from capture.sequencing import BinanceDepthTracker, HyperliquidStalenessTracker

# stream/symbol come from `extract()`, which reads them out of the wire
# payload (event name, "s"/"coin" fields, ...). They end up as filename
# components in RawWriter's path (see raw_writer.paths_for). A value
# containing "/" or ".." would be treated as extra path segments, letting a
# malformed or hostile frame steer writes outside the intended directory.
# Anything that isn't a plain token falls back to "unknown", the same bucket
# already used for frames whose routing fields could not be determined.
_SAFE_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _safe_path_token(value: str) -> str:
    return value if _SAFE_PATH_TOKEN.match(value) else "unknown"


class VenueRecorder:
    """Routes one venue's frames to per-stream writers, the ledger and gap trackers.

    `queue_size` is accepted for forward compatibility with a future bounded-queue
    task and is currently unused - see the note on `dropped` in `stats()`.
    """

    def __init__(self, venue, specs, root: Path, queue_size: int = 10_000,
                 clock_ns: Callable[[], int] = time.time_ns) -> None:
        self._venue = venue
        self._specs = specs
        self._root = Path(root)
        self._queue_size = queue_size
        self._clock_ns = clock_ns
        self._ledger = CaptureLedger(root, venue.name)
        self._writers: dict[tuple[str, str], RawWriter] = {}
        self._trackers: dict[tuple[str, str], object] = {}
        self._stats = {"written": 0, "dropped": 0, "control": 0, "malformed": 0}

    def _writer_for(self, stream: str, symbol: str) -> RawWriter:
        # Keyed on casefolded (stream, symbol) so the same logical stream reported
        # with different casing across frames (e.g. "BTCUSDT" vs "btcusdt") still
        # lands in one writer/file rather than silently splitting across two.
        # The first-seen casing is kept as the writer's on-disk name.
        key = (stream.casefold(), symbol.casefold())
        if key not in self._writers:
            self._writers[key] = RawWriter(self._root, self._venue.name, stream, symbol)
        return self._writers[key]

    def _tracker_for(self, stream: str, symbol: str):
        key = (stream.casefold(), symbol.casefold())
        if key not in self._trackers:
            if self._venue.name == "binance" and stream == "depth":
                self._trackers[key] = BinanceDepthTracker()
            elif self._venue.name == "hyperliquid" and stream == "l2Book":
                self._trackers[key] = HyperliquidStalenessTracker()
            else:
                self._trackers[key] = None
        return self._trackers[key]

    def _record_gap(self, stream: str, symbol: str, report, t_recv_ns: int) -> None:
        self._ledger.record(LedgerEvent(
            ts_ns=t_recv_ns, venue=self._venue.name, stream=stream,
            kind="gap", severity=report.severity,
            detail={"symbol": symbol, **report.detail},
        ))

    async def consume(self, frames: AsyncIterator[str]) -> None:
        # Every RawWriter and the CaptureLedger hold open file handles with
        # buffered zstandard writers. If a frame handler or the frame iterator
        # itself raises, those handles must still be closed (flushing what was
        # already appended) rather than leaked - hence the try/finally around
        # the whole loop rather than just around the happy path.
        try:
            async for payload in frames:
                t_recv_ns = self._clock_ns()
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    self._stats["malformed"] += 1
                    self._ledger.record(LedgerEvent(
                        ts_ns=t_recv_ns, venue=self._venue.name, stream="unknown",
                        kind="malformed", severity=SEVERITY_INFO,
                        detail={"bytes": len(payload)},
                    ))
                    self._writer_for("unknown", "unknown").append(
                        payload, t_recv_ns, None, None, kind="malformed")
                    self._stats["written"] += 1
                    continue

                meta = self._venue.extract(parsed)
                if meta.kind == "control":
                    self._stats["control"] += 1

                stream = _safe_path_token(meta.stream)
                symbol = _safe_path_token(meta.symbol)

                tracker = self._tracker_for(stream, symbol)
                if tracker is not None and meta.kind == "data":
                    body = parsed.get("data", parsed)
                    report = (tracker.check(body) if isinstance(tracker, BinanceDepthTracker)
                              else tracker.check(t_recv_ns))
                    if report is not None:
                        self._record_gap(stream, symbol, report, t_recv_ns)

                self._writer_for(stream, symbol).append(
                    payload, t_recv_ns, meta.t_exch_ms, meta.seq, kind=meta.kind)
                self._stats["written"] += 1
        finally:
            self.close()

    def stats(self) -> dict:
        # `dropped` stays at 0 forever in this task: nothing here has anywhere to
        # drop a frame *to*. It only becomes reachable once a bounded queue sits
        # in front of consume() (a later, unwritten task) and can overflow. The
        # key and the zero-assertion are kept now, deliberately, so that task
        # only has to wire in the increment rather than invent the contract.
        return dict(self._stats)

    def close(self) -> None:
        # Every writer (and the ledger) must get a close attempt regardless of
        # whether an earlier one raised - e.g. a disk-full during one writer's
        # zstd footer write must not orphan the rest with unflushed buffers.
        # Errors are collected and re-raised after every close was attempted,
        # rather than propagating from the first failure and abandoning the loop.
        errors: list[Exception] = []
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        try:
            self._ledger.close()
        except Exception as exc:
            errors.append(exc)

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("errors while closing venue recorder resources", errors)
