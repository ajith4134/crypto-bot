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
    CaptureLedger, LedgerEvent, SEVERITY_CORRUPTING, SEVERITY_INFO,
    SEVERITY_OBSERVATION_LOSS,
)
from capture.raw_writer import RawCaptureError, RawWriter
from capture.sequencing import BinanceDepthTracker, StalenessTracker

# stream/symbol come from `extract()`, which reads them out of the wire
# payload (event name, "s"/"coin" fields, ...). They end up as filename
# components in RawWriter's path (see raw_writer.paths_for). A value
# containing "/" or ".." would be treated as extra path segments, letting a
# malformed or hostile frame steer writes outside the intended directory.
# Anything that isn't a plain token falls back to "unknown", the same bucket
# already used for frames whose routing fields could not be determined.
_SAFE_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")

# How many times its own widest observed gap a stream may go quiet before the
# silence is reported. The same multiple `StalenessTracker` uses to tell an
# outage from ordinary slowness, for the same reason.
_SILENCE_STALL_MULTIPLE = 3.0


def _safe_path_token(value: str) -> str:
    return value if _SAFE_PATH_TOKEN.match(value) else "unknown"


class VenueRecorder:
    """Routes one venue's frames to per-stream writers, the ledger and gap trackers.

    `queue_size` is accepted for forward compatibility with a future bounded-queue
    task and is currently unused - see the note on `dropped` in `stats()`.
    """

    def __init__(self, venue, specs, root: Path, queue_size: int = 10_000,
                 clock_ns: Callable[[], int] = time.time_ns,
                 silence_grace_seconds: float = 60.0) -> None:
        self._venue = venue
        self._specs = specs
        self._root = Path(root)
        self._queue_size = queue_size
        self._clock_ns = clock_ns
        self._ledger = CaptureLedger(root, venue.name)
        # Subscribed streams, keyed the way writers are keyed, so "did this one
        # ever speak?" is a lookup in `_writers`. See `_record_silent_streams`.
        self._expected_streams = {
            (spec.stream.casefold(), spec.symbol.casefold()): (spec.stream, spec.symbol)
            for spec in specs
        }
        self._session_start_ns = clock_ns()
        self._silence_grace_ns = int(silence_grace_seconds * 1e9)
        # Per stream: when it last spoke, how many frames it has sent, and the
        # widest gap it has shown - the evidence `_record_silent_streams` judges
        # silence against. `_recorded_silent_streams` keeps that to one event
        # per stream per session.
        self._last_frame_ns: dict[tuple[str, str], int] = {}
        self._frames_seen: dict[tuple[str, str], int] = {}
        self._widest_gap_ns: dict[tuple[str, str], int] = {}
        self._recorded_silent_streams: set[tuple[str, str]] = set()
        self._writers: dict[tuple[str, str], RawWriter] = {}
        self._trackers: dict[tuple[str, str], object] = {}
        # Streams whose hour cannot be written, mapped to how many frames that
        # has cost. See `_append_or_quarantine_stream`.
        self._unwritable_streams: dict[tuple[str, str], int] = {}
        self._stats = {"written": 0, "dropped": 0, "control": 0, "malformed": 0,
                       "unwritable": 0}

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
        """The gap owner for one stream: a sequence chain where one exists,
        cadence everywhere else.

        Binance depth carries a U/u/pu chain, and a break in it corrupts the
        book, so that is what is checked (spec 5.4). Every other stream carries
        no sequence at all, and used to get no tracker whatsoever - `trade`,
        `markPrice` and `forceOrder` had no staleness owner, so a stream that
        slowed to a crawl was invisible. `StalenessTracker` works from receipt
        times alone, so it owns all of them.
        """
        key = (stream.casefold(), symbol.casefold())
        if key not in self._trackers:
            if self._venue.name == "binance" and stream == "depth":
                self._trackers[key] = BinanceDepthTracker()
            else:
                self._trackers[key] = StalenessTracker()
        return self._trackers[key]

    def _append_or_quarantine_stream(self, stream: str, symbol: str, payload: str,
                                     t_recv_ns: int, t_exch_ms: int | None,
                                     seq: dict | None, kind: str) -> None:
        """Write one frame, isolating a stream whose hour cannot be written.

        A damaged hour makes `RawWriter.append` raise `HourFileNotAppendable`
        every time, for that one (stream, symbol). Letting it out of the loop
        unwound `consume` entirely and took the venue down with it: on restart
        the first frame killed the session, so an undamaged sibling stream in the
        same session never even had a file created. A single torn hour became a
        permanent crash loop across every stream of the venue - strictly worse
        than the silent data loss the refusal replaced.

        So the damage is contained to the stream that owns it. The stream is
        quarantined on first failure rather than retried per frame: the condition
        is persistent by construction, and retrying would put one ledger event
        and one decompress of the damaged hour behind every frame that arrives.

        Only `RawCaptureError` is isolated. It names damage to one hour's files,
        which is inherently per-stream. Anything else - ENOSPC, a bad descriptor,
        MemoryError - is a whole-recorder condition, and pretending it is
        per-stream would spin instead of surfacing it.
        """
        key = (stream.casefold(), symbol.casefold())
        # Liveness is recorded before anything can go wrong with the write: a
        # stream whose hour is quarantined is still speaking, and reporting it
        # dead as well would be false.
        self._note_stream_spoke(key, t_recv_ns)
        if key in self._unwritable_streams:
            self._unwritable_streams[key] += 1
            self._stats["unwritable"] += 1
            return
        try:
            self._writer_for(stream, symbol).append(
                payload, t_recv_ns, t_exch_ms, seq, kind=kind)
        except RawCaptureError as exc:
            self._unwritable_streams[key] = 1
            self._stats["unwritable"] += 1
            self._ledger.record(LedgerEvent(
                ts_ns=t_recv_ns, venue=self._venue.name, stream=stream,
                kind="unwritable_stream", severity=SEVERITY_CORRUPTING,
                detail={"symbol": symbol, "error": type(exc).__name__,
                        "message": str(exc)},
            ))
            return
        self._stats["written"] += 1

    def _record_unwritable_stream_totals(self) -> None:
        """Put each quarantined stream's frame count in the ledger, not just RAM.

        The per-stream counter dies with the process; the ledger is the record an
        incident is reconstructed from, so the size of the loss has to reach it.

        Cleared as it is recorded, so a second `close()` - `consume` closes in a
        `finally` and callers close explicitly - does not double-count.
        """
        recorded, self._unwritable_streams = self._unwritable_streams, {}
        for (stream, symbol), lost in recorded.items():
            self._ledger.record(LedgerEvent(
                ts_ns=self._clock_ns(), venue=self._venue.name, stream=stream,
                kind="unwritable_stream_total", severity=SEVERITY_CORRUPTING,
                detail={"symbol": symbol, "frames_lost": lost},
            ))

    def _note_stream_spoke(self, key: tuple[str, str], t_recv_ns: int) -> None:
        """Remember that this stream is alive, and how far apart its frames come.

        The widest gap is what `_record_silent_streams` judges silence against,
        and it is kept here rather than taken from a `StalenessTracker` because
        that tracker cannot supply it for the streams that need it most: a
        stream whose ordinary cadence exceeds `stall_multiple` x its floor never
        learns a baseline at all, so its threshold stays pinned at a few seconds
        forever (see the learning-rule note in `sequencing.StalenessTracker`).
        Judging a liquidation feed by that would report it dead every session.
        """
        last = self._last_frame_ns.get(key)
        if last is not None:
            self._widest_gap_ns[key] = max(self._widest_gap_ns.get(key, 0),
                                           t_recv_ns - last)
        self._last_frame_ns[key] = t_recv_ns
        self._frames_seen[key] = self._frames_seen.get(key, 0) + 1

    def _record_silent_streams(self, now_ns: int) -> None:
        """Record every subscribed stream that is not producing frames.

        Two conditions, one event, because they are indistinguishable on disk
        and identical to an operator: a stream that never answered at all, and a
        stream that answered and then died. Neither leaves anything to notice -
        the first creates no file, the second leaves a file that simply stops
        growing, which looks exactly like a healthy stream in a quiet market.
        Both are measured facts here: on 2026-08-02 Binance delivered zero
        `aggTrade`, `markPrice@1s` and `forceOrder` frames while `depth` flowed,
        and an earlier version of this check excluded any stream the moment it
        produced one frame - so a stream that died mid-session was invisible.

        Nothing frame-driven can catch the second condition. A dead stream sends
        no frame, so its tracker never runs; only the venue clock, ticking on
        every other stream's frames and again at close, can ask on its behalf.
        That is why this is not delegated to the gap trackers - and the trackers
        genuinely do not own it, which the previous version of this docstring
        wrongly claimed they did.

        How long is too long is per stream, and never shorter than the grace
        period. A stream that has shown its cadence is judged against
        `_SILENCE_STALL_MULTIPLE` x the widest gap it has actually produced, so
        a liquidation feed minutes between frames is not called dead while a
        100ms depth stream is. A stream that has shown nothing (never spoke, or
        spoke exactly once) has only the grace period to go on.

        Recorded at most once per stream per session: `consume` calls this as
        frames arrive and `close` calls it again, and an event per frame would
        drown the ledger in the anomaly it exists to surface. A stream that
        recovers is deliberately not re-armed - it already had its incident.

        Severity is observation loss, not corruption: what was captured is
        intact, there is simply less of it than was asked for.
        """
        for key, (stream, symbol) in self._expected_streams.items():
            if key in self._recorded_silent_streams:
                continue
            last_ns = self._last_frame_ns.get(key, self._session_start_ns)
            quiet_ns = now_ns - last_ns
            threshold_ns = max(
                self._silence_grace_ns,
                int(_SILENCE_STALL_MULTIPLE * self._widest_gap_ns.get(key, 0)))
            if quiet_ns < threshold_ns:
                continue
            self._recorded_silent_streams.add(key)
            self._ledger.record(LedgerEvent(
                ts_ns=now_ns, venue=self._venue.name, stream=stream,
                kind="silent_stream", severity=SEVERITY_OBSERVATION_LOSS,
                detail={"symbol": symbol,
                        "frames_received": self._frames_seen.get(key, 0),
                        "silent_for_seconds": round(quiet_ns / 1e9, 3),
                        "threshold_seconds": round(threshold_ns / 1e9, 3)},
            ))

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
                    self._record_malformed_frame(payload, t_recv_ns)
                else:
                    self._route_frame(parsed, payload, t_recv_ns)
                # After the frame is routed, never before: a stream whose first
                # frame is this one has already been counted as having spoken,
                # so it cannot be reported silent in the same breath.
                self._record_silent_streams(t_recv_ns)
        finally:
            self.close()

    def _record_malformed_frame(self, payload: str, t_recv_ns: int) -> None:
        """Flag a frame that is not JSON - and store it verbatim anyway."""
        self._stats["malformed"] += 1
        self._ledger.record(LedgerEvent(
            ts_ns=t_recv_ns, venue=self._venue.name, stream="unknown",
            kind="malformed", severity=SEVERITY_INFO,
            detail={"bytes": len(payload)},
        ))
        self._append_or_quarantine_stream(
            "unknown", "unknown", payload, t_recv_ns, None, None,
            kind="malformed")

    def _route_frame(self, parsed, payload: str, t_recv_ns: int) -> None:
        """Send one parsed frame to its writer, tracker and - on a gap - the ledger."""
        meta = self._venue.extract(parsed)
        if meta.kind == "control":
            self._stats["control"] += 1

        stream = _safe_path_token(meta.stream)
        symbol = _safe_path_token(meta.symbol)

        tracker = self._tracker_for(stream, symbol)
        if tracker is not None and meta.kind == "data":
            body = parsed.get("data", parsed)
            if isinstance(tracker, BinanceDepthTracker):
                report = tracker.check(body)
            else:
                report = tracker.check(t_recv_ns)
                # Before a baseline exists, a staleness report says only that
                # the gap beat a fixed floor - it has not been compared against
                # this stream at all. For a stream whose ordinary cadence is
                # slower than that floor, and which can therefore never acquire
                # a baseline, that is a ledger event on every frame it will ever
                # receive. Whether such a stream has died is answered by
                # `_record_silent_streams` from the venue clock instead.
                if not tracker.has_baseline():
                    report = None
            if report is not None:
                self._record_gap(stream, symbol, report, t_recv_ns)

        self._append_or_quarantine_stream(
            stream, symbol, payload, t_recv_ns, meta.t_exch_ms, meta.seq,
            kind=meta.kind)

    def stats(self) -> dict:
        # `dropped` stays at 0 forever in this task: nothing here has anywhere to
        # drop a frame *to*. It only becomes reachable once a bounded queue sits
        # in front of consume() (a later, unwritten task) and can overflow. The
        # key and the zero-assertion are kept now, deliberately, so that task
        # only has to wire in the increment rather than invent the contract.
        #
        # `unwritable` is a different loss and deliberately a different key:
        # frames that reached the recorder and could not be stored because their
        # hour's files are damaged. It counts frames NOT in `written`.
        return dict(self._stats)

    def close(self) -> None:
        # Every writer (and the ledger) must get a close attempt regardless of
        # whether an earlier one raised - e.g. a disk-full during one writer's
        # zstd footer write must not orphan the rest with unflushed buffers.
        # Errors are collected and re-raised after every close was attempted,
        # rather than propagating from the first failure and abandoning the loop.
        errors: list[Exception] = []
        # Before the ledger is closed, and inside the same error collection: a
        # failure to record the totals must not cost the closes below.
        try:
            self._record_unwritable_stream_totals()
        except Exception as exc:
            errors.append(exc)
        # A venue that sent nothing at all never entered the frame loop, so this
        # is the only place that condition can be caught - and it is the worst
        # one, because it leaves no file anywhere to notice the absence of.
        try:
            self._record_silent_streams(self._clock_ns())
        except Exception as exc:
            errors.append(exc)
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
