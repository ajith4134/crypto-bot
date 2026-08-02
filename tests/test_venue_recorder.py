import json
import random
from pathlib import Path

import pytest

from capture.venue_recorder import VenueRecorder
from capture.venues.binance import BinanceVenue
from capture.capture_ledger import (
    read_all, LedgerEvent, SEVERITY_CORRUPTING, SEVERITY_INFO,
    SEVERITY_OBSERVATION_LOSS,
)


async def _frames(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_writes_frames_and_counts_them(tmp_path: Path):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    rec = VenueRecorder(venue, specs, tmp_path, clock_ns=lambda: 1785648600_000_000_000)

    payloads = [json.dumps({"stream": "btcusdt@depth@100ms", "data": {
        "e": "depthUpdate", "E": 1785650606214, "s": "BTCUSDT",
        "U": 1 + i * 10, "u": 10 + i * 10, "pu": i * 10}}) for i in range(3)]

    await rec.consume(_frames(payloads))
    assert rec.stats()["written"] == 3
    assert rec.stats()["dropped"] == 0


@pytest.mark.asyncio
async def test_broken_chain_records_corrupting_ledger_event(tmp_path: Path):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    rec = VenueRecorder(venue, specs, tmp_path, clock_ns=lambda: 1785648600_000_000_000)

    good = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                                "U": 1, "u": 10, "pu": 0}})
    broken = json.dumps({"data": {"e": "depthUpdate", "E": 2, "s": "BTCUSDT",
                                  "U": 50, "u": 60, "pu": 49}})
    await rec.consume(_frames([good, broken]))

    events = read_all(tmp_path, "binance", "2026-08-02")
    gaps = [e for e in events if e.kind == "gap"]
    assert len(gaps) == 1
    assert gaps[0].severity == SEVERITY_CORRUPTING


@pytest.mark.asyncio
async def test_malformed_frame_is_still_written(tmp_path: Path):
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)
    await rec.consume(_frames(["this is not json"]))

    assert rec.stats()["malformed"] == 1
    assert rec.stats()["written"] == 1          # written anyway, never discarded
    events = read_all(tmp_path, "binance", "2026-08-02")
    assert any(e.kind == "malformed" for e in events)


@pytest.mark.asyncio
async def test_unsafe_symbol_does_not_escape_output_directory(tmp_path: Path):
    """extract() reads `symbol` straight off the wire (e.g. body["s"]). A frame
    carrying "/" or ".." there must not be able to steer RawWriter's path into
    an unintended directory - it should fall back to the same "unknown" bucket
    already used when routing fields can't be determined."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)
    hostile = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "../../evil",
                                    "U": 1, "u": 10, "pu": 0}})
    await rec.consume(_frames([hostile]))

    assert rec.stats()["written"] == 1
    raw_dir = tmp_path / "raw" / "binance" / "2026-08-02"
    names = [p.name for p in raw_dir.iterdir()]
    assert names, "expected the frame to land somewhere under the venue's raw dir"
    assert all("evil" not in n for n in names)
    assert any(n.startswith("depth_unknown_") for n in names)
    # No directory traversal: nothing was created outside tmp_path's raw tree.
    assert not (tmp_path.parent / "evil").exists()


@pytest.mark.asyncio
async def test_casing_variants_of_same_symbol_share_one_writer(tmp_path: Path):
    """Two frames for the same logical (stream, symbol) but reported with
    different casing must collapse into a single writer/file, not silently
    split the stream across two files keyed by casing alone."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)
    upper = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                                  "U": 1, "u": 10, "pu": 0}})
    lower = json.dumps({"data": {"e": "depthUpdate", "E": 2, "s": "btcusdt",
                                  "U": 11, "u": 20, "pu": 10}})
    await rec.consume(_frames([upper, lower]))

    raw_dir = tmp_path / "raw" / "binance" / "2026-08-02"
    depth_raw_files = [p for p in raw_dir.iterdir() if p.name.startswith("depth_") and p.suffix == ".zst" and p.name.endswith("ndjson.zst")]
    assert len(depth_raw_files) == 1, f"expected one depth file, got {[p.name for p in depth_raw_files]}"


@pytest.mark.asyncio
async def test_frame_iterator_failure_still_closes_writers_and_ledger(tmp_path: Path):
    """If the frame source itself raises mid-stream (e.g. a websocket error),
    everything already appended must still be flushed and closed rather than
    left in buffered zstandard writers that were never finalized."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)

    good = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                                "U": 1, "u": 10, "pu": 0}})

    async def _flaky_frames():
        yield good
        raise ConnectionError("socket dropped")

    with pytest.raises(ConnectionError):
        await rec.consume(_flaky_frames())

    assert rec.stats()["written"] == 1
    raw_dir = tmp_path / "raw" / "binance" / "2026-08-02"
    raw_file = next(p for p in raw_dir.iterdir() if p.name.startswith("depth_") and p.name.endswith("ndjson.zst"))
    idx_file = next(p for p in raw_dir.iterdir() if p.name.startswith("depth_") and p.name.endswith("idx.zst"))

    from capture.raw_writer import read_pair
    pairs = read_pair(raw_file, idx_file)   # raises if the zstd frame was never finalized
    assert len(pairs) == 1
    assert pairs[0][0] == good


@pytest.mark.asyncio
async def test_close_still_closes_remaining_writers_and_ledger_when_one_fails(tmp_path: Path):
    """close() must attempt every writer (and the ledger) even if an earlier
    one raises - e.g. a disk-full during one writer's zstd footer write must
    not orphan every writer after it in iteration order, plus the ledger,
    with unflushed buffered data. The failure must still surface to the
    caller, just not at the cost of abandoning the rest of the cleanup."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)

    # Two distinct writers with open, buffered handles.
    w1 = rec._writer_for("depth", "BTCUSDT")
    w1.append('{"a":1}', 1785648600_000_000_000, None, None)
    w2 = rec._writer_for("aggTrade", "BTCUSDT")
    w2.append('{"a":2}', 1785648600_000_000_000, None, None)

    # An open ledger handle too.
    rec._ledger.record(LedgerEvent(
        ts_ns=1785648600_000_000_000, venue="binance", stream="depth",
        kind="info", severity=SEVERITY_INFO, detail={}))

    assert w1._raw_z is not None and w2._raw_z is not None
    assert rec._ledger._fh is not None

    def _boom() -> None:
        raise OSError("disk full")
    w1.close = _boom

    with pytest.raises(OSError, match="disk full"):
        rec.close()

    # w1 (first in iteration order) is the one that raised - w2 and the
    # ledger must still have been closed rather than left open/unflushed.
    assert w2._raw_z is None
    assert rec._ledger._fh is None


# --------------------------------------------------------------------------
# CRITICAL - one damaged hour must not take the whole venue down
# --------------------------------------------------------------------------

def _depth_frame(symbol: str, i: int) -> str:
    return json.dumps({"data": {"e": "depthUpdate", "E": i, "s": symbol,
                                "U": 1 + i * 10, "u": 10 + i * 10, "pu": i * 10}})


def _trade_frame(symbol: str, i: int) -> str:
    return json.dumps({"data": {"e": "trade", "E": i, "s": symbol, "t": i,
                                "p": "1", "q": "1"}})


def _chop_last_byte(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-1])


@pytest.mark.asyncio
async def test_a_damaged_hour_does_not_stop_a_sibling_stream_recording(tmp_path: Path):
    """The blast radius that made the refusal worse than the bug it replaced.

    `append` refuses a damaged hour with HourFileNotAppendable. Unguarded, that
    unwound `consume` entirely: the recorder died on the FIRST frame of the next
    session, so an undamaged trades/ETHUSDT stream in the same session never had
    a file created at all, and every restart died identically. One torn hour on
    one stream became a permanent crash loop across every stream of the venue.
    """
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    clock = lambda: 1785648600_000_000_000

    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(400)]))
    assert first.stats()["written"] == 400

    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await second.consume(_frames(
        [_depth_frame("BTCUSDT", 500)] + [_trade_frame("ETHUSDT", i) for i in range(3)]))

    # The damaged stream is isolated and its loss counted, not swallowed.
    assert second.stats()["unwritable"] == 1
    assert second.stats()["written"] == 3

    # The sibling stream recorded normally - this is the assertion that used to
    # be impossible, because the session died before it ever opened a file.
    trade_raw, trade_idx = paths_for(tmp_path, "binance", "trade", "ETHUSDT",
                                     "2026-08-02T05")
    from capture.raw_writer import read_pair
    assert len(read_pair(trade_raw, trade_idx)) == 3

    # And the loss is in the ledger, not only in memory.
    events = read_all(tmp_path, "binance", "2026-08-02")
    unwritable = [e for e in events if e.kind == "unwritable_stream"]
    assert len(unwritable) == 1
    assert unwritable[0].severity == SEVERITY_CORRUPTING
    assert unwritable[0].detail["error"] == "HourFileNotAppendable"
    totals = [e for e in events if e.kind == "unwritable_stream_total"]
    assert totals and totals[-1].detail["frames_lost"] == 1


@pytest.mark.asyncio
async def test_a_quarantined_stream_is_not_retried_per_frame(tmp_path: Path):
    """The condition is persistent, so retrying costs a decompress per frame."""
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    clock = lambda: 1785648600_000_000_000

    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(400)]))
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await second.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(20)]))

    assert second.stats()["unwritable"] == 20
    assert second.stats()["written"] == 0
    events = read_all(tmp_path, "binance", "2026-08-02")
    assert len([e for e in events if e.kind == "unwritable_stream"]) == 1, (
        "one event per quarantined stream, not one per frame")
    totals = [e for e in events if e.kind == "unwritable_stream_total"]
    assert totals[-1].detail["frames_lost"] == 20


@pytest.mark.asyncio
async def test_recording_resumes_after_the_hour_is_repaired(tmp_path: Path):
    """refuse -> repair -> resume, end to end through the recorder."""
    from capture.raw_writer import paths_for, reconcile_pair, read_pair

    venue = BinanceVenue()
    clock = lambda: 1785648600_000_000_000

    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(400)]))
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    blocked = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await blocked.consume(_frames([_depth_frame("BTCUSDT", 1)]))
    assert blocked.stats()["unwritable"] == 1

    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged

    resumed = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await resumed.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(5)]))
    assert resumed.stats()["written"] == 5
    assert resumed.stats()["unwritable"] == 0
    assert len(read_pair(raw, idx)) == 5


@pytest.mark.asyncio
async def test_a_non_capture_error_still_unwinds_the_loop(tmp_path: Path):
    """Isolation is for per-hour damage only.

    ENOSPC, a bad descriptor or MemoryError are whole-recorder conditions;
    treating them as per-stream would spin quietly instead of surfacing them.
    """
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)

    def _boom(*args, **kwargs):
        raise OSError("disk full")
    rec._writer_for("depth", "BTCUSDT").append = _boom

    with pytest.raises(OSError, match="disk full"):
        await rec.consume(_frames([_depth_frame("BTCUSDT", 1)]))


# --------------------------------------------------------------------------
# subscribed but silent
# --------------------------------------------------------------------------

def clock_advancing_by(start_ns: int, step_ns: int):
    """A clock that moves on every read, so a session can outrun a grace period
    without the test waiting out real time."""
    state = {"now": start_ns}

    def clock_ns() -> int:
        now = state["now"]
        state["now"] = now + step_ns
        return now

    return clock_ns


def silent_stream_events(root: Path) -> list:
    return [e for e in read_all(root, "binance", "2026-08-02")
            if e.kind == "silent_stream"]


@pytest.mark.asyncio
async def test_a_subscribed_stream_that_never_speaks_reaches_the_ledger(tmp_path: Path):
    """A stream we asked for and never heard from looks exactly like a healthy
    stream in a quiet market - an empty directory nobody notices. Measured
    2026-08-02: Binance delivered depth and nothing else, and only the frames
    that did arrive left any trace at all."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])      # depth, trade, markPrice, forceOrder
    rec = VenueRecorder(venue, specs, tmp_path, silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    40_000_000_000))

    await rec.consume(_frames([
        json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                             "U": 1, "u": 10, "pu": 0}}),
        json.dumps({"data": {"e": "depthUpdate", "E": 2, "s": "BTCUSDT",
                             "U": 11, "u": 20, "pu": 10}}),
    ]))

    events = silent_stream_events(tmp_path)
    assert {e.stream for e in events} == {"trade", "markPrice", "forceOrder"}
    assert all(e.severity == SEVERITY_OBSERVATION_LOSS for e in events)
    assert all(e.detail["symbol"] == "BTCUSDT" for e in events)
    assert all(e.detail["frames_received"] == 0 for e in events)


@pytest.mark.asyncio
async def test_a_stream_is_not_called_silent_during_the_startup_grace(tmp_path: Path):
    """Every stream is silent for the first moments of a session. Flagging that
    would put an event in the ledger on every single start."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    1_000_000_000))

    await rec.consume(_frames([_depth_frame("BTCUSDT", 1)]))

    assert silent_stream_events(tmp_path) == []


@pytest.mark.asyncio
async def test_a_venue_that_sends_nothing_at_all_is_still_recorded(tmp_path: Path):
    """The frame loop cannot notice this one: it never runs. Without a check on
    the way out, a session that captured absolutely nothing leaves an empty
    ledger - indistinguishable from a session that captured everything."""
    venue = BinanceVenue()
    # One tick at construction, one when close() looks at the time: the second
    # read has to be past the grace period for the check to have anything to say.
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    70_000_000_000))

    await rec.consume(_frames([]))

    assert {e.stream for e in silent_stream_events(tmp_path)} == {
        "depth", "trade", "markPrice", "forceOrder"}


@pytest.mark.asyncio
async def test_silence_is_recorded_once_not_on_every_frame(tmp_path: Path):
    """`consume` closes in a finally and callers close explicitly, and frames
    keep arriving after the grace expires. One event per stream per session, or
    the ledger drowns in the anomaly it is meant to surface."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    40_000_000_000))

    await rec.consume(_frames([_depth_frame("BTCUSDT", n) for n in range(1, 6)]))
    rec.close()

    assert len(silent_stream_events(tmp_path)) == 3


@pytest.mark.asyncio
async def test_silence_is_reported_during_the_run_not_only_at_shutdown(tmp_path: Path):
    """A `--seconds 0` capture runs for days. Learning at shutdown that a stream
    never spoke is learning far too late, so the check runs as frames arrive and
    the ledger is written before the session ends."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    40_000_000_000))
    recorded_mid_run = []

    async def frames_and_a_look_at_the_ledger():
        yield _depth_frame("BTCUSDT", 1)          # t0 + 40s: inside the grace
        yield _depth_frame("BTCUSDT", 2)          # t0 + 80s: grace has passed
        recorded_mid_run.extend(silent_stream_events(tmp_path))
        yield _depth_frame("BTCUSDT", 3)

    await rec.consume(frames_and_a_look_at_the_ledger())

    assert {e.stream for e in recorded_mid_run} == {"trade", "markPrice", "forceOrder"}


def gap_events(root: Path) -> list:
    return [e for e in read_all(root, "binance", "2026-08-02") if e.kind == "gap"]


@pytest.mark.asyncio
async def test_a_stream_that_speaks_once_and_then_dies_reaches_the_ledger(tmp_path: Path):
    """The condition the first version of this check could not see: `trade`
    fires once, and is thereafter permanently excluded from the silence check
    because it has a writer. A 24/7 recorder would never notice it died.

    Nothing frame-driven can catch this - a dead stream sends no frame for a
    tracker to run on - so the venue clock has to ask on its behalf.
    """
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    5_000_000_000))

    await rec.consume(_frames(
        [json.dumps({"data": {"e": "trade", "E": 1, "s": "BTCUSDT", "t": 1}})]
        + [_depth_frame("BTCUSDT", i) for i in range(1, 40)]))

    dead = [e for e in silent_stream_events(tmp_path) if e.stream == "trade"]
    assert len(dead) == 1
    assert dead[0].detail["frames_received"] == 1        # it spoke, then died
    assert dead[0].detail["silent_for_seconds"] >= 60
    assert dead[0].severity == SEVERITY_OBSERVATION_LOSS


@pytest.mark.asyncio
async def test_a_healthy_bursty_trade_stream_does_not_alarm(tmp_path: Path):
    """Trades arrive in bursts with quiet stretches between them. An earlier
    round of this project turned exactly that shape into an alarm on 57% of
    frames; a silence check that repeats the mistake is worse than none."""
    venue = BinanceVenue()
    rng = random.Random(7)
    gaps_ns = [int((6.0 if rng.random() < 0.6 else 0.5) * 1e9) for _ in range(300)]
    # One tick at construction, one per frame, one when close() checks silence.
    ticks = [1785648600_000_000_000]
    for gap in gaps_ns + [1_000_000_000] * 3:
        ticks.append(ticks[-1] + gap)
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60, clock_ns=iter(ticks).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "trade", "E": i, "s": "BTCUSDT", "t": i}})
        for i in range(len(gaps_ns))]))

    assert [e for e in silent_stream_events(tmp_path) if e.stream == "trade"] == []
    trade_gaps = [e for e in gap_events(tmp_path) if e.stream == "trade"]
    assert len(trade_gaps) <= len(gaps_ns) * 0.05, (
        f"{len(trade_gaps)} alarms on {len(gaps_ns)} healthy bursty frames")


@pytest.mark.asyncio
async def test_a_sparse_stream_is_judged_against_its_own_cadence(tmp_path: Path):
    """Liquidations arrive minutes apart. Judging that against the grace period
    would report a dead stream on a healthy one every session."""
    venue = BinanceVenue()
    every_180s = [1785648600_000_000_000 + i * 180_000_000_000 for i in range(6)]
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(every_180s + [every_180s[-1]]).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "forceOrder", "E": i,
                             "o": {"s": "BTCUSDT", "q": "1"}}})
        for i in range(len(every_180s) - 1)]))

    assert [e for e in silent_stream_events(tmp_path)
            if e.stream == "forceOrder"] == []


@pytest.mark.asyncio
async def test_a_stream_too_slow_to_learn_a_cadence_does_not_alarm_on_every_frame(
        tmp_path: Path):
    """A liquidation feed minutes between frames never builds a baseline (see
    test_a_stream_slower_than_the_stall_rule_never_learns_a_baseline), so its
    tracker measures every gap against the 5s floor and flags all of them,
    forever. One ledger event per frame for the life of the stream is the alarm
    storm this project has already been bitten by. A staleness report is only
    worth recording once the tracker has something to compare against; whether
    such a stream has died is answered by the silence check instead.
    """
    venue = BinanceVenue()
    every_180s = [1785648600_000_000_000 + i * 180_000_000_000 for i in range(9)]
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(every_180s + [every_180s[-1]]).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "forceOrder", "E": i,
                             "o": {"s": "BTCUSDT", "q": "1"}}})
        for i in range(len(every_180s) - 1)]))

    assert [e for e in gap_events(tmp_path) if e.stream == "forceOrder"] == []


@pytest.mark.asyncio
async def test_a_real_stall_on_a_settled_stream_is_still_recorded(tmp_path: Path):
    """The other half of that trade-off: once a stream has shown what its
    cadence is, a stall against it must still reach the ledger."""
    venue = BinanceVenue()
    ticks = [1785648600_000_000_000 + i * 1_000_000_000 for i in range(30)]
    ticks.append(ticks[-1] + 600_000_000_000)          # a 10 minute hole
    ticks.append(ticks[-1])
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60, clock_ns=iter(ticks).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "trade", "E": i, "s": "BTCUSDT", "t": i}})
        for i in range(30)]))

    stalls = [e for e in gap_events(tmp_path) if e.stream == "trade"]
    assert len(stalls) == 1
    assert stalls[0].severity == SEVERITY_OBSERVATION_LOSS
    assert stalls[0].detail["gap_seconds"] == 600.0


@pytest.mark.asyncio
async def test_a_quarantined_stream_is_not_also_reported_dead(tmp_path: Path):
    """A stream whose hour cannot be written is still speaking. Reporting it
    silent as well would send whoever reads the ledger looking for a venue
    problem that does not exist - the frames are arriving, they just cannot be
    stored, which the unwritable events already say."""
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                          clock_ns=lambda: 1785648600_000_000_000)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(20)]))
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                           silence_grace_seconds=60,
                           clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                       5_000_000_000))
    await second.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(30, 60)]))

    assert second.stats()["unwritable"] == 30
    assert [e for e in silent_stream_events(tmp_path) if e.stream == "depth"] == []
