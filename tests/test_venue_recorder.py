import json
from pathlib import Path

import pytest

from capture.venue_recorder import VenueRecorder
from capture.venues.binance import BinanceVenue
from capture.capture_ledger import read_all, LedgerEvent, SEVERITY_CORRUPTING, SEVERITY_INFO


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
