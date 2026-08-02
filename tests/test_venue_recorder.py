import json
from pathlib import Path

import pytest

from capture.venue_recorder import VenueRecorder
from capture.venues.binance import BinanceVenue
from capture.capture_ledger import read_all, SEVERITY_CORRUPTING


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
