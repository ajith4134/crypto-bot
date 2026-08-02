"""Regression tests for a raw/index pair surviving opens, closes and repairs.

Every test here reproduces a defect the pre-existing suite did not constrain:
opens that truncated the hour they were meant to extend, a close order that
guaranteed the one direction reconcile_pair cannot repair, and a repair tool
that orphaned the inode a live writer was still filling.
"""
import builtins
import errno
import subprocess
from pathlib import Path

import pytest
import zstandard

from capture.raw_writer import (
    RawWriter, read_pair, reconcile_pair, paths_for, _read_lines,
    TruncatedFrameFile, HourStillBeingWritten, HourFileNotAppendable,
    writing_marker_path,
)
from capture.frame_codec import IndexEntry, encode_index_entry

HOUR_05 = 1785648600_000_000_000        # 2026-08-02T05:30:00Z
HOUR_06 = 1785652200_000_000_000        # 2026-08-02T06:30:00Z


def _write_zst_lines(path: Path, lines: list[str]) -> None:
    cctx = zstandard.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            for line in lines:
                w.write((line + "\n").encode("utf-8"))


def _chop_last_byte(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[:-1])


def _count_lines_tolerantly(path: Path) -> int:
    """Line count that survives a torn file, for asserting on a damaged pair."""
    if not path.exists():
        return 0
    try:
        return len(_read_lines(path))
    except TruncatedFrameFile as exc:
        return len(exc.recovered_lines)

# --------------------------------------------------------------------------
# CRITICAL 1 - opening an hour file must never truncate it
# --------------------------------------------------------------------------

def test_restarting_mid_hour_appends_instead_of_truncating(tmp_path: Path):
    """The normal restart path: venue_recorder close()s on every run exit."""
    first = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(5):
        first.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    first.close()

    second = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    second.append('{"frame":5}', t_recv_ns=HOUR_05 + 5, t_exch_ms=None, seq=None)
    second.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == [f'{{"frame":{i}}}' for i in range(6)]
    # n keeps matching position across the restart, or every later entry would
    # describe the wrong frame.
    assert [p[1].n for p in pairs] == list(range(6))
    assert [p[1].t_recv_ns for p in pairs] == [HOUR_05 + i for i in range(6)]


def test_close_then_append_in_the_same_hour_keeps_earlier_frames(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "trades", "ETHUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()                              # close() clears _hour ...
    w.append('{"frame":3}', t_recv_ns=HOUR_05 + 3, t_exch_ms=None, seq=None)
    w.close()                              # ... so this append re-opens the hour

    raw, idx = paths_for(tmp_path, "binance", "trades", "ETHUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == [f'{{"frame":{i}}}' for i in range(4)]


def test_out_of_order_timestamp_back_across_an_hour_does_not_destroy_it(tmp_path: Path):
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC")
    w.append('{"frame":"05a"}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    w.append('{"frame":"05b"}', t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)
    w.append('{"frame":"06a"}', t_recv_ns=HOUR_06, t_exch_ms=None, seq=None)
    # One late frame reaches back over the boundary and re-opens hour 05.
    w.append('{"frame":"05c"}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)
    w.close()

    r5, i5 = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    assert [p[0] for p in read_pair(r5, i5)] == [
        '{"frame":"05a"}', '{"frame":"05b"}', '{"frame":"05c"}']
    r6, i6 = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T06")
    assert [p[0] for p in read_pair(r6, i6)] == ['{"frame":"06a"}']


def test_a_failed_open_does_not_destroy_the_existing_hour(tmp_path: Path, monkeypatch):
    """An open that never succeeds must not have already destroyed the file."""
    first = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(4):
        first.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    first.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    raw_bytes_before = raw.read_bytes()
    assert len(raw_bytes_before) > 0

    real_open = builtins.open

    def open_failing_on_idx_append(file, mode="r", *args, **kwargs):
        if str(file).endswith(".idx.zst") and "a" in mode:
            raise OSError(errno.EMFILE, "Too many open files")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_failing_on_idx_append)
    second = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    with pytest.raises(OSError):
        second.append('{"frame":4}', t_recv_ns=HOUR_05 + 4, t_exch_ms=None, seq=None)
    monkeypatch.undo()

    assert raw.read_bytes() == raw_bytes_before, "a failed open truncated the raw file"
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(4)]


def test_resuming_into_a_misaligned_hour_is_refused(tmp_path: Path):
    """Resuming needs the frame count; a damaged pair must not be papered over."""
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(raw, ['{"frame":0}', '{"frame":1}'])
    _write_zst_lines(idx, [encode_index_entry(IndexEntry(
        n=0, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None, kind="data", esc=False))])

    w = RawWriter(tmp_path, "v", "s", "SYM")
    with pytest.raises(HourFileNotAppendable):
        w.append('{"frame":2}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)

    # Refusing must not have altered the data it refused to extend.
    assert _read_lines(raw) == ['{"frame":0}', '{"frame":1}']


# --------------------------------------------------------------------------
# IMPORTANT 5 - close() must not leave the unrepairable direction
# --------------------------------------------------------------------------

def test_close_never_leaves_the_index_longer_than_the_raw_file(tmp_path: Path):
    """reconcile_pair rebuilds index entries from raw lines; the reverse is
    impossible. A close that fails on raw must therefore not go on to give the
    index a full tail."""
    from unittest.mock import MagicMock

    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(5):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)

    failing_raw_z = MagicMock(wraps=w._raw_z)
    failing_raw_z.close.side_effect = OSError(errno.ENOSPC, "No space left on device")
    w._raw_z = failing_raw_z

    with pytest.raises(OSError):
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    raw_count = _count_lines_tolerantly(raw)
    idx_count = _count_lines_tolerantly(idx)
    assert idx_count <= raw_count, (
        f"close() left idx={idx_count} lines against raw={raw_count}: "
        f"the one direction reconcile_pair cannot repair")


def test_close_still_releases_every_descriptor_when_raw_close_fails(tmp_path: Path):
    """Abandoning the index's frame must not mean leaking its handle."""
    from unittest.mock import MagicMock

    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"frame":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    idx_fh, raw_fh = w._idx_fh, w._raw_fh

    failing_raw_z = MagicMock(wraps=w._raw_z)
    failing_raw_z.close.side_effect = OSError(errno.ENOSPC, "No space left on device")
    w._raw_z = failing_raw_z

    with pytest.raises(OSError):
        w.close()

    assert idx_fh.closed, "index file handle leaked"
    assert raw_fh.closed, "raw file handle leaked"
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert not writing_marker_path(raw).exists(), "writing marker leaked"


def test_close_leaves_a_repairable_pair_when_the_index_close_fails(tmp_path: Path):
    """The other direction: raw survives whole and repair can rebuild the index."""
    from unittest.mock import MagicMock

    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(5):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)

    failing_idx_z = MagicMock(wraps=w._idx_z)
    failing_idx_z.close.side_effect = OSError(errno.ENOSPC, "No space left on device")
    w._idx_z = failing_idx_z

    with pytest.raises(OSError):
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert _read_lines(raw) == [f'{{"frame":{i}}}' for i in range(5)]
    assert reconcile_pair(raw, idx).entries_rebuilt == 5
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(5)]


# --------------------------------------------------------------------------
# IMPORTANT 6 - repairing a live hour must not orphan the writer's inode
# --------------------------------------------------------------------------

def test_reconcile_refuses_an_hour_a_writer_still_holds_open(tmp_path: Path):
    """_write_lines swaps the inode; a live writer would keep filling the old one."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.flush()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    with pytest.raises(HourStillBeingWritten):
        reconcile_pair(raw, idx)

    for i in range(3, 6):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    # Every frame is still there: the refused repair created no mismatch.
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(6)]


def test_the_writing_marker_is_removed_once_the_hour_is_closed(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"frame":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert writing_marker_path(raw).exists()
    w.close()
    assert not writing_marker_path(raw).exists()
    assert reconcile_pair(raw, idx).entries_rebuilt == 0   # repair is allowed again


def test_a_stale_marker_from_a_dead_process_does_not_block_repair(tmp_path: Path):
    """A crash leaves the marker behind - exactly when repair is needed most."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _write_zst_lines(idx, [encode_index_entry(IndexEntry(
        n=0, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None, kind="data", esc=False))])

    dead = subprocess.Popen(["true"])
    dead.wait()
    writing_marker_path(raw).write_text(str(dead.pid), encoding="utf-8")

    assert reconcile_pair(raw, idx).entries_rebuilt == 2
    assert len(read_pair(raw, idx)) == 3


def test_reconcile_rebuilds_an_index_whose_own_tail_is_torn(tmp_path: Path):
    """The index is rewritten wholesale, so its torn tail is salvageable."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(200):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(idx)
    with pytest.raises(TruncatedFrameFile):
        read_pair(raw, idx)

    outcome = reconcile_pair(raw, idx)
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == [f'{{"frame":{i}}}' for i in range(200)]
    # No raw frame was discarded to make the index fit.
    assert len(pairs) == 200
    assert outcome.entries_discarded == 0 and not outcome.raw_was_salvaged

    # Surface the real cost rather than resting on "repaired > 0". A torn zstd
    # frame loses a whole compressed block, and an hour written in one open is
    # one block: chopping a single byte off the index costs every timestamp in
    # it. Every frame's bytes survive; every frame's metadata does not.
    assert outcome.entries_rebuilt == 200, (
        "one chopped byte on the index costs all 200 entries, not a tail of them")
    assert all(p[1].kind == "recovered" for p in pairs)
    assert all(p[1].t_recv_ns == 0 for p in pairs)


# --------------------------------------------------------------------------
# CRITICAL - the advertised remedy must terminate: refuse -> repair -> resume
# --------------------------------------------------------------------------

def test_reconcile_repairs_a_torn_raw_file_so_appending_resumes(tmp_path: Path):
    """`HourFileNotAppendable` says "run reconcile_pair"; it has to work.

    Refusing a torn raw file left the operator with no exit at all: append
    refused, reconcile raised on the very file it was named as the remedy for,
    and append refused again. Since VenueRecorder.consume did not guard append,
    that was a permanent crash loop for the whole venue, not just the hour.

    Salvage keeps every complete line, quarantines the original bytes rather
    than destroying them, and leaves the pair appendable.
    """
    # Three sessions, so the hour holds three concatenated zstd frames and the
    # damage is confined to the last one. Salvage granularity is the zstd frame,
    # not the line - see test_a_single_block_hour_loses_all_of_it_to_one_byte.
    for session in range(3):
        w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
        for i in range(4):
            w.append(f'{{"session":{session},"i":{i}}}',
                     t_recv_ns=HOUR_05 + session * 4 + i, t_exch_ms=None, seq=None)
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    raw_bytes_before = raw.read_bytes()
    _chop_last_byte(raw)                       # the `kill -9` shape

    # 1. Refuse.
    blocked = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    with pytest.raises(HourFileNotAppendable):
        blocked.append('{"frame":"new"}', t_recv_ns=HOUR_05 + 100, t_exch_ms=None, seq=None)

    # 2. Repair - the step that used to raise TruncatedFrameFile.
    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged
    assert outcome.raw_frames_kept == 8, "the two intact zstd frames must survive"
    assert outcome.entries_discarded == 4, "the torn frame's 4 entries are unbacked"

    # The damaged original is kept verbatim, not destroyed.
    assert outcome.quarantined_paths
    quarantined_raw = next(p for p in outcome.quarantined_paths
                           if p.name.startswith(raw.name))
    assert quarantined_raw.read_bytes() == raw_bytes_before[:-1]

    # 3. Resume - and this is the assertion that makes the cycle terminate.
    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":"new"}', t_recv_ns=HOUR_05 + 100, t_exch_ms=None, seq=None)
    resumed.close()

    pairs = read_pair(raw, idx)
    assert len(pairs) == 9
    assert pairs[-1][0] == '{"frame":"new"}'
    assert [p[1].n for p in pairs] == list(range(9))
    # The salvaged frames are the intact prefix, in order, byte for byte, and
    # they kept the metadata they already had rather than being relabelled.
    assert [p[0] for p in pairs[:8]] == [
        f'{{"session":{s},"i":{i}}}' for s in range(2) for i in range(4)]
    assert [p[1].t_recv_ns for p in pairs[:8]] == [HOUR_05 + i for i in range(8)]


def test_a_single_block_hour_loses_all_of_it_to_one_byte(tmp_path: Path):
    """The honest limit of salvage: it is bounded by zstd's block, not the line.

    400 frames written in one open compress to a single ~370-byte block. Chopping
    one byte off it leaves NOTHING decodable - verified against zstandard 0.25.0
    for decompressobj, stream_reader and chunked stream_reader alike. Salvage
    cannot beat that; only writing more frequent zstd frames could bound it, and
    that is a capture-spec decision about storage cost, not a repair-tool one.

    What repair still guarantees is the part that was broken: the bytes are
    quarantined rather than destroyed, the loss is reported rather than implied,
    and the hour is appendable again.
    """
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    raw_bytes_before = raw.read_bytes()
    _chop_last_byte(raw)

    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged
    assert outcome.raw_frames_kept == 0, "one block, so one byte costs the whole hour"
    assert outcome.entries_discarded == 400
    quarantined_raw = next(p for p in outcome.quarantined_paths
                           if p.name.startswith(raw.name))
    assert quarantined_raw.read_bytes() == raw_bytes_before[:-1], (
        "the unsalvageable bytes must be preserved, not deleted")

    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":400}', t_recv_ns=HOUR_05 + 400, t_exch_ms=None, seq=None)
    resumed.close()
    assert [p[0] for p in read_pair(raw, idx)] == ['{"frame":400}']


def test_reconcile_of_a_torn_raw_file_is_idempotent(tmp_path: Path):
    """A second repair must be a no-op, not another round of salvage."""
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    first = reconcile_pair(raw, idx)
    second = reconcile_pair(raw, idx)
    assert first.raw_was_salvaged and not second.raw_was_salvaged
    assert second.entries_rebuilt == 0 and second.entries_discarded == 0
    assert second.raw_frames_kept == first.raw_frames_kept
    assert not second.quarantined_paths, "a clean pair must not be quarantined again"


def test_reconcile_salvages_when_both_files_are_torn(tmp_path: Path):
    """The shape a single `kill -9` actually leaves: both tails cut."""
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)
    _chop_last_byte(idx)

    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged
    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":400}', t_recv_ns=HOUR_05 + 400, t_exch_ms=None, seq=None)
    resumed.close()
    pairs = read_pair(raw, idx)
    assert pairs[-1][0] == '{"frame":400}'
    assert [p[1].n for p in pairs] == list(range(len(pairs)))


def test_a_repair_interrupted_halfway_can_still_be_finished(tmp_path: Path):
    """Repair runs after a crash, so it has to survive being crashed itself.

    The write order is load-bearing. Writing the salvaged raw file first and
    dying leaves a short raw file beside the original long index; the raw file is
    no longer torn, so the overrunning index is correctly refused as
    UnrepairableIndex - a dead end, which is the trap reconcile_pair exists to
    remove. Writing the index first leaves the torn raw file untouched, and the
    next run redoes the whole salvage.
    """
    import capture.raw_writer as raw_writer_module

    for session in range(3):
        w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
        for i in range(4):
            w.append(f'{{"session":{session},"i":{i}}}',
                     t_recv_ns=HOUR_05 + session * 4 + i, t_exch_ms=None, seq=None)
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    real_write_lines = raw_writer_module._write_lines
    calls = []

    def write_lines_dying_on_the_second_file(path, lines):
        calls.append(Path(path))
        if len(calls) == 2:
            raise OSError(errno.EIO, "crashed mid-repair")
        return real_write_lines(path, lines)

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(raw_writer_module, "_write_lines",
                          write_lines_dying_on_the_second_file)
    with pytest.raises(OSError):
        reconcile_pair(raw, idx)
    monkeypatched.undo()

    # The property that matters: the interrupted repair is finishable. Under the
    # other write order this raises UnrepairableIndex and the hour is stuck.
    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_frames_kept == 8
    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":"new"}', t_recv_ns=HOUR_05 + 100, t_exch_ms=None, seq=None)
    resumed.close()
    assert len(read_pair(raw, idx)) == 9

    # Why it is finishable: the raw file was still the torn original when the
    # crash landed, so the second run redid the whole salvage.
    assert calls[0] == idx and calls[1] == raw


def test_reconcile_rebuilds_an_index_that_is_missing_entirely(tmp_path: Path):
    """A missing index is the extreme of the damage repair exists to fix.

    `_open` refuses it the same way it refuses a torn pair, so refusing to
    repair it would be the same dead end.
    """
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    idx.unlink()

    blocked = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    with pytest.raises(HourFileNotAppendable):
        blocked.append('{"frame":3}', t_recv_ns=HOUR_05 + 3, t_exch_ms=None, seq=None)

    outcome = reconcile_pair(raw, idx)
    assert outcome.entries_rebuilt == 3

    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":3}', t_recv_ns=HOUR_05 + 3, t_exch_ms=None, seq=None)
    resumed.close()
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(4)]


def test_reconcile_reports_a_missing_raw_file_as_a_capture_error(tmp_path: Path):
    """A bare FileNotFoundError escapes the documented base class."""
    from capture.raw_writer import MissingPairFile, RawCaptureError

    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(idx, [encode_index_entry(IndexEntry(
        n=0, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None, kind="data", esc=False))])

    with pytest.raises(MissingPairFile) as exc_info:
        reconcile_pair(raw, idx)
    assert isinstance(exc_info.value, RawCaptureError)


def test_reconcile_of_an_absent_hour_is_a_no_op(tmp_path: Path):
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    outcome = reconcile_pair(raw, idx)
    assert not outcome and outcome.entries_rebuilt == 0
