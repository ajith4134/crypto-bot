from pathlib import Path
import pytest
from capture.raw_writer import RawWriter, read_pair, hour_key, paths_for, PairLengthMismatch


def test_hour_key_is_utc():
    # 2026-08-02T05:30:00Z
    assert hour_key(1785648600_000_000_000) == "2026-08-02T05"


def test_written_bytes_are_identical_to_input(tmp_path: Path):
    payload = '{"e":"depthUpdate","E":1785650606214,"U":98135459427,"u":98135459442}'
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append(payload, t_recv_ns=1785648600_000_000_000,
             t_exch_ms=1785650606214, seq={"U": 98135459427, "u": 98135459442})
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert len(pairs) == 1
    assert pairs[0][0] == payload          # byte-exact
    assert pairs[0][1].n == 0
    assert pairs[0][1].esc is False


def test_index_line_count_matches_raw_line_count(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "trades", "ETHUSDT")
    for i in range(50):
        w.append(f'{{"i":{i}}}', t_recv_ns=1785648600_000_000_000 + i,
                 t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "binance", "trades", "ETHUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert len(pairs) == 50
    assert [p[1].n for p in pairs] == list(range(50))


def test_payload_with_newline_roundtrips(tmp_path: Path):
    payload = '{"a":"x\ny"}'
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC")
    w.append(payload, t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert pairs[0][0] == payload
    assert pairs[0][1].esc is True


def test_rotation_creates_new_hour_file(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"h":5}', t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.append('{"h":6}', t_recv_ns=1785652200_000_000_000, t_exch_ms=None, seq=None)
    w.close()

    r5, i5 = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    r6, i6 = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T06")
    assert read_pair(r5, i5)[0][0] == '{"h":5}'
    assert read_pair(r6, i6)[0][0] == '{"h":6}'
    # n resets per file
    assert read_pair(r6, i6)[0][1].n == 0


def test_payload_with_backslash_roundtrips_exactly(tmp_path: Path):
    """Regression test for finding 1: backslash should not be corrupted."""
    payload = '{"path":"C:\\\\Users\\\\data"}'
    w = RawWriter(tmp_path, "test", "trades", "SYMBOL")
    w.append(payload, t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "test", "trades", "SYMBOL", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert pairs[0][0] == payload  # byte-exact, with all backslashes intact
    assert pairs[0][1].esc is False


def test_payload_with_u2028_roundtrips_exactly(tmp_path: Path):
    """Regression test for finding 2: U+2028 should not cause over-split."""
    payload = '{"text":"before\u2028after"}'  # U+2028 line separator
    w = RawWriter(tmp_path, "test", "trades", "SYMBOL")
    w.append(payload, t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "test", "trades", "SYMBOL", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert len(pairs) == 1  # Should not split on U+2028
    assert pairs[0][0] == payload  # byte-exact roundtrip
    assert pairs[0][1].esc is False


def test_read_pair_detects_length_mismatch_truncated_idx(tmp_path: Path):
    """Test for finding 4: read_pair raises when idx file is truncated."""
    w = RawWriter(tmp_path, "test", "trades", "SYMBOL")
    w.append('{"frame":0}', t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.append('{"frame":1}', t_recv_ns=1785648600_000_000_001, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "test", "trades", "SYMBOL", "2026-08-02T05")

    # Truncate the idx file to simulate a mid-write failure
    # (delete it entirely to simulate the idx write failing completely)
    idx.unlink()
    # Write an empty compressed file to idx so it's valid but empty
    import zstandard
    cctx = zstandard.ZstdCompressor(level=3)
    with open(idx, "wb") as fh:
        with cctx.stream_writer(fh) as z:
            z.write(b"")  # Empty file

    # read_pair should raise PairLengthMismatch, not silently truncate
    with pytest.raises(PairLengthMismatch) as exc_info:
        read_pair(raw, idx)

    # Verify exception carries both line counts and file paths
    err = exc_info.value
    assert err.raw_count == 2
    assert err.idx_count == 0
    assert err.raw_path == raw
    assert err.idx_path == idx
    assert "2" in str(err)  # raw count in message
    assert "0" in str(err)  # idx count in message


def test_read_pair_exception_includes_counts_and_paths(tmp_path: Path):
    """Test that PairLengthMismatch exception provides actionable diagnostics."""
    w = RawWriter(tmp_path, "venue", "stream", "SYMBOL")
    w.append('{"data":1}', t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.append('{"data":2}', t_recv_ns=1785648600_000_000_001, t_exch_ms=None, seq=None)
    w.append('{"data":3}', t_recv_ns=1785648600_000_000_002, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "venue", "stream", "SYMBOL", "2026-08-02T05")

    # Create a truncated idx file with only 1 entry (simulates write failing after 1st frame)
    import zstandard
    cctx = zstandard.ZstdCompressor(level=3)
    with open(idx, "wb") as fh:
        with cctx.stream_writer(fh) as z:
            z.write(b'{"n":0,"t_recv_ns":1785648600000000000,"t_exch_ms":null,"seq":null,"kind":"data","esc":false}\n')

    with pytest.raises(PairLengthMismatch) as exc_info:
        read_pair(raw, idx)

    err = exc_info.value
    assert err.raw_count == 3
    assert err.idx_count == 1
    assert "reconcile_pair" in str(err).lower()  # Message should reference the repair function
