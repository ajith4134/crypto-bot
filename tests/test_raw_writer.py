from pathlib import Path
from capture.raw_writer import RawWriter, read_pair, hour_key, paths_for


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
