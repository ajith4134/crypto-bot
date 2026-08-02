from pathlib import Path
from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, read_all,
    SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS,
)


def test_records_and_reads_back(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(
        ts_ns=1785648600_000_000_000, venue="binance", stream="depth",
        kind="gap", severity=SEVERITY_CORRUPTING,
        detail={"symbol": "BTCUSDT", "expected_pu": 5, "got_pu": 9},
    ))
    ledger.close()

    events = read_all(tmp_path, "binance", "2026-08-02")
    assert len(events) == 1
    assert events[0].kind == "gap"
    assert events[0].severity == SEVERITY_CORRUPTING
    assert events[0].detail["expected_pu"] == 5


def test_appends_across_multiple_records(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "hyperliquid")
    for i in range(3):
        ledger.record(LedgerEvent(
            ts_ns=1785648600_000_000_000 + i, venue="hyperliquid",
            stream="l2Book", kind="stale", severity=SEVERITY_OBSERVATION_LOSS,
            detail={"symbol": "BTC"},
        ))
    ledger.close()
    assert len(read_all(tmp_path, "hyperliquid", "2026-08-02")) == 3


def test_handles_non_serializable_types_in_detail(tmp_path: Path):
    """Bytes and other non-JSON types degrade to string, not lost."""
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(
        ts_ns=1785648600_000_000_000, venue="binance", stream="depth",
        kind="malformed", severity=SEVERITY_CORRUPTING,
        detail={"payload": b"corrupted_bytes_data", "symbol": "BTCUSDT"},
    ))
    ledger.close()

    events = read_all(tmp_path, "binance", "2026-08-02")
    assert len(events) == 1
    # The bytes should be stringified, not lost
    assert "payload" in events[0].detail
    assert isinstance(events[0].detail["payload"], str)
    assert "corrupted_bytes_data" in events[0].detail["payload"]


def test_midnight_rollover_routes_to_correct_date(tmp_path: Path):
    """Events straddling UTC midnight land in their respective date files."""
    ledger = CaptureLedger(tmp_path, "kraken")
    # Reference: 1785648600_000_000_000 ns = 2026-08-02T05:30:00Z
    # Last moment of 2026-08-02: reference + (18.5 hours - 1 second) = 2026-08-02T23:59:59Z
    ts_aug_2_last = 1785648600_000_000_000 + 66599_000_000_000
    # First moment of 2026-08-03: reference + 24 hours = 2026-08-03T05:30:00Z
    ts_aug_3_first = 1785648600_000_000_000 + 86400_000_000_000

    ledger.record(LedgerEvent(
        ts_ns=ts_aug_2_last, venue="kraken", stream="trades",
        kind="late", severity=SEVERITY_OBSERVATION_LOSS,
        detail={"symbol": "BTC"},
    ))
    ledger.record(LedgerEvent(
        ts_ns=ts_aug_3_first, venue="kraken", stream="trades",
        kind="early", severity=SEVERITY_OBSERVATION_LOSS,
        detail={"symbol": "BTC"},
    ))
    ledger.close()

    events_aug_2 = read_all(tmp_path, "kraken", "2026-08-02")
    events_aug_3 = read_all(tmp_path, "kraken", "2026-08-03")

    assert len(events_aug_2) == 1
    assert len(events_aug_3) == 1
    assert events_aug_2[0].kind == "late"
    assert events_aug_3[0].kind == "early"
