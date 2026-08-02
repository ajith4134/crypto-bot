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
