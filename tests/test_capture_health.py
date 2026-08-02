import json
from pathlib import Path

from capture.capture_health import (
    compute_runway_days, classify_runway, build_report, write_alerts,
)
from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, SEVERITY_CORRUPTING,
)

TS = 1785648600_000_000_000


def test_runway_is_free_over_daily():
    assert compute_runway_days(100_000_000_000, 2_000_000_000) == 50.0


def test_runway_is_infinite_when_nothing_written():
    assert compute_runway_days(100, 0) == float("inf")


def test_classify_thresholds():
    assert classify_runway(90) == "ok"
    assert classify_runway(25) == "warn"
    assert classify_runway(10) == "alert"
    assert classify_runway(5) == "decision_point"


def test_report_counts_gaps_by_severity(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "depth", "gap",
                              SEVERITY_CORRUPTING, {"symbol": "BTCUSDT"}))
    ledger.close()

    report = build_report(tmp_path, "binance", "2026-08-02",
                          free_bytes=100_000_000_000, daily_bytes=2_000_000_000)
    assert report["gaps"]["corrupting"] == 1
    assert report["runway_days"] == 50.0
    assert report["runway_status"] == "ok"


def test_write_alerts_emits_lines_for_bad_states(tmp_path: Path):
    report = build_report(tmp_path, "binance", "2026-08-02",
                          free_bytes=4_000_000_000, daily_bytes=2_000_000_000)
    count = write_alerts(tmp_path, report)
    assert count == 1
    lines = (tmp_path / "health" / "alerts.ndjson").read_text().splitlines()
    assert json.loads(lines[0])["reason"] == "runway_decision_point"
