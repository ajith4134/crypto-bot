from pathlib import Path

from capture.universe_tracker import UniverseTracker, diff_universe

TS = 1785648600_000_000_000


def test_diff_detects_listing_and_delisting():
    events = diff_universe(["BTCUSDT", "OLDUSDT"], ["BTCUSDT", "NEWUSDT"],
                           "binance", TS)
    kinds = {(e.symbol, e.kind) for e in events}
    assert ("NEWUSDT", "listed") in kinds
    assert ("OLDUSDT", "delisted") in kinds
    assert ("BTCUSDT", "listed") not in kinds


def test_first_snapshot_lists_everything_as_listed(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    events = tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    assert {e.kind for e in events} == {"listed"}
    assert len(events) == 2


def test_second_snapshot_only_reports_changes(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    events = tracker.record_snapshot(["BTCUSDT", "SOLUSDT"], TS + 1)
    kinds = {(e.symbol, e.kind) for e in events}
    assert kinds == {("SOLUSDT", "listed"), ("ETHUSDT", "delisted")}


def test_snapshot_is_persisted_and_reloadable(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT"], TS)
    reloaded = UniverseTracker(tmp_path, "binance")
    assert reloaded.load_last(TS) == ["BTCUSDT"]
