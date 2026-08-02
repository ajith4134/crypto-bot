from capture.sequencing import BinanceDepthTracker, HyperliquidStalenessTracker
from capture.capture_ledger import SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS

S = 1_000_000_000  # one second in ns


def test_binance_first_frame_is_not_a_gap():
    t = BinanceDepthTracker()
    assert t.check({"U": 10, "u": 20, "pu": 9}) is None


def test_binance_contiguous_chain_has_no_gap():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})
    assert t.check({"U": 21, "u": 30, "pu": 20}) is None


def test_binance_broken_chain_is_corrupting():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})
    report = t.check({"U": 40, "u": 50, "pu": 39})
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING
    assert report.detail["expected_pu"] == 20
    assert report.detail["got_pu"] == 39


def test_binance_spot_without_pu_uses_u_chain():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20})
    assert t.check({"U": 21, "u": 30}) is None
    report = t.check({"U": 99, "u": 120})
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING


def test_hyperliquid_learns_cadence_then_flags_stall():
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(20):
        assert t.check(base + i * S) is None          # steady 1s cadence
    report = t.check(base + 20 * S + 60 * S)          # 60s later
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 60


def test_hyperliquid_floor_prevents_false_alarm_on_fast_streams():
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(20):
        t.check(base + int(i * 0.01 * S))             # 10ms cadence
    # 1s gap: 100x the median, but under the 5s floor -> not an alarm
    assert t.check(base + int(20 * 0.01 * S) + S) is None
