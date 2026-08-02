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


def test_binance_malformed_message_does_not_disarm_detection():
    """A malformed message missing 'u' should not reset state."""
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})  # Good message
    t.check({"U": 21})  # Malformed - missing u
    report = t.check({"U": 999, "u": 1010, "pu": 998})  # Gap should be detected
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING


def test_hyperliquid_long_gap_during_warmup_is_detected():
    """A long gap as the second interval should be detected, not absorbed."""
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    t.check(base)  # First timestamp
    # 500s gap as the very next interval (well above floor)
    report = t.check(base + 500 * S)
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 500


def test_hyperliquid_stalls_do_not_poison_median():
    """Stalls should not be included in the cadence learning window."""
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S

    # Start with first timestamp
    t.check(base)

    # Six genuine 50s stalls during initial learning
    for i in range(1, 7):
        report = t.check(base + i * 60 * S)  # 60s apart
        assert report is not None, f"Stall {i} should be flagged"

    # Six normal 1s intervals to establish cadence
    for i in range(6):
        t.check(base + 7 * 60 * S + i * S)

    # Another stall should still be detected despite the prior burst
    report = t.check(base + 7 * 60 * S + 6 * S + 40 * S)  # 40s gap
    assert report is not None, "Stall after burst should be detected"
    assert report.severity == SEVERITY_OBSERVATION_LOSS
