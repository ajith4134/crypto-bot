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


def test_hyperliquid_stream_slower_than_the_floor_learns_its_cadence():
    """An illiquid l2Book updating every 8s against a 5s floor must not alarm forever.

    Refusing to learn from any gap above the floor meant the window could never
    fill on such a stream: the threshold stayed pinned at the floor and every
    single frame became an observation_loss event. That floods the ledger and,
    worse, makes a genuine outage arrive with the same severity and shape as the
    hundreds of false ones around it.
    """
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    cadence_seconds = 8

    reports = [t.check(base + i * cadence_seconds * S) for i in range(200)]
    alarms = [r for r in reports if r is not None]

    assert len(alarms) <= 10, f"{len(alarms)} false alarms on a steady 8s stream"
    assert reports[-1] is None, "still alarming on ordinary cadence after 200 frames"
    assert t.has_baseline(), "the tracker never learned a cadence"


def test_hyperliquid_slow_stream_still_flags_a_real_outage():
    """Learning a slow cadence must not cost the detection it exists for."""
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(200):
        t.check(base + i * 8 * S)

    last = base + 199 * 8 * S
    report = t.check(last + 30 * 60 * S)          # a 30-minute outage
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 1800
    # The threshold is derived from the learned 8s cadence, not the 5s floor.
    assert report.detail["threshold_seconds"] > 5.0


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
