"""Gap detection. Deliberately different per venue - see spec 5.4.

Binance depth is a stateful diff stream: a break in the chain corrupts the book
until a REST resync. Hyperliquid l2Book is stateless snapshots: a gap loses an
observation but nothing is corrupted, so only staleness can be detected.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from capture.capture_ledger import SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS


@dataclass(frozen=True)
class GapReport:
    severity: str
    detail: dict


class BinanceDepthTracker:
    """Validates the U/u/pu chain. Uses pu when present (futures), else u (spot)."""

    def __init__(self) -> None:
        self._last_u: int | None = None

    def check(self, parsed: dict) -> GapReport | None:
        first_id, final_id = parsed.get("U"), parsed.get("u")
        prev_final = parsed.get("pu")

        # If u is missing or None, this message is malformed.
        # Don't touch state, just skip it.
        if final_id is None:
            return None

        last_u, self._last_u = self._last_u, final_id
        if last_u is None:
            return None

        if prev_final is not None:
            if prev_final == last_u:
                return None
            return GapReport(SEVERITY_CORRUPTING,
                             {"expected_pu": last_u, "got_pu": prev_final})

        if first_id == last_u + 1:
            return None
        return GapReport(SEVERITY_CORRUPTING,
                         {"expected_U": last_u + 1, "got_U": first_id})


class HyperliquidStalenessTracker:
    """No sequence numbers exist, so cadence is learned and stalls are inferred.

    Fires at `multiple` x the learned inter-frame cadence, floored at
    `floor_seconds` so fast streams do not alarm on ordinary jitter.

    Two details make the learning work on a stream whose natural cadence is
    *slower* than the floor - an illiquid l2Book updating every 8s against a 5s
    floor:

    Warmup learns from every gap, alarming ones included. Refusing to learn from
    a gap above the floor is self-defeating there: no gap ever qualifies, the
    window never fills, the threshold stays pinned at the floor, and every single
    frame is reported as an observation loss. That both floods the ledger and
    makes a genuine 30-minute outage indistinguishable from the noise, since it
    arrives with the same severity and shape as the false alarms around it. Once
    `min_samples` gaps are in hand a baseline exists, and from then on stalls are
    kept out of the window as before.

    Cadence is estimated from a low quantile of the window rather than its
    median. Stalls only ever push gaps upward, so the fast end of the
    distribution is where the true cadence lives; a low quantile survives a
    window that warmup filled with stalls, where a median would be dragged up by
    them and blind the tracker to later outages.
    """

    def __init__(self, floor_seconds: float = 5.0, multiple: float = 10.0,
                 window: int = 200, min_samples: int = 10,
                 cadence_quantile: float = 0.25) -> None:
        self._floor_ns = int(floor_seconds * 1e9)
        self._multiple = multiple
        self._min_samples = min_samples
        self._cadence_quantile = cadence_quantile
        self._gaps: deque[int] = deque(maxlen=window)
        self._last_ns: int | None = None

    def _estimate_cadence_ns(self) -> int:
        ordered = sorted(self._gaps)
        index = min(int(len(ordered) * self._cadence_quantile), len(ordered) - 1)
        return ordered[index]

    def has_baseline(self) -> bool:
        return len(self._gaps) >= self._min_samples

    def check(self, t_recv_ns: int) -> GapReport | None:
        last, self._last_ns = self._last_ns, t_recv_ns
        if last is None:
            return None
        gap = t_recv_ns - last

        # Before a baseline exists there is nothing to compare against, so the
        # floor is the only usable threshold.
        if self.has_baseline():
            threshold = max(self._floor_ns,
                            int(self._estimate_cadence_ns() * self._multiple))
        else:
            threshold = self._floor_ns

        report = None
        if gap > threshold:
            report = GapReport(SEVERITY_OBSERVATION_LOSS, {
                "gap_seconds": gap / 1e9,
                "threshold_seconds": threshold / 1e9,
            })

        # During warmup every gap is learned from, or a stream slower than the
        # floor could never establish a baseline. Once one exists, stalls are
        # anomalies and stay out of the learning window.
        if report is None or not self.has_baseline():
            self._gaps.append(gap)

        return report
