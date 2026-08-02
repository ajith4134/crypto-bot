"""Gap detection. Deliberately different per venue - see spec 5.4.

Binance depth is a stateful diff stream: a break in the chain corrupts the book
until a REST resync. Hyperliquid l2Book is stateless snapshots: a gap loses an
observation but nothing is corrupted, so only staleness can be detected.
"""
from __future__ import annotations

import statistics
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

    Fires at `multiple` x the rolling median inter-frame gap, floored at
    `floor_seconds` so fast streams do not alarm on ordinary jitter.
    """

    def __init__(self, floor_seconds: float = 5.0, multiple: float = 10.0,
                 window: int = 200) -> None:
        self._floor_ns = int(floor_seconds * 1e9)
        self._multiple = multiple
        self._gaps: deque[int] = deque(maxlen=window)
        self._last_ns: int | None = None

    def check(self, t_recv_ns: int) -> GapReport | None:
        last, self._last_ns = self._last_ns, t_recv_ns
        if last is None:
            return None
        gap = t_recv_ns - last
        report = None
        if len(self._gaps) >= 10:
            threshold = max(self._floor_ns,
                            int(statistics.median(self._gaps) * self._multiple))
            if gap > threshold:
                report = GapReport(SEVERITY_OBSERVATION_LOSS, {
                    "gap_seconds": gap / 1e9,
                    "threshold_seconds": threshold / 1e9,
                })
        self._gaps.append(gap)
        return report
