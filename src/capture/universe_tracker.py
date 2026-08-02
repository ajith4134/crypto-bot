"""Records point-in-time universe membership.

Without this, any backtest over "all symbols" silently conditions on survival.
Exchanges do not reliably publish historical membership, so it must be captured
as it happens.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class UniverseEvent:
    ts_ns: int
    venue: str
    symbol: str
    kind: str
    detail: dict


def _date_of(ts_ns: int) -> str:
    return dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def diff_universe(previous: list[str], current: list[str],
                  venue: str, ts_ns: int) -> list[UniverseEvent]:
    before, after = set(previous), set(current)
    events = [UniverseEvent(ts_ns, venue, s, "listed", {}) for s in sorted(after - before)]
    events += [UniverseEvent(ts_ns, venue, s, "delisted", {}) for s in sorted(before - after)]
    return events


class UniverseTracker:
    def __init__(self, root: Path, venue_name: str) -> None:
        self._root = Path(root)
        self._venue = venue_name

    def _dir_for(self, ts_ns: int) -> Path:
        return self._root / "universe" / self._venue / _date_of(ts_ns)

    def _state_path(self) -> Path:
        return self._root / "universe" / self._venue / "last_snapshot.json"

    def load_last(self, ts_ns: int) -> list[str]:
        path = self._state_path()
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))["symbols"]

    def record_snapshot(self, symbols: list[str], ts_ns: int) -> list[UniverseEvent]:
        previous = self.load_last(ts_ns)
        events = diff_universe(previous, symbols, self._venue, ts_ns)

        folder = self._dir_for(ts_ns)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "instruments.ndjson", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts_ns": ts_ns, "kind": "snapshot",
                                 "symbols": symbols}, separators=(",", ":")) + "\n")
            for event in events:
                fh.write(json.dumps(asdict(event), separators=(",", ":"),
                                    sort_keys=True) + "\n")

        state = self._state_path()
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"ts_ns": ts_ns, "symbols": symbols}),
                         encoding="utf-8")
        return events
