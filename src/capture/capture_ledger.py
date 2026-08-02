"""Records capture anomalies as first-class, queryable events.

Plain NDJSON, uncompressed: this file is read during incidents, and volume is
tiny compared to market data.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SEVERITY_INFO = "info"
SEVERITY_OBSERVATION_LOSS = "observation_loss"
SEVERITY_CORRUPTING = "corrupting"


@dataclass(frozen=True)
class LedgerEvent:
    ts_ns: int
    venue: str
    stream: str
    kind: str
    severity: str
    detail: dict


def _date_of(ts_ns: int) -> str:
    return dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def _path_for(root: Path, venue: str, date: str) -> Path:
    return Path(root) / "ledger" / venue / date / "events.ndjson"


class CaptureLedger:
    def __init__(self, root: Path, venue: str) -> None:
        self._root, self._venue = Path(root), venue
        self._fh = None
        self._date: str | None = None

    def record(self, event: LedgerEvent) -> None:
        date = _date_of(event.ts_ns)
        if date != self._date:
            self.close()
            path = _path_for(self._root, self._venue, date)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            self._date = date
        self._fh.write(json.dumps(asdict(event), separators=(",", ":"), sort_keys=True, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._date = None


def read_all(root: Path, venue: str, date: str) -> list[LedgerEvent]:
    path = _path_for(root, venue, date)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [LedgerEvent(**json.loads(line)) for line in fh if line.strip()]
