"""Answers 'is capture healthy?' with evidence rather than assumption.

Runway is measured in days remaining, not percent used, because percent
thresholds mean nothing when the write rate changes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from capture.capture_ledger import read_all

WARN_DAYS = 30.0
ALERT_DAYS = 14.0
DECISION_DAYS = 7.0


def compute_runway_days(free_bytes: int, daily_bytes: float) -> float:
    if daily_bytes <= 0:
        return float("inf")
    return free_bytes / daily_bytes


def classify_runway(days: float) -> str:
    if days <= DECISION_DAYS:
        return "decision_point"
    if days <= ALERT_DAYS:
        return "alert"
    if days <= WARN_DAYS:
        return "warn"
    return "ok"


def measure_daily_bytes(root: Path, days: int = 7) -> float:
    raw_root = Path(root) / "raw"
    if not raw_root.exists():
        return 0.0
    cutoff = time.time() - days * 86400
    total = sum(
        path.stat().st_size
        for path in raw_root.rglob("*.zst")
        if path.stat().st_mtime >= cutoff
    )
    return total / days


def build_report(root: Path, venue: str, date: str,
                 free_bytes: int, daily_bytes: float) -> dict:
    events = read_all(root, venue, date)
    gaps = {"corrupting": 0, "observation_loss": 0, "info": 0}
    for event in events:
        if event.kind == "gap":
            gaps[event.severity] = gaps.get(event.severity, 0) + 1

    runway = compute_runway_days(free_bytes, daily_bytes)
    return {
        "venue": venue,
        "date": date,
        "events_total": len(events),
        "gaps": gaps,
        "free_bytes": free_bytes,
        "daily_bytes": daily_bytes,
        "runway_days": runway,
        "runway_status": classify_runway(runway),
    }


def write_alerts(root: Path, report: dict) -> int:
    alerts = []
    if report["runway_status"] != "ok":
        alerts.append({"reason": f"runway_{report['runway_status']}",
                       "runway_days": report["runway_days"]})
    if report["gaps"].get("corrupting", 0) > 0:
        alerts.append({"reason": "corrupting_gaps",
                       "count": report["gaps"]["corrupting"]})

    if alerts:
        folder = Path(root) / "health"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "alerts.ndjson", "a", encoding="utf-8") as fh:
            for alert in alerts:
                fh.write(json.dumps({**alert, "venue": report["venue"],
                                     "date": report["date"]},
                                    separators=(",", ":"), sort_keys=True) + "\n")
    return len(alerts)
