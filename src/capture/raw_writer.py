"""Writes venue frames verbatim, one per line, with a parallel index sidecar."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import zstandard

from capture.frame_codec import IndexEntry, encode_index_entry, escape_payload, decode_index_entry, unescape_payload


def hour_key(ts_ns: int) -> str:
    moment = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H")


def paths_for(root: Path, venue: str, stream: str, symbol: str, hour: str) -> tuple[Path, Path]:
    date = hour.split("T")[0]
    folder = Path(root) / "raw" / venue / date
    stem = f"{stream}_{symbol}_{hour}"
    return folder / f"{stem}.ndjson.zst", folder / f"{stem}.idx.zst"


class RawWriter:
    """Append-only writer for one (venue, stream, symbol). Rotates hourly by UTC."""

    def __init__(self, root: Path, venue: str, stream: str, symbol: str) -> None:
        self._root = Path(root)
        self._venue, self._stream, self._symbol = venue, stream, symbol
        self._hour: str | None = None
        self._raw_fh = self._idx_fh = None
        self._raw_z = self._idx_z = None
        self._n = 0

    def _open(self, hour: str) -> None:
        raw_path, idx_path = paths_for(self._root, self._venue, self._stream, self._symbol, hour)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_fh = open(raw_path, "wb")
        self._idx_fh = open(idx_path, "wb")
        self._raw_z = zstandard.ZstdCompressor(level=3).stream_writer(self._raw_fh)
        self._idx_z = zstandard.ZstdCompressor(level=3).stream_writer(self._idx_fh)
        self._hour, self._n = hour, 0

    def append(self, payload: str, t_recv_ns: int, t_exch_ms: int | None,
               seq: dict | None, kind: str = "data") -> int:
        hour = hour_key(t_recv_ns)
        if hour != self._hour:
            self.close()
            self._open(hour)

        escaped, was_escaped = escape_payload(payload)
        entry = IndexEntry(n=self._n, t_recv_ns=t_recv_ns, t_exch_ms=t_exch_ms,
                           seq=seq, kind=kind, esc=was_escaped)
        self._raw_z.write((escaped + "\n").encode("utf-8"))
        self._idx_z.write((encode_index_entry(entry) + "\n").encode("utf-8"))
        self._n += 1
        return entry.n

    def flush(self) -> None:
        if self._raw_z is not None:
            self._raw_z.flush(zstandard.FLUSH_FRAME)
            self._idx_z.flush(zstandard.FLUSH_FRAME)
            self._raw_fh.flush()
            self._idx_fh.flush()

    def close(self) -> None:
        if self._raw_z is None:
            return
        self._raw_z.close(); self._idx_z.close()
        self._raw_fh.close(); self._idx_fh.close()
        self._raw_z = self._idx_z = self._raw_fh = self._idx_fh = None
        self._hour = None


def read_pair(raw_path: Path, idx_path: Path) -> list[tuple[str, IndexEntry]]:
    dctx = zstandard.ZstdDecompressor()
    with open(raw_path, "rb") as fh:
        raw_lines = dctx.stream_reader(fh).read().decode("utf-8").splitlines()
    with open(idx_path, "rb") as fh:
        idx_lines = dctx.stream_reader(fh).read().decode("utf-8").splitlines()
    return [
        (unescape_payload(r), decode_index_entry(i))
        for r, i in zip(raw_lines, idx_lines)
    ]
