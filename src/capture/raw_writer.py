"""Writes venue frames verbatim, one per line, with a parallel index sidecar."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import zstandard

from capture.frame_codec import IndexEntry, encode_index_entry, escape_payload, decode_index_entry, unescape_payload


def hour_key(ts_ns: int) -> str:
    moment = dt.datetime.fromtimestamp(ts_ns // 1_000_000_000, tz=dt.timezone.utc)
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
        self._raw_fh = None
        self._idx_fh = None
        self._raw_z = None
        self._idx_z = None
        try:
            self._raw_fh = open(raw_path, "wb")
            try:
                self._idx_fh = open(idx_path, "wb")
                try:
                    self._raw_z = zstandard.ZstdCompressor(level=3).stream_writer(self._raw_fh)
                    try:
                        self._idx_z = zstandard.ZstdCompressor(level=3).stream_writer(self._idx_fh)
                    except Exception:
                        if self._raw_z is not None:
                            self._raw_z.close()
                        raise
                except Exception:
                    if self._idx_fh is not None:
                        self._idx_fh.close()
                    raise
            except Exception:
                if self._raw_fh is not None:
                    self._raw_fh.close()
                raise
        except Exception:
            self._raw_fh = self._idx_fh = self._raw_z = self._idx_z = None
            raise
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
        # Increment n after raw write succeeds. If idx write fails, raw and idx will
        # be mismatched, but n stays consistent with the number of raw lines written.
        # The exception propagates so the caller sees the failure.
        self._n += 1
        self._idx_z.write((encode_index_entry(entry) + "\n").encode("utf-8"))
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
        try:
            # Try to close both stream writers
            try:
                if self._raw_z is not None:
                    self._raw_z.close()
            finally:
                if self._idx_z is not None:
                    self._idx_z.close()
        finally:
            # Try to close both file handles and reset state
            try:
                if self._raw_fh is not None:
                    self._raw_fh.close()
            finally:
                if self._idx_fh is not None:
                    self._idx_fh.close()
                self._raw_z = self._idx_z = self._raw_fh = self._idx_fh = None
                self._hour = None


def read_pair(raw_path: Path, idx_path: Path) -> list[tuple[str, IndexEntry]]:
    dctx = zstandard.ZstdDecompressor()
    with open(raw_path, "rb") as fh:
        raw_text = dctx.stream_reader(fh).read().decode("utf-8")
        raw_lines = raw_text.rstrip("\n").split("\n") if raw_text else []
    with open(idx_path, "rb") as fh:
        idx_text = dctx.stream_reader(fh).read().decode("utf-8")
        idx_lines = idx_text.rstrip("\n").split("\n") if idx_text else []
    return [
        (unescape_payload(r) if entry.esc else r, entry)
        for r, i in zip(raw_lines, idx_lines)
        for entry in [decode_index_entry(i)]
    ]
