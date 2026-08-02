"""Writes venue frames verbatim, one per line, with a parallel index sidecar."""
from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import zstandard

from capture.frame_codec import IndexEntry, encode_index_entry, escape_payload, decode_index_entry, unescape_payload


class PairLengthMismatch(Exception):
    """Raised when raw and index files have mismatched line counts.

    This indicates a partial write failure: the raw file received a frame but the
    corresponding index entry failed to write. The files are corrupt and cannot be
    safely read without repair.

    Task 4's `reconcile_pair` function exists to rebuild missing index entries from
    the raw file. A silent truncation would hide the exact condition it is designed
    to fix, so this exception is raised instead of silently returning a truncated list.
    """

    def __init__(self, raw_path: Path, idx_path: Path, raw_count: int, idx_count: int) -> None:
        self.raw_path = raw_path
        self.idx_path = idx_path
        self.raw_count = raw_count
        self.idx_count = idx_count
        super().__init__(
            f"Pair length mismatch: {raw_path} has {raw_count} lines, "
            f"{idx_path} has {idx_count} lines. Run reconcile_pair to repair."
        )


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
        # Encode the index entry BEFORE writing anything. If encode_index_entry
        # raises, no data is written to either file. Avoids creating a mismatch.
        idx_line = encode_index_entry(entry)
        # Now that encoding succeeded, write to both files. If raw write fails,
        # neither file is affected. If idx write fails, raw has the frame but idx
        # doesn't; read_pair will detect and raise PairLengthMismatch.
        self._raw_z.write((escaped + "\n").encode("utf-8"))
        # Increment n immediately after raw write succeeds. If idx write fails,
        # self._n stays consistent with the number of raw lines written, so the
        # next append() will get a fresh n value and not reuse a duplicate.
        self._n += 1
        self._idx_z.write((idx_line + "\n").encode("utf-8"))
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


def _read_lines(path: Path) -> list[str]:
    """Split strictly on newline.

    NOT str.splitlines(): that also splits on \v, \f, \x1c-\x1e, \x85,
    U+2028 and U+2029, none of which escape_payload guards. Such a payload
    would yield an extra raw line with no matching index entry, and every
    subsequent line would pair with the wrong entry.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        text = dctx.stream_reader(fh).read().decode("utf-8")
    return text.rstrip("\n").split("\n") if text else []


def _write_lines(path: Path, lines: list[str]) -> None:
    """Write lines atomically using a temporary file.

    A crash during write leaves the original file untouched; any reader or
    retry sees either the old content or the new content, never a partial
    or corrupted frame.
    """
    cctx = zstandard.ZstdCompressor(level=3)
    # Write to a temporary file in the same directory so os.replace() is atomic
    fd, tmpfile = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            with cctx.stream_writer(fh) as w:
                for line in lines:
                    w.write((line + "\n").encode("utf-8"))
        # Atomic replacement: any reader sees either old or new, never partial
        os.replace(tmpfile, path)
    except Exception:
        os.unlink(tmpfile)
        raise


def read_pair(raw_path: Path, idx_path: Path) -> list[tuple[str, IndexEntry]]:
    """Read a raw/index pair, returning (payload, entry) tuples.

    Raises PairLengthMismatch if the files have different line counts (indicating
    a partial write failure that reconcile_pair is designed to repair).

    For recovered entries (kind=="recovered"), the escape state is unrecoverable
    from the raw line alone, so the payload is returned as stored and may still
    be in escaped form. This is why kind="recovered" exists — so downstream can
    exclude these entries explicitly and handle them as needed.
    """
    raw_lines = _read_lines(raw_path)
    idx_lines = _read_lines(idx_path)

    # Detect and refuse to paper over partial write failures. If raw and idx have
    # different line counts, it means a frame was written but its index entry was not.
    # This is the exact condition that Task 4's reconcile_pair is designed to repair.
    if len(raw_lines) != len(idx_lines):
        raise PairLengthMismatch(Path(raw_path), Path(idx_path), len(raw_lines), len(idx_lines))

    result = []
    for r, i in zip(raw_lines, idx_lines):
        entry = decode_index_entry(i)
        # For recovered entries, the escape state is unknown, so return as-stored
        # (which may still be escaped). For data entries, unescape if marked.
        if entry.kind == "recovered":
            payload = r
        else:
            payload = unescape_payload(r) if entry.esc else r
        result.append((payload, entry))
    return result


def reconcile_pair(raw_path: Path, idx_path: Path) -> int:
    """Rebuild index entries for raw lines a crash left undescribed.

    Returns the number of entries repaired. Never discards raw data, and never
    invents a receipt timestamp - unknown times are recorded as 0 with
    kind="recovered" so downstream can exclude them explicitly.

    Limitation: The escape state (whether a payload was escaped) is unrecoverable
    from the raw line alone. Given only stored bytes, you cannot distinguish
    "original contained a real newline, was escaped" from "original literally
    contained backslash-then-n and was not escaped". Both produce identical disk
    bytes. Therefore, read_pair returns recovered entries' payloads as stored,
    which may still be in escaped form. This is why kind="recovered" exists —
    downstream must exclude or handle these entries explicitly.
    """
    raw_lines = _read_lines(raw_path)
    idx_lines = _read_lines(idx_path)
    if len(idx_lines) >= len(raw_lines):
        return 0

    repaired = 0
    for n in range(len(idx_lines), len(raw_lines)):
        entry = IndexEntry(n=n, t_recv_ns=0, t_exch_ms=None,
                           seq=None, kind="recovered", esc=False)
        idx_lines.append(encode_index_entry(entry))
        repaired += 1
    _write_lines(idx_path, idx_lines)
    return repaired
