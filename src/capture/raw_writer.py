"""Writes venue frames verbatim, one per line, with a parallel index sidecar."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import zstandard

from capture.frame_codec import IndexEntry, encode_index_entry, escape_payload, decode_index_entry, unescape_payload

RAW_SUFFIX = ".ndjson.zst"
IDX_SUFFIX = ".idx.zst"
WRITING_MARKER_SUFFIX = ".writing"
QUARANTINE_SUFFIX = ".quarantine"


class RawCaptureError(Exception):
    """Base for every damage a raw/index pair can be found in.

    Callers that only need "is this pair usable?" catch this; callers that need
    to tell damage apart catch the specific subclasses below.
    """


class PairLengthMismatch(RawCaptureError):
    """Raised when raw and index files have mismatched line counts.

    This indicates a partial write failure: the raw file received a frame but the
    corresponding index entry failed to write. The files are corrupt and cannot be
    safely read without repair.

    `reconcile_pair` exists to rebuild missing index entries from the raw file. A
    silent truncation would hide the exact condition it is designed to fix, so this
    exception is raised instead of silently returning a truncated list.
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


class TruncatedFrameFile(RawCaptureError):
    """Raised when a .zst file's last frame is incomplete, or its last line is.

    `ZstdDecompressor.stream_reader(...).read()` does NOT raise on a truncated
    frame - it returns whatever prefix it could decode. A `kill -9` or a power
    loss produces exactly that shape, and the decodable prefix ends mid-payload,
    so the final line comes back looking like a complete frame when it is half of
    one. Silently accepting the prefix is indistinguishable from success, so the
    damage is raised instead.

    `recovered_lines` carries the complete lines decoded before the damage, so an
    operator (or `reconcile_pair`) can salvage the intact prefix deliberately
    rather than by accident.
    """

    def __init__(self, path: Path, reason: str, recovered_lines: list[str]) -> None:
        self.path = path
        self.reason = reason
        self.recovered_lines = recovered_lines
        super().__init__(
            f"{path} is truncated ({reason}); "
            f"{len(recovered_lines)} complete lines recovered before the damage."
        )


class IndexPositionMismatch(RawCaptureError):
    """Raised when an index entry's `n` does not match its line position.

    `n` is the authority on which raw line an index entry describes. Pairing raw
    line i with index line i positionally - without checking `n` - hands back a
    frame's bytes labelled with a different frame's receipt time and sequence
    numbers whenever an entry is missing from the middle. A plausible wrong
    timestamp is worse than an obviously missing one, so this is refused.
    """

    def __init__(self, raw_path: Path, idx_path: Path, position: int, entry_n: int) -> None:
        self.raw_path = raw_path
        self.idx_path = idx_path
        self.position = position
        self.entry_n = entry_n
        super().__init__(
            f"Index position mismatch in {idx_path}: line {position} carries n={entry_n}. "
            f"An index entry is missing from the middle of the file. "
            f"Run reconcile_pair to repair."
        )


class UnrepairableIndex(RawCaptureError):
    """Raised when `reconcile_pair` cannot rebuild the index from the raw file.

    Repair only ever adds entries for raw lines that exist. An index that
    describes frames the raw file does not hold, or whose `n` values do not
    ascend, cannot be reconciled without inventing or discarding data.
    """

    def __init__(self, raw_path: Path, idx_path: Path, reason: str) -> None:
        self.raw_path = raw_path
        self.idx_path = idx_path
        self.reason = reason
        super().__init__(f"Cannot reconcile {idx_path} against {raw_path}: {reason}")


class MissingPairFile(RawCaptureError):
    """Raised when `reconcile_pair` is given a path that does not exist.

    A bare `FileNotFoundError` escapes `RawCaptureError`, so a caller catching
    the documented base class - the whole point of that base class - misses it
    and treats an absent hour as an unexpected crash.

    A missing *index* is not this error: it is the extreme of the damage repair
    exists to fix, and is rebuilt from the raw file like any other hole.
    """

    def __init__(self, path: Path, role: str) -> None:
        self.path = path
        self.role = role
        super().__init__(f"Cannot reconcile: the {role} file {path} does not exist.")


class HourStillBeingWritten(RawCaptureError):
    """Raised when `reconcile_pair` is asked to repair an hour a writer holds open.

    `reconcile_pair` replaces the index file with `os.replace`. A live `RawWriter`
    still holds the old inode's descriptor, so every entry it writes afterwards
    lands in the orphaned inode and is lost - the repair tool would create a
    mismatch larger than the one it fixed. Repair therefore refuses while the
    hour is live.
    """

    def __init__(self, raw_path: Path, marker_path: Path, pid: int | None) -> None:
        self.raw_path = raw_path
        self.marker_path = marker_path
        self.pid = pid
        super().__init__(
            f"Refusing to repair {raw_path}: a writer is still recording into this hour "
            f"(marker {marker_path}, pid {pid}). Close the writer, or delete the marker "
            f"if you are certain no process holds the file."
        )


class HourFileNotAppendable(RawCaptureError):
    """Raised when a `RawWriter` is asked to resume into a damaged hour.

    Resuming requires knowing how many frames the hour already holds, because
    `n` must keep matching each frame's position in the file. If the existing
    pair is torn or already misaligned, that count is unknowable and appending
    would bury the damage under new data instead of surfacing it.
    """

    def __init__(self, raw_path: Path, idx_path: Path, reason: str) -> None:
        self.raw_path = raw_path
        self.idx_path = idx_path
        self.reason = reason
        super().__init__(
            f"Cannot resume appending to {raw_path}: {reason}. "
            f"Run reconcile_pair to repair the pair first."
        )


def hour_key(ts_ns: int) -> str:
    moment = dt.datetime.fromtimestamp(ts_ns // 1_000_000_000, tz=dt.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H")


def paths_for(root: Path, venue: str, stream: str, symbol: str, hour: str) -> tuple[Path, Path]:
    date = hour.split("T")[0]
    folder = Path(root) / "raw" / venue / date
    stem = f"{stream}_{symbol}_{hour}"
    return folder / f"{stem}{RAW_SUFFIX}", folder / f"{stem}{IDX_SUFFIX}"


def writing_marker_path(raw_path: Path) -> Path:
    """Sidecar a `RawWriter` holds while it has this hour open."""
    raw_path = Path(raw_path)
    name = raw_path.name
    stem = name[: -len(RAW_SUFFIX)] if name.endswith(RAW_SUFFIX) else name
    return raw_path.parent / f"{stem}{WRITING_MARKER_SUFFIX}"


def is_hour_being_written(raw_path: Path) -> tuple[bool, Path, int | None]:
    """Answer whether a live process still holds this hour open.

    Returns (is_live, marker_path, pid). A marker whose pid is gone is stale -
    left by a crash - and does not block repair. A marker that cannot be parsed
    is treated as live, because refusing a repair is recoverable and clobbering a
    live writer's index is not.
    """
    marker = writing_marker_path(raw_path)
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, NotADirectoryError):
        return False, marker, None
    try:
        pid = int(text)
    except ValueError:
        return True, marker, None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, marker, pid
    except PermissionError:
        return True, marker, pid
    return True, marker, pid


class RawWriter:
    """Append-only writer for one (venue, stream, symbol). Rotates hourly by UTC.

    Files are opened with O_APPEND, never O_TRUNC: a process restart, a
    `close()` followed by another `append()`, or an out-of-order timestamp that
    reaches back over an hour boundary must all extend the hour's files rather
    than destroy them. Each open starts a fresh zstd frame appended to the
    previous ones; concatenated frames read back as one stream (verified against
    zstandard 0.25.0).
    """

    def __init__(self, root: Path, venue: str, stream: str, symbol: str) -> None:
        self._root = Path(root)
        self._venue, self._stream, self._symbol = venue, stream, symbol
        self._hour: str | None = None
        self._raw_fh = self._idx_fh = None
        self._raw_z = self._idx_z = None
        self._marker_path: Path | None = None
        self._n = 0

    @staticmethod
    def _count_frames_already_written(raw_path: Path, idx_path: Path) -> int:
        """How many frames this hour already holds, so `n` can resume in step.

        Raises HourFileNotAppendable when that count is unknowable: `n` has to
        keep matching each frame's position in the file, and guessing it would
        make every later entry mislabel its frame.
        """
        raw_exists, idx_exists = raw_path.exists(), idx_path.exists()
        if not raw_exists and not idx_exists:
            return 0
        try:
            raw_lines = _read_lines(raw_path) if raw_exists else []
            idx_lines = _read_lines(idx_path) if idx_exists else []
        except TruncatedFrameFile as exc:
            raise HourFileNotAppendable(raw_path, idx_path, str(exc)) from exc
        if len(raw_lines) != len(idx_lines):
            raise HourFileNotAppendable(
                raw_path, idx_path,
                f"raw holds {len(raw_lines)} frames but the index holds {len(idx_lines)}")
        if idx_lines:
            last_n = decode_index_entry(idx_lines[-1]).n
            if last_n != len(idx_lines) - 1:
                raise HourFileNotAppendable(
                    raw_path, idx_path,
                    f"the last index entry carries n={last_n} at position {len(idx_lines) - 1}, "
                    f"so an entry is missing from the middle")
        return len(raw_lines)

    def _open(self, hour: str) -> None:
        raw_path, idx_path = paths_for(self._root, self._venue, self._stream, self._symbol, hour)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # Read the existing pair BEFORE opening anything for append: a damaged
        # hour must be refused while nothing has been touched.
        resume_n = self._count_frames_already_written(raw_path, idx_path)

        self._raw_fh = None
        self._idx_fh = None
        self._raw_z = None
        self._idx_z = None
        self._marker_path = None
        try:
            self._raw_fh = open(raw_path, "ab")
            try:
                self._idx_fh = open(idx_path, "ab")
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
        marker = writing_marker_path(raw_path)
        try:
            marker.write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            # The marker only guards reconcile_pair; failing to place it must not
            # cost live frames, so the hour is opened without one.
            #
            # Be honest about what that costs: with no marker on disk,
            # is_hour_being_written() reports this live hour as idle and
            # reconcile_pair will repair it, swapping the index inode out from
            # under the descriptor this writer still holds. Every index entry
            # written afterwards lands in the orphaned inode and is lost. The
            # post-repair appendability check in reconcile_pair does not prevent
            # that - it runs before those entries are written, so it can only
            # catch a writer that was already mid-hour, not one that keeps
            # writing after the swap.
            #
            # This is accepted rather than fixed: refusing to open the hour would
            # turn one unwritable sidecar into total capture loss for the stream,
            # and the trigger (ENOSPC or a permission fault on a directory whose
            # raw files just opened successfully) is both rare and one that stops
            # frames landing anyway.
            marker = None
        self._marker_path = marker
        self._hour, self._n = hour, resume_n

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
        # Increment n immediately after raw write succeeds, so the next frame's
        # n matches its position in the raw file. The n this frame would have had
        # is deliberately left unused in the index; reconcile_pair finds that hole
        # by n and inserts the recovered entry at the right position.
        self._n += 1
        self._idx_z.write((idx_line + "\n").encode("utf-8"))
        return entry.n

    def flush(self) -> None:
        # Raw is flushed to completion before the index gains its own frame, for
        # the same reason close() finishes them in that order - see close().
        if self._raw_z is not None:
            self._raw_z.flush(zstandard.FLUSH_FRAME)
            self._raw_fh.flush()
            self._idx_z.flush(zstandard.FLUSH_FRAME)
            self._idx_fh.flush()

    def close(self) -> None:
        """Finish both files, attempting every resource even if one fails.

        Ordering is load-bearing. `reconcile_pair` can rebuild missing index
        entries from raw lines, but nothing can rebuild raw frames from index
        entries - so an interrupted close must never leave the index holding
        frames the raw file does not. Raw is therefore finished completely
        first, and the index stream's footer is written only once raw is safely
        down. If raw's close fails, the index's unfinished frame is deliberately
        abandoned; its descriptor is still closed so nothing leaks.
        """
        if self._raw_fh is None and self._idx_fh is None:
            self._hour = None
            return
        errors: list[Exception] = []

        raw_is_complete = True
        for finish in (self._raw_z, self._raw_fh):
            if finish is None:
                continue
            try:
                finish.close()
            except Exception as exc:
                raw_is_complete = False
                errors.append(exc)

        if raw_is_complete and self._idx_z is not None:
            try:
                self._idx_z.close()
            except Exception as exc:
                errors.append(exc)

        if self._idx_fh is not None:
            try:
                self._idx_fh.close()
            except Exception as exc:
                errors.append(exc)

        if self._marker_path is not None:
            try:
                self._marker_path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(exc)

        self._raw_z = self._idx_z = self._raw_fh = self._idx_fh = None
        self._marker_path = None
        self._hour = None

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("errors while closing raw writer resources", errors)


def _decode_concatenated_frames(path: Path, data: bytes) -> str:
    """Decode every zstd frame in `data`, refusing to accept a truncated tail.

    Each open appends a new frame, so a file holds a run of concatenated frames.
    `stream_reader(...).read()` walks them but returns the decodable prefix of a
    torn frame without complaint; `decompressobj()` exposes `eof`, so an
    unfinished frame is detectable. `ZstdDecompressor.decompress()` is not an
    option here: stream-written frames carry no content size in their header.
    """
    decompressor = zstandard.ZstdDecompressor()
    chunks: list[bytes] = []
    position = 0

    def complete_lines_so_far() -> list[str]:
        text = b"".join(chunks).decode("utf-8", errors="replace")
        lines = text.split("\n")
        return lines[:-1]

    while position < len(data):
        frame_reader = decompressor.decompressobj()
        try:
            chunks.append(frame_reader.decompress(data[position:]))
        except zstandard.ZstdError as exc:
            raise TruncatedFrameFile(
                path, f"zstd frame at byte {position} is unreadable: {exc}",
                complete_lines_so_far()) from exc
        if not frame_reader.eof:
            raise TruncatedFrameFile(
                path, f"zstd frame starting at byte {position} is incomplete",
                complete_lines_so_far())
        consumed = len(data) - position - len(frame_reader.unused_data)
        if consumed <= 0:
            raise TruncatedFrameFile(
                path, f"zstd frame at byte {position} consumed no input",
                complete_lines_so_far())
        position += consumed

    return b"".join(chunks).decode("utf-8")


def _read_lines(path: Path) -> list[str]:
    r"""Split strictly on newline, and strip exactly one trailing newline.

    NOT str.splitlines(): that also splits on \v, \f, \x1c-\x1e, \x85,
    U+2028 and U+2029, none of which escape_payload guards. Such a payload
    would yield an extra raw line with no matching index entry, and every
    subsequent line would pair with the wrong entry.

    NOT rstrip("\n") either: payloads are untrusted network text and an empty
    websocket text frame is legal, so '{"a":1}\n\n' is two frames - one of them
    empty - not one. Stripping every trailing newline collapses them into one
    raw line against two index entries, an unrepairable mismatch caused by a
    frame the venue was entitled to send.

    Every line the writer emits is newline-terminated, so text that does not end
    in a newline is a partial final line and is refused rather than returned as
    a complete frame.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        data = fh.read()
    text = _decode_concatenated_frames(path, data)
    if not text:
        return []
    if not text.endswith("\n"):
        lines = text.split("\n")
        raise TruncatedFrameFile(path, "final line has no terminating newline", lines[:-1])
    return text[:-1].split("\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    """Write lines atomically using a temporary file.

    A crash during write leaves the original file untouched; any reader or
    retry sees either the old content or the new content, never a partial
    or corrupted frame.

    `os.replace` swaps the inode, so this is only safe once the caller has
    established that no writer holds the old descriptor - see
    `is_hour_being_written`.
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

    `n` is authoritative: index line i must carry n == i. Pairing positionally
    without checking would hand back one frame's bytes labelled with another
    frame's receipt time and sequence numbers whenever an entry is missing from
    the middle of the index.

    Raises PairLengthMismatch when the files hold different numbers of lines,
    IndexPositionMismatch when an entry does not describe the raw line it sits
    against, and TruncatedFrameFile when either file's tail is torn - all
    conditions `reconcile_pair` exists to repair or an operator must know about.

    For recovered entries (kind=="recovered"), the escape state is unrecoverable
    from the raw line alone, so the payload is returned as stored and may still
    be in escaped form. This is why kind="recovered" exists - so downstream can
    exclude these entries explicitly and handle them as needed.
    """
    raw_lines = _read_lines(raw_path)
    idx_lines = _read_lines(idx_path)

    # Detect and refuse to paper over partial write failures. If raw and idx have
    # different line counts, it means a frame was written but its index entry was not.
    # This is the exact condition that reconcile_pair is designed to repair.
    if len(raw_lines) != len(idx_lines):
        raise PairLengthMismatch(Path(raw_path), Path(idx_path), len(raw_lines), len(idx_lines))

    result = []
    for position, (raw_line, idx_line) in enumerate(zip(raw_lines, idx_lines)):
        entry = decode_index_entry(idx_line)
        if entry.n != position:
            raise IndexPositionMismatch(Path(raw_path), Path(idx_path), position, entry.n)
        # For recovered entries, the escape state is unknown, so return as-stored
        # (which may still be escaped). For data entries, unescape if marked.
        if entry.kind == "recovered":
            payload = raw_line
        else:
            payload = unescape_payload(raw_line) if entry.esc else raw_line
        result.append((payload, entry))
    return result


@dataclass(frozen=True)
class RepairOutcome:
    """What `reconcile_pair` actually did, including what it could not save.

    `entries_rebuilt` alone reads reassuringly on a repair that recovered every
    frame's bytes while losing every frame's timestamp, so the losses are
    reported beside it rather than left to be inferred from the file.
    """

    entries_rebuilt: int = 0
    """Index entries reconstructed as kind="recovered" for raw frames that had none.

    Each one is a frame whose receipt time, exchange time and sequence numbers
    are gone. The payload bytes survive.
    """

    entries_discarded: int = 0
    """Index entries dropped because the frames they described did not survive.

    Their metadata still exists in the quarantined index.
    """

    raw_frames_kept: int = 0
    """Raw frames the pair holds after the repair."""

    raw_was_salvaged: bool = False
    """True when the raw file was torn and was rewritten to its intact prefix.

    The number of raw frames *lost* is deliberately not reported: a torn zstd
    block yields no count of what it was carrying. `entries_discarded` is a lower
    bound whenever the index outlived the raw file, and nothing bounds it when
    both were torn together. Reporting a number here would invent one.
    """

    quarantined_paths: tuple[Path, ...] = field(default_factory=tuple)
    """Byte-for-byte copies of every file this repair overwrote."""

    def __bool__(self) -> bool:
        return bool(self.entries_rebuilt or self.entries_discarded
                    or self.raw_was_salvaged)


def _quarantine_file(path: Path) -> Path:
    """Copy `path` beside itself before a repair overwrites it.

    Repair is the only thing in this module that destroys data, and it does so
    exactly when the data is already damaged - the worst moment to be wrong about
    what was salvageable. The original bytes are kept verbatim so a later, better
    tool (or a human) can still work on them.

    The suffix goes on the end of the full name, not in place of the existing
    one, so a quarantined file never matches a raw/index glob and cannot be
    mistaken for an hour of its own.
    """
    path = Path(path)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = path.parent / f"{path.name}{QUARANTINE_SUFFIX}-{stamp}"
    attempt = 0
    while destination.exists():
        attempt += 1
        destination = path.parent / f"{path.name}{QUARANTINE_SUFFIX}-{stamp}.{attempt}"
    shutil.copy2(path, destination)
    return destination


def _salvage_raw_lines(raw_path: Path) -> tuple[list[str], bool]:
    """Read a raw file, falling back to its intact prefix when the tail is torn.

    Returns (lines, was_torn). Every complete, newline-terminated line that
    decoded before the damage is a whole frame and is kept. What is available to
    keep is bounded by zstd's compressed block, not by the line - see the
    limitation noted on `reconcile_pair`.
    """
    try:
        return _read_lines(raw_path), False
    except TruncatedFrameFile as exc:
        return list(exc.recovered_lines), True


def reconcile_pair(raw_path: Path, idx_path: Path) -> RepairOutcome:
    """Repair a raw/index pair until a `RawWriter` can append to it again.

    This is the remedy `HourFileNotAppendable` names, so it has one hard
    obligation: refuse -> repair -> resume must terminate. Every damage shape a
    crash can leave has to come out of here appendable, or the refusal is a trap
    with no exit and the venue never records again.

    What it does, in order:

    * Refuses while a writer still holds the hour open (HourStillBeingWritten).
      The index is replaced by inode swap, and a live writer would keep filling
      the orphaned inode.
    * Salvages a torn raw file down to its last intact line. Those lines are
      whole frames; the bytes after them cannot be reconstructed from anywhere,
      so they are counted as lost rather than quietly written out of existence.
    * Salvages a torn index the same way. The index is rebuilt wholesale anyway.
    * Rebuilds entries for raw lines that have none, located by `n` rather than
      assumed to be a suffix - an index write can fail mid-stream with later ones
      succeeding, and appending the recovered entries at the end would shift
      every later entry onto the wrong frame.
    * Drops index entries describing frames the salvage could not keep. This is
      allowed ONLY when the raw file was torn, because only then is their absence
      explained. An index that overruns an intact raw file is still refused
      (UnrepairableIndex): that is corruption, not a crash, and an operator
      should look at it.
    * Verifies afterwards that the pair is genuinely appendable, so "repaired"
      never means "still stuck".

    Nothing is overwritten without a byte-for-byte copy being quarantined beside
    it first - see `_quarantine_file`. The returned `RepairOutcome` reports what
    was rebuilt and what was lost; the quarantine files are the durable record.

    Never invents a receipt timestamp: unknown times are recorded as 0 with
    kind="recovered" so downstream can exclude them explicitly.

    Limitation: the escape state (whether a payload was escaped) is unrecoverable
    from the raw line alone. Given only stored bytes, you cannot distinguish
    "original contained a real newline, was escaped" from "original literally
    contained backslash-then-n and was not escaped". Both produce identical disk
    bytes. Therefore read_pair returns recovered entries' payloads as stored,
    which may still be in escaped form. This is why kind="recovered" exists -
    downstream must exclude or handle these entries explicitly.

    Limitation: salvage granularity is zstd's compressed block, not the line.
    Nothing at all decodes from a torn block - verified against zstandard 0.25.0
    for decompressobj, stream_reader and chunked stream_reader alike - and an
    hour written in one open is a single block. So one chopped byte costs the
    whole hour's raw frames, or the whole index's timestamps, not a tail of them.
    An hour that survived several opens keeps every complete frame before the
    damaged one. Only a more frequent zstd frame boundary in `RawWriter` would
    bound this, and that is a capture-spec decision about storage cost rather
    than something the repair tool can decide. `entries_rebuilt` and
    `entries_discarded` are reported separately so the size of the loss is
    visible instead of hidden behind one reassuring "repaired" number.
    """
    raw_path, idx_path = Path(raw_path), Path(idx_path)

    is_live, marker, pid = is_hour_being_written(raw_path)
    if is_live:
        raise HourStillBeingWritten(raw_path, marker, pid)

    raw_exists, idx_exists = raw_path.exists(), idx_path.exists()
    if not raw_exists:
        if not idx_exists:
            # Neither file exists: an unwritten hour is already appendable.
            return RepairOutcome()
        raise MissingPairFile(raw_path, "raw")

    raw_lines, raw_was_torn = _salvage_raw_lines(raw_path)
    idx_lines, idx_was_torn = _salvage_raw_lines(idx_path) if idx_exists else ([], False)

    entry_line_by_n: dict[int, str] = {}
    entries_discarded = 0
    previous_n = -1
    for position, line in enumerate(idx_lines):
        n = decode_index_entry(line).n
        if n <= previous_n:
            raise UnrepairableIndex(
                raw_path, idx_path,
                f"index line {position} carries n={n}, which does not follow n={previous_n}")
        if n >= len(raw_lines):
            if not raw_was_torn:
                raise UnrepairableIndex(
                    raw_path, idx_path,
                    f"index line {position} carries n={n} but the raw file holds only "
                    f"{len(raw_lines)} frames, so it describes a frame that does not exist")
            # The raw tail is torn, so this entry describes a frame the damage
            # took. Its metadata survives only in the quarantined index.
            entries_discarded += 1
            previous_n = n
            continue
        entry_line_by_n[n] = line
        previous_n = n

    missing = [n for n in range(len(raw_lines)) if n not in entry_line_by_n]
    if not missing and not idx_was_torn and not raw_was_torn and idx_exists:
        return RepairOutcome(raw_frames_kept=len(raw_lines))

    quarantined: list[Path] = []
    if raw_was_torn:
        quarantined.append(_quarantine_file(raw_path))
    if idx_exists:
        quarantined.append(_quarantine_file(idx_path))

    for n in missing:
        entry_line_by_n[n] = encode_index_entry(IndexEntry(
            n=n, t_recv_ns=0, t_exch_ms=None, seq=None, kind="recovered", esc=False))

    # Index first, and the order is load-bearing: repair must survive being
    # interrupted halfway by the same crash that caused the damage.
    #
    # Writing the raw file first and dying leaves a salvaged (short) raw file
    # beside the original (long) index. On the next run the raw file is no longer
    # torn, so the overrunning index is refused as UnrepairableIndex - and that
    # refusal is correct, which makes it a dead end: exactly the trap this
    # function exists to remove.
    #
    # Writing the index first and dying leaves the torn raw file untouched beside
    # a short index. The next run salvages the raw file exactly as this one did,
    # finds the index consistent with it, and finishes the job. The raw file's
    # bytes are not at risk either way - both originals are already quarantined
    # above, and `_write_lines` replaces by inode swap rather than in place.
    _write_lines(idx_path, [entry_line_by_n[n] for n in range(len(raw_lines))])
    if raw_was_torn:
        _write_lines(raw_path, raw_lines)

    # "Repaired" must mean "a writer can resume", not "the tool ran". This is the
    # same check `_open` makes, so passing it here is the property that makes
    # refuse -> repair -> resume terminate.
    RawWriter._count_frames_already_written(raw_path, idx_path)

    return RepairOutcome(
        entries_rebuilt=len(missing),
        entries_discarded=entries_discarded,
        raw_frames_kept=len(raw_lines),
        raw_was_salvaged=raw_was_torn,
        quarantined_paths=tuple(quarantined),
    )
