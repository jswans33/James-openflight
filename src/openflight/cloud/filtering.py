"""Client-side filtering — the raw-ADC strip.

This is the load-bearing privacy boundary. The FlightWeb server stores a
device upload **verbatim**; it does not re-filter raw radar data out of a
device upload. So the product promise "raw radar data never leaves your Pi"
is enforced *here*, by applying an allowlist before upload.

Use an allowlist, not a blocklist — any future heavy entry type the session
logger gains must never leak by default.
"""

import gzip
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .. import __version__

CLIENT_VERSION = __version__

# Allowlisted entry types — only these are uploaded. ``error`` and
# ``session_error`` are both kept: the session logger currently emits ``error``
# (see session_logger.py), while the server spec names ``session_error``;
# keeping both is privacy-safe (error entries carry only error strings/context,
# never raw ADC) and future-proofs a rename.
KEEP_ENTRY_TYPES = frozenset(
    {
        "session_start",
        "session_end",
        "shot_detected",
        "trigger_event",
        "session_error",
        "error",
    }
)

MANIFEST_TYPE = "upload_manifest"
MANIFEST_FORMAT_VERSION = 1

# Per-line cap mirroring the server's per-line guard. Belt-and-suspenders.
MAX_LINE_BYTES = 32 * 1024
# Body caps mirroring the server. A filtered session is normally tens of KB,
# so these are safety checks, not normal operating limits.
MAX_GZIP_BYTES = 20 * 1024 * 1024
MAX_INFLATED_BYTES = 64 * 1024 * 1024

# Fixed namespace for deterministic UUIDv5 of (device_id, session_filename),
# used for older sessions that predate the embedded session_uuid. A stable
# namespace makes the same file always map to the same id (dedupe + safe retry).
SESSION_NAMESPACE = uuid.UUID("8d8ac610-566d-4ef0-9c22-186b2a5ed793")


class BodyTooLargeError(Exception):
    """Raised when a filtered body exceeds the gzip/inflated caps."""


@dataclass
class FilterResult:
    """Outcome of filtering one session file."""

    manifest: Dict[str, Any]
    kept_lines: List[str]
    dropped_oversize: int = 0
    kept_type_counts: Dict[str, int] = field(default_factory=dict)
    session_id: str = ""


def _iter_entries(lines: Iterable[str]):
    """Yield (raw_line, parsed_dict) for parseable JSON lines, skipping junk."""
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            yield stripped, parsed


def _filter_entries(
    lines: Iterable[str], device_id: str, filename: str, client_version: str
) -> FilterResult:
    """Single-pass core: consume ``lines`` once, applying the allowlist.

    Works on any line iterable — including an open file object, which streams
    line by line so a huge raw-ADC session is never held in memory at once.
    Captures the ``session_uuid`` from ``session_start`` during the same pass
    so we don't need a second read to resolve the upload id.
    """
    kept_lines: List[str] = []
    kept_type_counts: Dict[str, int] = {}
    dropped_oversize = 0
    embedded_uuid = ""

    for raw, entry in _iter_entries(lines):
        entry_type = entry.get("type")
        if entry_type == "session_start" and not embedded_uuid:
            session_uuid = entry.get("session_uuid")
            if session_uuid:
                embedded_uuid = str(session_uuid).lower()
        if entry_type not in KEEP_ENTRY_TYPES:
            continue
        if len(raw.encode("utf-8")) > MAX_LINE_BYTES:
            dropped_oversize += 1
            continue
        kept_lines.append(raw)
        kept_type_counts[entry_type] = kept_type_counts.get(entry_type, 0) + 1

    session_id = embedded_uuid or str(uuid.uuid5(SESSION_NAMESPACE, f"{device_id}:{filename}"))
    manifest = {
        "type": MANIFEST_TYPE,
        "format_version": MANIFEST_FORMAT_VERSION,
        "client_version": client_version,
        "device_id": device_id,
        "filtered": True,
        "kept_entry_types": sorted(kept_type_counts),
    }
    return FilterResult(
        manifest=manifest,
        kept_lines=kept_lines,
        dropped_oversize=dropped_oversize,
        kept_type_counts=kept_type_counts,
        session_id=session_id,
    )


def filter_session_lines(
    lines: Iterable[str],
    device_id: str,
    client_version: str = CLIENT_VERSION,
    filename: str = "",
) -> FilterResult:
    """Filter raw session lines to the allowlist and build the manifest.

    Drops non-allowlisted types, drops any kept line over ``MAX_LINE_BYTES``
    (counting them), and skips blank/unparseable lines. ``filename`` is only
    used for the UUIDv5 session-id fallback when no ``session_uuid`` is present.
    """
    return _filter_entries(lines, device_id, filename, client_version)


def filter_session_file(path, device_id: str, client_version: str = CLIENT_VERSION) -> FilterResult:
    """Stream-filter a session file by path without loading it into memory.

    The raw ADC (rolling_buffer_capture / kld7_buffer / iq_blocks) is dropped as
    each line is read, so peak memory stays near a single line regardless of how
    large the raw-ADC file is. This is the production path for ``push``.
    """
    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as handle:
        return _filter_entries(handle, device_id, path.name, client_version)


def build_upload_body(
    result: FilterResult,
    max_gzip_bytes: Optional[int] = None,
    max_inflated_bytes: Optional[int] = None,
) -> bytes:
    """Build the gzipped NDJSON upload body (manifest first), enforcing caps.

    Raises BodyTooLargeError if the body would exceed either cap — the caller
    should park the session and report it rather than upload raw.
    """
    # Resolve at call time so tests (and config) can adjust the module caps.
    max_gzip_bytes = MAX_GZIP_BYTES if max_gzip_bytes is None else max_gzip_bytes
    max_inflated_bytes = MAX_INFLATED_BYTES if max_inflated_bytes is None else max_inflated_bytes
    out_lines = [json.dumps(result.manifest)]
    out_lines.extend(result.kept_lines)
    ndjson = ("\n".join(out_lines) + "\n").encode("utf-8")

    if len(ndjson) > max_inflated_bytes:
        raise BodyTooLargeError(
            f"inflated body {len(ndjson)} bytes exceeds cap {max_inflated_bytes}"
        )

    # mtime=0 keeps the gzip output deterministic (stable retries/dedupe).
    body = gzip.compress(ndjson, mtime=0)
    if len(body) > max_gzip_bytes:
        raise BodyTooLargeError(f"gzip body {len(body)} bytes exceeds cap {max_gzip_bytes}")
    return body
