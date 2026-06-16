#!/usr/bin/env python3
"""Replay flagged K-LD7 RADC logs against TrackMan launch angles.

This is the bridge from the temporary TrackMan collection workflow back to
real signal-processing work. It expects OpenFlight JSONL logs captured with
``scripts/start-kiosk.sh --trackman-test`` so each ``kld7_buffer.frames`` row
contains ``radc_b64``. It then reruns
``openflight.kld7.radc.extract_launch_angle`` with candidate parameters and
compares the replayed angle to TrackMan.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pickle
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from kld7_session_review_lib import _validate_frames  # noqa: E402

from openflight.kld7.radc import (  # noqa: E402
    RADC_PAYLOAD_BYTES,
    extract_launch_angle,
    radc_capture_diagnostics,
    select_best_shot_result,
)


@dataclass(frozen=True)
class TrackmanTarget:
    shot_number: int
    orientation: str
    trackman_angle_deg: float
    ball_speed_mph: float
    club: str
    openflight_timestamp: datetime | None = None
    club_speed_mph: float | None = None


@dataclass(frozen=True)
class ReplayParams:
    speed_tolerance_mph: float
    impact_energy_threshold: float
    centroid_floor_frac: float
    ops_bin_outlier_tol: int
    ops_bin_outlier_penalty: float
    ops_anchored_peak_min_snr: float = 5.0
    require_ops_anchored_peak: bool = False
    horizontal_angle_limit_deg: float = 15.0
    vertical_angle_offset_deg: float = 0.0
    horizontal_angle_offset_deg: float = 0.0


@dataclass(frozen=True)
class ReplayRow:
    shot_number: int
    orientation: str
    club: str
    trackman_angle_deg: float
    replay_angle_deg: float | None
    error_deg: float | None
    frame_count: int
    avg_snr_db: float | None
    reason: str
    target_ball_speed_mph: float | None = None
    buffer_frame_count: int | None = None
    detection_frame_count: int | None = None


@dataclass(frozen=True)
class ReplaySummary:
    params: ReplayParams
    attempted: int
    detected: int
    detection_rate: float
    mae: float | None
    p90_abs_error: float | None
    max_abs_error: float | None
    within_half_degree: int
    reason_counts: dict[str, int]


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    return values


def _parse_pickle_first_shot_number(raw: str) -> int | str:
    if raw == "auto":
        return raw
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer shot number or 'auto'") from error
    if value < 1:
        raise argparse.ArgumentTypeError("shot number must be >= 1")
    return value


def _parse_axis(raw: str) -> str:
    aliases = {
        "all": "all",
        "v": "vertical",
        "vertical": "vertical",
        "h": "horizontal",
        "horizontal": "horizontal",
    }
    try:
        return aliases[raw.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "expected one of: all, vertical, v, horizontal, h"
        ) from error


def _filter_frames_to_shot_window(
    frames: list[dict[str, Any]],
    shot_timestamp: float | None,
    *,
    window_before_s: float,
    window_after_s: float,
) -> list[dict[str, Any]]:
    """Mirror live KLD7Tracker shot-time filtering for JSONL replay."""
    if shot_timestamp is None:
        return frames
    start = shot_timestamp - max(0.0, float(window_before_s))
    end = shot_timestamp + max(0.0, float(window_after_s))
    return [frame for frame in frames if start <= float(frame["timestamp"]) <= end]


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(p * (len(ordered) - 1))))
    return ordered[index]


def _canonical_club(value: str) -> str:
    club = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "7i": "7-iron",
        "7-iron": "7-iron",
        "9i": "9-iron",
        "9-iron": "9-iron",
        "pw": "pw",
        "wedge": "pw",
        "driver": "driver",
        "3w": "3-wood",
        "3-wood": "3-wood",
    }
    return aliases.get(club, club)


def load_targets(comparison_csv: Path, axis: str = "all") -> list[TrackmanTarget]:
    """Load good OpenFlight/TrackMan angle pairs from compare_trackman output."""
    allowed = {
        "all": {"vertical", "horizontal"},
        "vertical": {"vertical"},
        "horizontal": {"horizontal"},
    }[axis]
    targets: list[TrackmanTarget] = []
    with comparison_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("match_quality") != "good":
                continue
            shot_number = _to_int(row.get("shot_number_of"))
            ball_speed = _to_float(row.get("ball_speed_of"))
            if shot_number is None or ball_speed is None:
                continue
            for orientation, field in (
                ("vertical", "launch_v_tm"),
                ("horizontal", "launch_h_tm"),
            ):
                if orientation not in allowed:
                    continue
                trackman_angle = _to_float(row.get(field))
                if trackman_angle is None:
                    continue
                targets.append(
                    TrackmanTarget(
                        shot_number=shot_number,
                        orientation=orientation,
                        trackman_angle_deg=trackman_angle,
                        ball_speed_mph=ball_speed,
                        club=row.get("club") or "",
                        openflight_timestamp=_parse_datetime(row.get("timestamp_of")),
                        club_speed_mph=_to_float(row.get("club_speed_of")),
                    )
                )
    return targets


def _normalize_pickle_frame(frame: Any) -> dict[str, Any] | None:
    if not isinstance(frame, dict):
        return None
    timestamp = _to_float(frame.get("timestamp"))
    radc = frame.get("radc")
    if timestamp is None or not isinstance(radc, bytes):
        return None
    return {"timestamp": timestamp, "radc": radc}


def load_pickle_buffers(
    openflight_pickle: Path,
    *,
    first_shot_number: int = 1,
    buffer_seconds: float = 6.0,
    shot_window_after_s: float = 0.75,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Load raw K-LD7 pickle captures keyed by mapped shot number/orientation.

    The standalone capture scripts store raw RADC frames plus OPS243 shot
    timestamps, but they do not know the OpenFlight comparison CSV shot number.
    ``first_shot_number`` maps capture index 0 to a comparison row; subsequent
    OPS shots increment from there.
    """
    with openflight_pickle.open("rb") as handle:
        capture = pickle.load(handle)
    if not isinstance(capture, dict):
        raise ValueError(f"{openflight_pickle} must contain a dict capture")

    metadata = capture.get("metadata") if isinstance(capture.get("metadata"), dict) else {}
    orientation = metadata.get("orientation")
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError(f"{openflight_pickle} metadata.orientation must be vertical or horizontal")

    frames = [
        normalized
        for normalized in (_normalize_pickle_frame(frame) for frame in capture.get("frames", []))
        if normalized is not None
    ]
    shots = [
        shot
        for shot in capture.get("ops243_shots", [])
        if isinstance(shot, dict) and _to_float(shot.get("timestamp")) is not None
    ]
    if not shots:
        if frames:
            return {(first_shot_number, orientation): frames}
        return {}

    buffers: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for idx, shot in enumerate(shots):
        shot_ts = float(shot["timestamp"])
        start = shot_ts - max(0.0, buffer_seconds)
        end = shot_ts + max(0.0, shot_window_after_s)
        buffers[(first_shot_number + idx, orientation)] = [
            frame for frame in frames if start <= float(frame["timestamp"]) <= end
        ]
    return buffers


def pickle_capture_info(openflight_pickle: Path) -> dict[str, Any]:
    """Return lightweight metadata needed to align standalone pickle captures."""
    with openflight_pickle.open("rb") as handle:
        capture = pickle.load(handle)
    if not isinstance(capture, dict):
        raise ValueError(f"{openflight_pickle} must contain a dict capture")
    metadata = capture.get("metadata") if isinstance(capture.get("metadata"), dict) else {}
    shots = [
        shot
        for shot in capture.get("ops243_shots", [])
        if isinstance(shot, dict) and _to_float(shot.get("timestamp")) is not None
    ]
    return {
        "orientation": metadata.get("orientation"),
        "club": metadata.get("club"),
        "shot_count": len(shots),
        "expected_shots": _to_int(metadata.get("expected_shots")),
        "capture_start": _parse_datetime(metadata.get("capture_start")),
        "capture_end": _parse_datetime(metadata.get("capture_end")),
    }


def jsonl_capture_info(openflight_jsonl: Path) -> dict[str, Any]:
    """Return wall-clock bounds and K-LD7 raw-payload metadata for a JSONL log."""
    timestamps: list[datetime] = []
    kld7_buffer_count = 0
    kld7_radc_frames_total = 0
    kld7_radc_payloads_total = 0
    kld7_radc_payloads_valid_total = 0
    kld7_radc_payloads_invalid_total = 0
    kld7_payload_expected_count = 0
    kld7_payload_complete_count = 0
    kld7_payload_incomplete_count = 0
    kld7_experiments: dict[str, Any] | None = None
    with openflight_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {openflight_jsonl}: {error.msg}"
                ) from error
            for field in ("ts", "start_time"):
                parsed = _parse_datetime(entry.get(field))
                if parsed is not None:
                    timestamps.append(parsed)
            if entry.get("type") == "session_start":
                config = entry.get("config")
                if isinstance(config, dict):
                    experiments = config.get("kld7_experiments")
                    if isinstance(experiments, dict):
                        kld7_experiments = experiments
            if entry.get("type") != "kld7_buffer":
                continue
            kld7_buffer_count += 1
            radc_frame_count = _to_int(entry.get("radc_frame_count"))
            radc_payload_count = _to_int(entry.get("radc_payload_count"))
            radc_payload_valid_count = _to_int(entry.get("radc_payload_valid_count"))
            radc_payload_invalid_count = _to_int(entry.get("radc_payload_invalid_count"))
            frames = entry.get("frames") if isinstance(entry.get("frames"), list) else []
            if radc_frame_count is None:
                radc_frame_count = sum(
                    1
                    for frame in frames
                    if isinstance(frame, dict) and (frame.get("has_radc") or frame.get("radc_b64"))
                )
            if radc_payload_count is None:
                radc_payload_count = sum(
                    1 for frame in frames if isinstance(frame, dict) and frame.get("radc_b64")
                )
            if radc_payload_valid_count is None:
                radc_payload_valid_count = sum(
                    1
                    for frame in frames
                    if isinstance(frame, dict)
                    and frame.get("radc_b64")
                    and frame.get("radc_payload_bytes") == RADC_PAYLOAD_BYTES
                )
            if radc_payload_invalid_count is None:
                radc_payload_invalid_count = sum(
                    1
                    for frame in frames
                    if isinstance(frame, dict)
                    and frame.get("radc_b64")
                    and frame.get("radc_payload_bytes") is not None
                    and frame.get("radc_payload_bytes") != RADC_PAYLOAD_BYTES
                )
            kld7_radc_frames_total += radc_frame_count
            kld7_radc_payloads_total += radc_payload_count
            kld7_radc_payloads_valid_total += radc_payload_valid_count
            kld7_radc_payloads_invalid_total += radc_payload_invalid_count
            if entry.get("radc_payload_expected") is True:
                kld7_payload_expected_count += 1
                if entry.get("radc_payload_complete") is True:
                    kld7_payload_complete_count += 1
                else:
                    kld7_payload_incomplete_count += 1
    return {
        "capture_start": min(timestamps) if timestamps else None,
        "capture_end": max(timestamps) if timestamps else None,
        "kld7_buffer_count": kld7_buffer_count,
        "kld7_radc_frames_total": kld7_radc_frames_total,
        "kld7_radc_payloads_total": kld7_radc_payloads_total,
        "kld7_radc_payloads_valid_total": kld7_radc_payloads_valid_total,
        "kld7_radc_payloads_invalid_total": kld7_radc_payloads_invalid_total,
        "kld7_payload_expected_count": kld7_payload_expected_count,
        "kld7_payload_complete_count": kld7_payload_complete_count,
        "kld7_payload_incomplete_count": kld7_payload_incomplete_count,
        "kld7_experiments": kld7_experiments,
    }


def targets_outside_capture_window(
    targets: list[TrackmanTarget],
    capture_info: dict[str, Any],
    *,
    tolerance_seconds: float = 300.0,
) -> list[TrackmanTarget]:
    """Return timestamped comparison targets not covered by a raw-RADC capture.

    Standalone pickle captures and JSONL session logs can be mapped to
    comparison shot numbers, but that mapping is only meaningful when the
    capture window overlaps the TrackMan comparison rows. Without this guard a
    same-club capture from a different practice block can look like a failed
    signal-processing replay.
    """
    start = capture_info.get("capture_start")
    end = capture_info.get("capture_end")
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return []
    tol = timedelta(seconds=max(0.0, float(tolerance_seconds)))
    lo = start - tol
    hi = end + tol
    return [
        target
        for target in targets
        if target.openflight_timestamp is not None and not (lo <= target.openflight_timestamp <= hi)
    ]


def targets_inside_capture_window(
    targets: list[TrackmanTarget],
    capture_info: dict[str, Any],
    *,
    tolerance_seconds: float = 300.0,
) -> list[TrackmanTarget]:
    """Return timestamped comparison targets covered by a capture window."""
    start = capture_info.get("capture_start")
    end = capture_info.get("capture_end")
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return []
    tol = timedelta(seconds=max(0.0, float(tolerance_seconds)))
    lo = start - tol
    hi = end + tol
    return [
        target
        for target in targets
        if target.openflight_timestamp is not None and lo <= target.openflight_timestamp <= hi
    ]


def target_timestamp_count(targets: list[TrackmanTarget]) -> int:
    """Return how many comparison targets can be checked against capture time."""
    return sum(1 for target in targets if target.openflight_timestamp is not None)


def pickle_first_shot_candidates(
    targets: list[TrackmanTarget],
    capture_info: dict[str, Any],
) -> list[int]:
    """Return plausible comparison shot numbers for pickle capture index 0.

    Standalone captures have OPS shot order but no comparison shot numbers.
    Prefer targets matching the pickle orientation and club metadata, then
    enumerate starts that could cover at least one target in that subset.
    """
    orientation = capture_info.get("orientation")
    filtered = [target for target in targets if target.orientation == orientation]
    club = capture_info.get("club")
    if club:
        capture_club = _canonical_club(str(club))
        same_club = [target for target in filtered if _canonical_club(target.club) == capture_club]
        if same_club:
            filtered = same_club
    shot_numbers = sorted({target.shot_number for target in filtered})
    if not shot_numbers:
        return [1]

    shot_count = int(capture_info.get("shot_count") or 0)
    if shot_count <= 1:
        return shot_numbers

    first = max(1, min(shot_numbers) - shot_count + 1)
    last = max(shot_numbers)
    return list(range(first, last + 1))


def load_jsonl_buffers(
    openflight_jsonl: Path,
    *,
    window_before_s: float = 6.0,
    window_after_s: float = 0.75,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Load and decode experimental K-LD7 buffers keyed by shot/orientation."""
    buffers: dict[tuple[int, str], list[dict[str, Any]]] = {}
    with openflight_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {openflight_jsonl}: {error.msg}"
                ) from error
            if entry.get("type") != "kld7_buffer":
                continue
            shot_number = _to_int(entry.get("shot_number"))
            orientation = entry.get("orientation")
            if shot_number is None or orientation not in {"vertical", "horizontal"}:
                continue
            frames = _validate_frames(shot_number, entry, require_radc_payload_size=True)
            shot_timestamp = _to_float(entry.get("shot_timestamp"))
            frames = _filter_frames_to_shot_window(
                frames,
                shot_timestamp,
                window_before_s=window_before_s,
                window_after_s=window_after_s,
            )
            buffers[(shot_number, orientation)] = frames
    return buffers


def load_buffers(
    openflight_path: Path,
    *,
    pickle_first_shot_number: int = 1,
    pickle_buffer_seconds: float = 6.0,
    pickle_shot_window_after_s: float = 0.75,
    jsonl_window_before_s: float = 6.0,
    jsonl_window_after_s: float = 0.75,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Load experimental JSONL or standalone raw-RADC pickle buffers."""
    if openflight_path.suffix == ".pkl":
        return load_pickle_buffers(
            openflight_path,
            first_shot_number=pickle_first_shot_number,
            buffer_seconds=pickle_buffer_seconds,
            shot_window_after_s=pickle_shot_window_after_s,
        )
    return load_jsonl_buffers(
        openflight_path,
        window_before_s=jsonl_window_before_s,
        window_after_s=jsonl_window_after_s,
    )


def replay_one(
    target: TrackmanTarget,
    frames: list[dict[str, Any]] | None,
    params: ReplayParams,
) -> ReplayRow:
    if not frames:
        return ReplayRow(
            shot_number=target.shot_number,
            orientation=target.orientation,
            club=target.club,
            trackman_angle_deg=target.trackman_angle_deg,
            replay_angle_deg=None,
            error_deg=None,
            frame_count=0,
            avg_snr_db=None,
            reason="missing_kld7_buffer",
            target_ball_speed_mph=target.ball_speed_mph,
            buffer_frame_count=0,
            detection_frame_count=0,
        )
    if not any(frame.get("radc") for frame in frames):
        return ReplayRow(
            shot_number=target.shot_number,
            orientation=target.orientation,
            club=target.club,
            trackman_angle_deg=target.trackman_angle_deg,
            replay_angle_deg=None,
            error_deg=None,
            frame_count=len(frames),
            avg_snr_db=None,
            reason="missing_radc_payload",
            target_ball_speed_mph=target.ball_speed_mph,
            buffer_frame_count=len(frames),
            detection_frame_count=0,
        )

    results = extract_launch_angle(
        frames,
        ops243_ball_speed_mph=target.ball_speed_mph,
        angle_offset_deg=(
            params.horizontal_angle_offset_deg
            if target.orientation == "horizontal"
            else params.vertical_angle_offset_deg
        ),
        speed_tolerance_mph=params.speed_tolerance_mph,
        impact_energy_threshold=params.impact_energy_threshold,
        centroid_floor_frac=params.centroid_floor_frac,
        ops_bin_outlier_tol=params.ops_bin_outlier_tol,
        ops_bin_outlier_penalty=params.ops_bin_outlier_penalty,
        ops_anchored_peak_min_snr=params.ops_anchored_peak_min_snr,
        require_ops_anchored_peak=params.require_ops_anchored_peak,
        horizontal_angle_limit_deg=params.horizontal_angle_limit_deg,
        orientation=target.orientation,
    )
    if not results:
        return ReplayRow(
            shot_number=target.shot_number,
            orientation=target.orientation,
            club=target.club,
            trackman_angle_deg=target.trackman_angle_deg,
            replay_angle_deg=None,
            error_deg=None,
            frame_count=len(frames),
            avg_snr_db=None,
            reason="no_radc_detection",
            target_ball_speed_mph=target.ball_speed_mph,
            buffer_frame_count=len(frames),
            detection_frame_count=0,
        )

    best = select_best_shot_result(results)
    replay_angle = float(best["launch_angle_deg"])
    error = replay_angle - target.trackman_angle_deg
    detection_frame_count = int(best.get("frame_count") or 0)
    return ReplayRow(
        shot_number=target.shot_number,
        orientation=target.orientation,
        club=target.club,
        trackman_angle_deg=target.trackman_angle_deg,
        replay_angle_deg=replay_angle,
        error_deg=error,
        frame_count=detection_frame_count,
        avg_snr_db=_to_float(best.get("avg_snr_db")),
        reason="ok",
        target_ball_speed_mph=target.ball_speed_mph,
        buffer_frame_count=len(frames),
        detection_frame_count=detection_frame_count,
    )


def replay_all(
    targets: list[TrackmanTarget],
    buffers: dict[tuple[int, str], list[dict[str, Any]]],
    params: ReplayParams,
) -> list[ReplayRow]:
    return [
        replay_one(target, buffers.get((target.shot_number, target.orientation)), params)
        for target in targets
    ]


def filter_targets_to_buffers(
    targets: list[TrackmanTarget],
    buffers: dict[tuple[int, str], list[dict[str, Any]]],
) -> list[TrackmanTarget]:
    """Keep only comparison targets that have a mapped K-LD7 buffer."""
    return [target for target in targets if (target.shot_number, target.orientation) in buffers]


def raw_radc_readiness(
    targets: list[TrackmanTarget],
    buffers: dict[tuple[int, str], list[dict[str, Any]]],
) -> dict[str, int]:
    """Summarize whether comparison targets have replayable raw K-LD7 payloads."""
    buffered = 0
    with_radc = 0
    missing_buffer = 0
    missing_radc_payload = 0
    invalid_radc_payload = 0
    for target in targets:
        frames = buffers.get((target.shot_number, target.orientation))
        if not frames:
            missing_buffer += 1
            continue
        buffered += 1
        radc_payloads = [frame.get("radc") for frame in frames if frame.get("radc")]
        if any(
            isinstance(payload, bytes) and len(payload) == RADC_PAYLOAD_BYTES
            for payload in radc_payloads
        ):
            with_radc += 1
        elif radc_payloads:
            invalid_radc_payload += 1
        else:
            missing_radc_payload += 1
    return {
        "targets": len(targets),
        "buffered": buffered,
        "with_radc": with_radc,
        "missing_buffer": missing_buffer,
        "missing_radc_payload": missing_radc_payload,
        "invalid_radc_payload": invalid_radc_payload,
    }


def raw_radc_readiness_passes(readiness: dict[str, int]) -> bool:
    """Return whether every evaluated target has replayable raw RADC bytes."""
    return readiness["targets"] > 0 and readiness["with_radc"] == readiness["targets"]


def summarize(params: ReplayParams, rows: list[ReplayRow]) -> ReplaySummary:
    errors = [abs(row.error_deg) for row in rows if row.error_deg is not None]
    reason_counts = Counter(row.reason for row in rows)
    return ReplaySummary(
        params=params,
        attempted=len(rows),
        detected=len(errors),
        detection_rate=len(errors) / len(rows) if rows else 0.0,
        mae=statistics.fmean(errors) if errors else None,
        p90_abs_error=_percentile(errors, 0.9),
        max_abs_error=max(errors) if errors else None,
        within_half_degree=sum(1 for error in errors if error <= 0.5),
        reason_counts=dict(sorted(reason_counts.items())),
    )


def write_rows(path: Path, rows: list[ReplayRow], params: ReplayParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "shot_number",
                "orientation",
                "club",
                "target_ball_speed_mph",
                "trackman_angle_deg",
                "replay_angle_deg",
                "error_deg",
                "abs_error_deg",
                "frame_count",
                "buffer_frame_count",
                "detection_frame_count",
                "avg_snr_db",
                "reason",
                "speed_tolerance_mph",
                "impact_energy_threshold",
                "centroid_floor_frac",
                "ops_bin_outlier_tol",
                "ops_bin_outlier_penalty",
                "ops_anchored_peak_min_snr",
                "require_ops_anchored_peak",
                "horizontal_angle_limit_deg",
                "vertical_angle_offset_deg",
                "horizontal_angle_offset_deg",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "shot_number": row.shot_number,
                    "orientation": row.orientation,
                    "club": row.club,
                    "target_ball_speed_mph": row.target_ball_speed_mph,
                    "trackman_angle_deg": row.trackman_angle_deg,
                    "replay_angle_deg": row.replay_angle_deg,
                    "error_deg": row.error_deg,
                    "abs_error_deg": abs(row.error_deg) if row.error_deg is not None else None,
                    "frame_count": row.frame_count,
                    "buffer_frame_count": row.buffer_frame_count,
                    "detection_frame_count": row.detection_frame_count,
                    "avg_snr_db": row.avg_snr_db,
                    "reason": row.reason,
                    "speed_tolerance_mph": params.speed_tolerance_mph,
                    "impact_energy_threshold": params.impact_energy_threshold,
                    "centroid_floor_frac": params.centroid_floor_frac,
                    "ops_bin_outlier_tol": params.ops_bin_outlier_tol,
                    "ops_bin_outlier_penalty": params.ops_bin_outlier_penalty,
                    "ops_anchored_peak_min_snr": params.ops_anchored_peak_min_snr,
                    "require_ops_anchored_peak": params.require_ops_anchored_peak,
                    "horizontal_angle_limit_deg": params.horizontal_angle_limit_deg,
                    "vertical_angle_offset_deg": params.vertical_angle_offset_deg,
                    "horizontal_angle_offset_deg": params.horizontal_angle_offset_deg,
                }
            )


def _replay_params_payload(params: ReplayParams) -> dict[str, float | int]:
    return {
        "speed_tolerance_mph": params.speed_tolerance_mph,
        "impact_energy_threshold": params.impact_energy_threshold,
        "centroid_floor_frac": params.centroid_floor_frac,
        "ops_bin_outlier_tol": params.ops_bin_outlier_tol,
        "ops_bin_outlier_penalty": params.ops_bin_outlier_penalty,
        "ops_anchored_peak_min_snr": params.ops_anchored_peak_min_snr,
        "require_ops_anchored_peak": params.require_ops_anchored_peak,
        "horizontal_angle_limit_deg": params.horizontal_angle_limit_deg,
        "vertical_angle_offset_deg": params.vertical_angle_offset_deg,
        "horizontal_angle_offset_deg": params.horizontal_angle_offset_deg,
    }


def _replay_row_payload(row: ReplayRow) -> dict[str, object]:
    return {
        "shot_number": row.shot_number,
        "orientation": row.orientation,
        "club": row.club,
        "target_ball_speed_mph": row.target_ball_speed_mph,
        "trackman_angle_deg": row.trackman_angle_deg,
        "replay_angle_deg": row.replay_angle_deg,
        "error_deg": row.error_deg,
        "abs_error_deg": abs(row.error_deg) if row.error_deg is not None else None,
        "frame_count": row.frame_count,
        "buffer_frame_count": row.buffer_frame_count,
        "detection_frame_count": row.detection_frame_count,
        "avg_snr_db": row.avg_snr_db,
        "reason": row.reason,
    }


def write_diagnostics(
    path: Path,
    targets: list[TrackmanTarget],
    buffers: dict[tuple[int, str], list[dict[str, Any]]],
    rows: list[ReplayRow],
    params: ReplayParams,
) -> None:
    """Write per-target RADC replay diagnostics as JSONL.

    This artifact intentionally excludes raw RADC bytes. It records the live
    extraction inputs, replay result, per-frame signal diagnostics, and summary
    stats needed to decide what to tune next against TrackMan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row_by_key = {(row.shot_number, row.orientation): row for row in rows}
    with path.open("w", encoding="utf-8") as handle:
        for target in targets:
            key = (target.shot_number, target.orientation)
            frames = buffers.get(key) or []
            row = row_by_key.get(key)
            frame_diagnostics = []
            diagnostics_summary: dict[str, object] = {
                "frame_count": len(frames),
                "radc_frame_count": 0,
                "valid_payload_count": 0,
                "peak_frame_count": 0,
            }
            if any(frame.get("radc") for frame in frames):
                diagnostics, diagnostics_summary = radc_capture_diagnostics(
                    frames,
                    ops243_ball_speed_mph=target.ball_speed_mph,
                    speed_tolerance_mph=params.speed_tolerance_mph,
                    orientation=target.orientation,
                    centroid_floor_frac=params.centroid_floor_frac,
                    ops_bin_warn_tol=params.ops_bin_outlier_tol,
                )
                frame_diagnostics = [diagnostic.to_dict() for diagnostic in diagnostics]

            payload = {
                "target": {
                    "shot_number": target.shot_number,
                    "orientation": target.orientation,
                    "club": target.club,
                    "trackman_angle_deg": target.trackman_angle_deg,
                    "ball_speed_mph": target.ball_speed_mph,
                    "club_speed_mph": target.club_speed_mph,
                    "openflight_timestamp": (
                        target.openflight_timestamp.isoformat()
                        if target.openflight_timestamp is not None
                        else None
                    ),
                },
                "params": _replay_params_payload(params),
                "replay": _replay_row_payload(row) if row is not None else None,
                "diagnostics_summary": diagnostics_summary,
                "frame_diagnostics": frame_diagnostics,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summary_payload(
    summary: ReplaySummary,
    *,
    min_detection_rate: float,
    axis: str,
    pickle_first_shot_number: int | None = None,
    only_buffered_targets: bool = False,
    raw_radc_readiness_summary: dict[str, int] | None = None,
    capture_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable best-summary payload for validation artifacts."""
    within_half_degree_passes = passes_within_half_degree_gate(
        summary,
        min_detection_rate=min_detection_rate,
    )
    raw_radc_passes = (
        raw_radc_readiness_passes(raw_radc_readiness_summary)
        if raw_radc_readiness_summary is not None
        else False
    )
    provenance_issues = (
        trackman_test_provenance_issues(capture_info) if capture_info is not None else []
    )
    gate_issues = []
    if not within_half_degree_passes:
        gate_issues.append("within-half-degree gate failed")
    if raw_radc_readiness_summary is None:
        gate_issues.append("raw RADC readiness not evaluated")
    elif not raw_radc_passes:
        gate_issues.append("raw RADC readiness failed")
    if capture_info is None:
        gate_issues.append("TrackMan-test provenance not evaluated")
    gate_issues.extend(f"TrackMan-test provenance: {issue}" for issue in provenance_issues)

    payload = {
        "axis": axis,
        "only_buffered_targets": only_buffered_targets,
        "params": _replay_params_payload(summary.params),
        "attempted": summary.attempted,
        "detected": summary.detected,
        "detection_rate": summary.detection_rate,
        "min_detection_rate": min_detection_rate,
        "eligible": summary.detection_rate >= min_detection_rate,
        "within_half_degree": summary.within_half_degree,
        "mae": summary.mae,
        "p90_abs_error": summary.p90_abs_error,
        "max_abs_error": summary.max_abs_error,
        "reason_counts": summary.reason_counts,
        "passes_within_half_degree_gate": within_half_degree_passes,
        "raw_radc_readiness_passes": raw_radc_passes,
        "trackman_replay_gate_passes": not gate_issues,
        "trackman_replay_gate_issues": gate_issues,
    }
    if pickle_first_shot_number is not None:
        payload["pickle_first_shot_number"] = pickle_first_shot_number
    if raw_radc_readiness_summary is not None:
        payload["raw_radc_readiness"] = raw_radc_readiness_summary
    if capture_info is not None:
        payload["capture_info"] = {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in capture_info.items()
        }
        payload["trackman_test_provenance_passes"] = not provenance_issues
        payload["trackman_test_provenance_issues"] = provenance_issues
    return payload


def write_summary(
    path: Path,
    summary: ReplaySummary,
    *,
    min_detection_rate: float,
    axis: str,
    pickle_first_shot_number: int | None = None,
    only_buffered_targets: bool = False,
    raw_radc_readiness_summary: dict[str, int] | None = None,
    capture_info: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary_payload(
        summary,
        min_detection_rate=min_detection_rate,
        axis=axis,
        pickle_first_shot_number=pickle_first_shot_number,
        only_buffered_targets=only_buffered_targets,
        raw_radc_readiness_summary=raw_radc_readiness_summary,
        capture_info=capture_info,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight_payload(
    *,
    axis: str,
    only_buffered_targets: bool,
    capture_info: dict[str, Any],
    readiness_by_first_shot: dict[int, dict[str, int]],
) -> dict[str, Any]:
    """Return machine-readable raw-RADC readiness before replay/tuning."""
    provenance_issues = trackman_test_provenance_issues(capture_info)
    readiness_payload = {
        str(first_shot_number): {
            **readiness,
            "passes": raw_radc_readiness_passes(readiness),
        }
        for first_shot_number, readiness in readiness_by_first_shot.items()
    }
    return {
        "mode": "raw_radc_preflight",
        "axis": axis,
        "only_buffered_targets": only_buffered_targets,
        "capture_info": {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in capture_info.items()
        },
        "raw_radc_readiness_by_first_shot": readiness_payload,
        "raw_radc_readiness_passes": bool(readiness_payload)
        and all(item["passes"] for item in readiness_payload.values()),
        "trackman_test_provenance_passes": not provenance_issues,
        "trackman_test_provenance_issues": provenance_issues,
    }


def write_preflight_summary(
    path: Path,
    *,
    axis: str,
    only_buffered_targets: bool,
    capture_info: dict[str, Any],
    readiness_by_first_shot: dict[int, dict[str, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = preflight_payload(
        axis=axis,
        only_buffered_targets=only_buffered_targets,
        capture_info=capture_info,
        readiness_by_first_shot=readiness_by_first_shot,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_reason_counts(reason_counts: dict[str, int]) -> str:
    if not reason_counts:
        return ""
    return "|".join(f"{reason}:{count}" for reason, count in reason_counts.items())


def _summary_sort_key(
    summary: ReplaySummary,
    min_detection_rate: float,
) -> tuple[bool, float, float, float]:
    """Rank by goal fit: enough coverage, then coverage, then worst error."""
    eligible = summary.detection_rate >= min_detection_rate
    return (
        not eligible,
        -summary.detection_rate,
        summary.max_abs_error if summary.max_abs_error is not None else math.inf,
        summary.mae if summary.mae is not None else math.inf,
    )


def _summary_csv_header() -> str:
    return (
        "pickle_first_shot,speed_tol,impact_energy,centroid_floor,ops_bin_tol,"
        "ops_bin_penalty,ops_anchored_min_snr,require_ops_anchor,horizontal_angle_limit,"
        "vertical_angle_offset,horizontal_angle_offset,attempted,detected,detection_rate,"
        "within_0.5,mae,p90_abs,max_abs,eligible,reason_counts"
    )


def _summary_csv_row(
    summary: ReplaySummary,
    first_shot_number: int,
    *,
    min_detection_rate: float,
) -> str:
    p = summary.params
    eligible = summary.detection_rate >= min_detection_rate
    return (
        f"{first_shot_number},{p.speed_tolerance_mph:g},{p.impact_energy_threshold:g},"
        f"{p.centroid_floor_frac:g},{p.ops_bin_outlier_tol},"
        f"{p.ops_bin_outlier_penalty:g},{p.ops_anchored_peak_min_snr:g},"
        f"{p.require_ops_anchored_peak},{p.horizontal_angle_limit_deg:g},"
        f"{p.vertical_angle_offset_deg:g},{p.horizontal_angle_offset_deg:g},"
        f"{summary.attempted},{summary.detected},"
        f"{summary.detection_rate:.3f},{summary.within_half_degree},"
        f"{_fmt(summary.mae)},{_fmt(summary.p90_abs_error)},"
        f"{_fmt(summary.max_abs_error)},{eligible},"
        f"{_fmt_reason_counts(summary.reason_counts)}"
    )


def recommended_start_kiosk_flags(params: ReplayParams, axis: str = "all") -> str:
    """Return start-kiosk flags that reproduce a replay parameter set live."""
    flags = [
        "--experimental-kld7-radc-tuning",
        f"--experimental-kld7-speed-tolerance {params.speed_tolerance_mph:g}",
        f"--experimental-kld7-centroid-floor {params.centroid_floor_frac:g}",
        f"--experimental-kld7-ops-bin-tol {params.ops_bin_outlier_tol}",
        f"--experimental-kld7-ops-bin-penalty {params.ops_bin_outlier_penalty:g}",
        f"--experimental-kld7-ops-anchored-min-snr {params.ops_anchored_peak_min_snr:g}",
    ]
    if axis in {"all", "vertical"}:
        flags.append(
            f"--experimental-kld7-vertical-impact-energy {params.impact_energy_threshold:g}"
        )
        if params.vertical_angle_offset_deg:
            flags.append(f"--kld7-angle-offset {params.vertical_angle_offset_deg:g}")
    if axis in {"all", "horizontal"}:
        flags.extend(
            [
                f"--experimental-kld7-horizontal-impact-energy {params.impact_energy_threshold:g}",
                f"--experimental-kld7-horizontal-retry-impact-energy "
                f"{params.impact_energy_threshold:g}",
                f"--experimental-kld7-horizontal-angle-limit {params.horizontal_angle_limit_deg:g}",
            ]
        )
        if params.horizontal_angle_offset_deg:
            flags.append(f"--kld7-horizontal-offset {params.horizontal_angle_offset_deg:g}")
    return " \\\n  ".join(flags)


def passes_within_half_degree_gate(
    summary: ReplaySummary,
    *,
    min_detection_rate: float,
) -> bool:
    """Return whether a replay summary satisfies the TrackMan accuracy target."""
    if summary.attempted <= 0:
        return False
    if summary.detection_rate < min_detection_rate:
        return False
    if summary.detected != summary.attempted:
        return False
    return summary.max_abs_error is not None and summary.max_abs_error <= 0.5


def trackman_test_provenance_issues(capture_info: dict[str, Any]) -> list[str]:
    """Return reasons a JSONL capture does not look like default --trackman-test."""
    experiments = capture_info.get("kld7_experiments")
    if not isinstance(experiments, dict):
        return ["missing session_start config.kld7_experiments"]

    checks = [
        (
            experiments.get("raw_radc_payload_logging_requested") is True,
            "raw_radc_payload_logging_requested is not true",
        ),
        (
            experiments.get("raw_radc_payload_logging_enabled") is True,
            "raw_radc_payload_logging_enabled is not true",
        ),
        (
            experiments.get("trackman_calibration_enabled") is False,
            "trackman_calibration_enabled is not false",
        ),
        (
            experiments.get("radc_tuning_enabled") is False,
            "radc_tuning_enabled is not false",
        ),
    ]
    return [message for passed, message in checks if not passed]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--openflight", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help=(
            "Write per-target JSONL diagnostics for the best replay. This excludes raw RADC "
            "bytes but includes frame-level peak/SNR/coherence summaries for tuning."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Write the best replay summary and gate result as JSON",
    )
    parser.add_argument(
        "--only-buffered-targets",
        action="store_true",
        help=(
            "Evaluate only comparison rows that have a mapped K-LD7 buffer. "
            "Useful for partial standalone .pkl captures; leave disabled for "
            "full-session validation."
        ),
    )
    parser.add_argument(
        "--pickle-first-shot-number",
        type=_parse_pickle_first_shot_number,
        default=1,
        help=(
            "For .pkl raw-RADC captures, comparison shot number assigned to "
            "the first OPS243 shot in the pickle, or 'auto' to rank plausible "
            "alignments from comparison targets and pickle metadata (default: 1)"
        ),
    )
    parser.add_argument(
        "--pickle-buffer-seconds",
        type=float,
        default=6.0,
        help="For .pkl captures, seconds of RADC before each OPS shot (default: 6.0)",
    )
    parser.add_argument(
        "--pickle-shot-window-after",
        type=float,
        default=0.75,
        help="For .pkl captures, seconds of RADC after each OPS shot (default: 0.75)",
    )
    parser.add_argument(
        "--jsonl-window-before",
        type=float,
        default=6.0,
        help=(
            "For JSONL kld7_buffer replay, seconds of RADC before shot_timestamp. "
            "Default 6.0 mirrors server KLD7Tracker.buffer_seconds."
        ),
    )
    parser.add_argument(
        "--jsonl-window-after",
        type=float,
        default=0.75,
        help=(
            "For JSONL kld7_buffer replay, seconds of RADC after shot_timestamp. "
            "Default 0.75 mirrors KLD7Tracker.shot_window_after_s."
        ),
    )
    parser.add_argument("--speed-tolerance", type=_parse_float_list, default=[10.0])
    parser.add_argument("--impact-energy", type=_parse_float_list, default=[3.0])
    parser.add_argument("--centroid-floor", type=_parse_float_list, default=[0.5])
    parser.add_argument("--ops-bin-tol", type=_parse_int_list, default=[25])
    parser.add_argument("--ops-bin-penalty", type=_parse_float_list, default=[10.0])
    parser.add_argument("--ops-anchored-min-snr", type=_parse_float_list, default=[5.0])
    parser.add_argument(
        "--require-ops-anchor",
        action="store_true",
        help=(
            "Require a usable local peak near the OPS243-expected speed bin; "
            "otherwise skip the frame instead of falling back to the strongest in-band peak"
        ),
    )
    parser.add_argument("--horizontal-angle-limit", type=_parse_float_list, default=[15.0])
    parser.add_argument(
        "--vertical-angle-offset",
        type=float,
        default=0.0,
        help=(
            "Vertical replay angle offset in degrees. Use 8.0 to mirror the "
            "current start-kiosk --kld7 default."
        ),
    )
    parser.add_argument(
        "--horizontal-angle-offset",
        type=float,
        default=0.0,
        help="Horizontal replay angle offset in degrees (default: 0.0)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Print only the top N ranked parameter/alignment rows (default: all)",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=1.0,
        help=(
            "Minimum detection rate for a parameter set to be ranked as eligible "
            "(default: 1.0, because the target is every matched shot within 0.5°)"
        ),
    )
    parser.add_argument(
        "--require-within-half-degree",
        action="store_true",
        help=(
            "Exit nonzero unless the best replay detects every target and "
            "has max absolute TrackMan error <= 0.5°"
        ),
    )
    parser.add_argument(
        "--require-raw-radc",
        action="store_true",
        help=(
            "Exit nonzero before replay unless every evaluated comparison target "
            "has a mapped K-LD7 buffer with raw RADC bytes"
        ),
    )
    parser.add_argument(
        "--require-trackman-test-provenance",
        action="store_true",
        help=(
            "Exit nonzero unless the OpenFlight JSONL was collected with clean default "
            "scripts/start-kiosk.sh --trackman-test provenance: raw RADC logging on, "
            "saved-angle calibration off, and RADC tuning off"
        ),
    )
    parser.add_argument(
        "--check-raw-radc-only",
        action="store_true",
        help=(
            "Only print raw-RADC replay readiness for the comparison/log mapping, "
            "then exit. This avoids running the parameter grid when validating "
            "whether a TrackMan session is replayable."
        ),
    )
    parser.add_argument(
        "--axis",
        type=_parse_axis,
        default="all",
        help="Limit replay to one launch-angle axis: all, vertical/v, horizontal/h (default: all)",
    )
    args = parser.parse_args(argv)

    targets = load_targets(args.comparison, axis=args.axis)
    if args.openflight.suffix == ".pkl":
        capture_info = pickle_capture_info(args.openflight)
    else:
        capture_info = jsonl_capture_info(args.openflight)
    if args.require_trackman_test_provenance:
        if args.openflight.suffix == ".pkl":
            parser.error("--require-trackman-test-provenance is only valid for JSONL sessions")
        provenance_issues = trackman_test_provenance_issues(capture_info)
        if provenance_issues:
            parser.error(
                "OpenFlight session was not collected with clean default "
                "scripts/start-kiosk.sh --trackman-test provenance: " + "; ".join(provenance_issues)
            )
    if args.openflight.suffix == ".pkl" and target_timestamp_count(targets) > 0:
        overlapping_targets = targets_inside_capture_window(targets, capture_info)
        if not overlapping_targets:
            start = capture_info.get("capture_start")
            end = capture_info.get("capture_end")
            parser.error(
                f"{args.openflight} capture window {start} to {end} does not overlap any "
                "timestamped comparison targets for this axis. Use a raw-RADC capture "
                "from the same TrackMan comparison session, or verify the comparison CSV "
                "timestamps before tuning."
            )
    if args.openflight.suffix == ".pkl" and args.pickle_first_shot_number == "auto":
        first_shot_numbers = pickle_first_shot_candidates(
            targets,
            capture_info,
        )
    elif args.pickle_first_shot_number == "auto":
        parser.error("--pickle-first-shot-number auto is only valid for .pkl captures")
    else:
        first_shot_numbers = [int(args.pickle_first_shot_number)]

    try:
        buffers_by_first_shot = {
            first_shot_number: load_buffers(
                args.openflight,
                pickle_first_shot_number=first_shot_number,
                pickle_buffer_seconds=args.pickle_buffer_seconds,
                pickle_shot_window_after_s=args.pickle_shot_window_after,
                jsonl_window_before_s=args.jsonl_window_before,
                jsonl_window_after_s=args.jsonl_window_after,
            )
            for first_shot_number in first_shot_numbers
        }
    except ValueError as error:
        parser.error(f"failed to load OpenFlight K-LD7 buffers: {error}")
    targets_by_first_shot = {
        first_shot_number: (
            filter_targets_to_buffers(targets, buffers) if args.only_buffered_targets else targets
        )
        for first_shot_number, buffers in buffers_by_first_shot.items()
    }
    for first_shot_number, replay_targets in targets_by_first_shot.items():
        mismatched = targets_outside_capture_window(replay_targets, capture_info)
        if mismatched:
            first = mismatched[0]
            start = capture_info.get("capture_start")
            end = capture_info.get("capture_end")
            mapping_label = (
                f"pickle first shot {first_shot_number}"
                if args.openflight.suffix == ".pkl"
                else f"JSONL shot mapping {first_shot_number}"
            )
            parser.error(
                f"{args.openflight} capture window {start} to {end} does not cover "
                f"comparison shot {first.shot_number} at {first.openflight_timestamp} "
                f"({mapping_label}). Use a raw-RADC capture from the same TrackMan "
                "comparison session."
            )

    if args.require_raw_radc:
        not_ready: list[tuple[int, dict[str, int]]] = []
        for first_shot_number, replay_targets in targets_by_first_shot.items():
            readiness = raw_radc_readiness(
                replay_targets,
                buffers_by_first_shot[first_shot_number],
            )
            if readiness["with_radc"] != readiness["targets"]:
                not_ready.append((first_shot_number, readiness))
        if not_ready:
            first_shot_number, readiness = not_ready[0]
            mapping_label = (
                f"pickle first shot {first_shot_number}"
                if args.openflight.suffix == ".pkl"
                else "JSONL shot mapping"
            )
            parser.error(
                "raw RADC replay is not possible for every evaluated target "
                f"({mapping_label}): "
                f"{readiness['with_radc']}/{readiness['targets']} targets have raw RADC, "
                f"{readiness['missing_buffer']} missing buffers, "
                f"{readiness['missing_radc_payload']} buffers missing radc_b64, "
                f"{readiness['invalid_radc_payload']} buffers with invalid RADC payload size. "
                "Re-run with scripts/start-kiosk.sh --trackman-test for TrackMan validation."
            )

    if args.check_raw_radc_only:
        all_ready = True
        print(
            "capture_raw_payloads,kld7_buffers,radc_frames,radc_payloads,"
            "payload_valid,payload_invalid,payload_expected,payload_complete,payload_incomplete"
        )
        print(
            "capture_raw_payloads,"
            f"{capture_info.get('kld7_buffer_count', 0)},"
            f"{capture_info.get('kld7_radc_frames_total', 0)},"
            f"{capture_info.get('kld7_radc_payloads_total', 0)},"
            f"{capture_info.get('kld7_radc_payloads_valid_total', 0)},"
            f"{capture_info.get('kld7_radc_payloads_invalid_total', 0)},"
            f"{capture_info.get('kld7_payload_expected_count', 0)},"
            f"{capture_info.get('kld7_payload_complete_count', 0)},"
            f"{capture_info.get('kld7_payload_incomplete_count', 0)}"
        )
        print(
            "raw_radc_readiness,pickle_first_shot,targets,buffered,with_radc,"
            "missing_buffer,missing_radc_payload,invalid_radc_payload,passes"
        )
        readiness_by_first_shot = {}
        for first_shot_number, replay_targets in targets_by_first_shot.items():
            readiness = raw_radc_readiness(
                replay_targets,
                buffers_by_first_shot[first_shot_number],
            )
            readiness_by_first_shot[first_shot_number] = readiness
            passes = raw_radc_readiness_passes(readiness)
            all_ready = all_ready and passes
            print(
                f"raw_radc_readiness,{first_shot_number},{readiness['targets']},"
                f"{readiness['buffered']},"
                f"{readiness['with_radc']},{readiness['missing_buffer']},"
                f"{readiness['missing_radc_payload']},{readiness['invalid_radc_payload']},"
                f"{passes}"
            )
        if args.openflight.suffix != ".pkl":
            provenance_issues = trackman_test_provenance_issues(capture_info)
            print("trackman_test_provenance,passes,issues")
            print(f"trackman_test_provenance,{not provenance_issues},{'|'.join(provenance_issues)}")
        if args.summary_output:
            write_preflight_summary(
                args.summary_output,
                axis=args.axis,
                only_buffered_targets=args.only_buffered_targets,
                capture_info=capture_info,
                readiness_by_first_shot=readiness_by_first_shot,
            )
            print(f"Wrote raw-RADC preflight summary to {args.summary_output}")
        return 0 if all_ready else 2

    summaries: list[tuple[ReplaySummary, list[ReplayRow], int]] = []
    for (
        first_shot_number,
        speed_tolerance,
        impact_energy,
        centroid_floor,
        ops_bin_tol,
        ops_bin_penalty,
        ops_anchored_min_snr,
        require_ops_anchor,
        horizontal_angle_limit,
    ) in itertools.product(
        first_shot_numbers,
        args.speed_tolerance,
        args.impact_energy,
        args.centroid_floor,
        args.ops_bin_tol,
        args.ops_bin_penalty,
        args.ops_anchored_min_snr,
        [args.require_ops_anchor],
        args.horizontal_angle_limit,
    ):
        buffers = buffers_by_first_shot[first_shot_number]
        params = ReplayParams(
            speed_tolerance_mph=speed_tolerance,
            impact_energy_threshold=impact_energy,
            centroid_floor_frac=centroid_floor,
            ops_bin_outlier_tol=ops_bin_tol,
            ops_bin_outlier_penalty=ops_bin_penalty,
            ops_anchored_peak_min_snr=ops_anchored_min_snr,
            require_ops_anchored_peak=require_ops_anchor,
            horizontal_angle_limit_deg=horizontal_angle_limit,
            vertical_angle_offset_deg=args.vertical_angle_offset,
            horizontal_angle_offset_deg=args.horizontal_angle_offset,
        )
        replay_targets = targets_by_first_shot[first_shot_number]
        rows = replay_all(replay_targets, buffers, params)
        summaries.append((summarize(params, rows), rows, first_shot_number))

    min_detection_rate = max(0.0, min(1.0, args.min_detection_rate))
    summaries.sort(key=lambda item: _summary_sort_key(item[0], min_detection_rate))

    print(_summary_csv_header())
    output_summaries = summaries[: args.top] if args.top > 0 else summaries
    for summary, _, first_shot_number in output_summaries:
        print(
            _summary_csv_row(
                summary,
                first_shot_number,
                min_detection_rate=min_detection_rate,
            )
        )
    if args.top > 0 and len(summaries) > args.top:
        print(f"# omitted {len(summaries) - args.top} lower-ranked rows")

    if args.output and summaries:
        best_summary, best_rows, _ = summaries[0]
        write_rows(args.output, best_rows, best_summary.params)
        print(f"Wrote best replay rows to {args.output} (MAE={_fmt(best_summary.mae)})")

    if args.diagnostics_output and summaries:
        best_summary, best_rows, best_first_shot_number = summaries[0]
        best_targets = targets_by_first_shot[best_first_shot_number]
        write_diagnostics(
            args.diagnostics_output,
            best_targets,
            buffers_by_first_shot[best_first_shot_number],
            best_rows,
            best_summary.params,
        )
        print(f"Wrote best replay diagnostics to {args.diagnostics_output}")

    if args.summary_output and summaries:
        best_summary, _, best_first_shot_number = summaries[0]
        best_targets = targets_by_first_shot[best_first_shot_number]
        best_readiness = raw_radc_readiness(
            best_targets,
            buffers_by_first_shot[best_first_shot_number],
        )
        write_summary(
            args.summary_output,
            best_summary,
            min_detection_rate=min_detection_rate,
            axis=args.axis,
            pickle_first_shot_number=(
                best_first_shot_number if args.openflight.suffix == ".pkl" else None
            ),
            only_buffered_targets=args.only_buffered_targets,
            raw_radc_readiness_summary=best_readiness,
            capture_info=capture_info,
        )
        print(f"Wrote best replay summary to {args.summary_output}")

    if summaries:
        best_summary, _, best_first_shot_number = summaries[0]
        if args.openflight.suffix == ".pkl":
            print(f"Best pickle first shot number: {best_first_shot_number}")
        print("Recommended start-kiosk experimental RADC flags:")
        print(recommended_start_kiosk_flags(best_summary.params, axis=args.axis))
        if args.require_within_half_degree:
            if passes_within_half_degree_gate(
                best_summary,
                min_detection_rate=min_detection_rate,
            ):
                print("PASS: best replay satisfies the within-0.5° TrackMan gate")
            else:
                print(
                    "FAIL: best replay does not satisfy the within-0.5° TrackMan gate",
                    file=sys.stderr,
                )
                return 2
    elif args.require_within_half_degree:
        print("FAIL: no replay summaries produced", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
