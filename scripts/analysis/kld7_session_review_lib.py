"""Standalone helpers for offline K-LD7 session review."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openflight.kld7.radc import RADC_PAYLOAD_BYTES

DISTANCE_MIN_M = 0.6
DISTANCE_MAX_M = 5.2
POST_IMPACT_WINDOW_MS = 260.0
PRE_IMPACT_WINDOW_MS = 180.0
TRANSITION_MIN_DT_MS = 20.0
TRANSITION_MAX_DT_MS = 95.0
TRANSITION_MAX_DISTANCE_GAIN_M = 1.8
TRANSITION_MAX_ANGLE_DELTA_DEG = 18.0
CLUB_DIST_MAX_M = 1.10
CLUB_MAG_MIN = 2200.0
CLUB_CLUSTER_GAP_S = 0.45


@dataclass(frozen=True)
class Detection:
    frame_index: int
    time_ms: float
    distance_m: float
    angle_deg: float
    magnitude: float
    speed_raw: float


@dataclass(frozen=True)
class PathMetrics:
    start_ms: float
    end_ms: float
    duration_ms: float
    start_distance_m: float
    end_distance_m: float
    distance_gain_m: float
    point_count: int
    monotonicity: float
    angle_span_deg: float
    mean_angle_deg: float
    start_angle_deg: float
    end_angle_deg: float
    max_magnitude: float


@dataclass(frozen=True)
class AnchorCandidate:
    club_event_index: int
    club_time: float
    club_distance_m: float
    club_magnitude: float
    path: list[Detection]
    all_post_hits: list[Detection]
    track_score: float
    selection_score: float
    lingering_hits: int
    metrics: PathMetrics


@dataclass(frozen=True)
class ShotReview:
    shot_number: int
    club_label: str
    logged_ball_speed_mph: float | None
    logged_club_speed_mph: float | None
    logged_launch_angle_deg: float | None
    logged_launch_confidence: float | None
    logged_ball_angle_frames: int | None
    logged_ball_angle_accepted: bool | None
    expected_launch_deg: float | None
    allowed_delta_deg: float | None
    observed_delta_deg: float | None
    rolling_ball_dt_ms: float | None
    anchor: AnchorCandidate
    quality: str


def detection_value(detection: dict[str, Any] | None, key: str) -> float:
    if detection is None:
        return 0.0
    value = detection.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def _coerce_int(value: Any, description: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be an integer, got {value!r}") from error


def _coerce_float(value: Any, description: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be numeric, got {value!r}") from error


def _validate_frames(
    shot_number: int,
    buffer_entry: dict[str, Any],
    *,
    require_radc_payload_size: bool = False,
) -> list[dict[str, Any]]:
    frames = buffer_entry.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"shot {shot_number} is missing a usable kld7_buffer.frames list")

    normalized_frames: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"shot {shot_number} frame {frame_index} is not an object")
        if "timestamp" not in frame:
            raise ValueError(f"shot {shot_number} frame {frame_index} is missing timestamp")

        try:
            timestamp = _coerce_float(
                frame["timestamp"],
                f"shot {shot_number} frame {frame_index} timestamp",
            )
        except ValueError as error:
            raise ValueError(
                f"shot {shot_number} frame {frame_index} has non-numeric timestamp"
            ) from error

        pdat = frame.get("pdat")
        if pdat is None:
            hits: list[dict[str, Any]] = []
        elif not isinstance(pdat, list):
            raise ValueError(f"shot {shot_number} frame {frame_index} has non-list pdat data")
        else:
            hits = []
            for hit_index, hit in enumerate(pdat):
                if not isinstance(hit, dict):
                    raise ValueError(
                        f"shot {shot_number} frame {frame_index} hit {hit_index} is not an object"
                    )
                for field in ("distance", "angle", "magnitude", "speed"):
                    if field in hit and hit[field] is not None:
                        _coerce_float(
                            hit[field],
                            f"shot {shot_number} frame {frame_index} hit {hit_index} field {field}",
                        )
                hits.append(hit)

        normalized_frame: dict[str, Any] = {"timestamp": timestamp, "pdat": hits}
        radc_b64 = frame.get("radc_b64")
        if radc_b64 is not None:
            if not isinstance(radc_b64, str):
                raise ValueError(
                    f"shot {shot_number} frame {frame_index} has non-string radc_b64 data"
                )
            try:
                radc = base64.b64decode(radc_b64, validate=True)
            except ValueError as error:
                raise ValueError(
                    f"shot {shot_number} frame {frame_index} has invalid radc_b64 data"
                ) from error
            if require_radc_payload_size and len(radc) != RADC_PAYLOAD_BYTES:
                raise ValueError(
                    f"shot {shot_number} frame {frame_index} has invalid radc_b64 payload "
                    f"size: expected {RADC_PAYLOAD_BYTES} bytes, got {len(radc)}"
                )
            normalized_frame["radc"] = radc

        normalized_frames.append(normalized_frame)

    return normalized_frames


def group_records(
    records: list[dict[str, Any]],
    gap_seconds: float,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_ts: float | None = None
    for record in records:
        timestamp = float(record["timestamp"])
        if previous_ts is None or timestamp - previous_ts <= gap_seconds:
            current.append(record)
        else:
            groups.append(current)
            current = [record]
        previous_ts = timestamp
    if current:
        groups.append(current)
    return groups


def find_club_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    club_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        close_hits = []
        for detection in frame.get("pdat") or []:
            distance = detection_value(detection, "distance")
            magnitude = detection_value(detection, "magnitude")
            if distance <= CLUB_DIST_MAX_M and magnitude >= CLUB_MAG_MIN:
                close_hits.append(detection)
        if not close_hits:
            continue
        best_hit = min(
            close_hits,
            key=lambda det: (
                detection_value(det, "distance"),
                -detection_value(det, "magnitude"),
            ),
        )
        club_frames.append(
            {"index": index, "timestamp": float(frame["timestamp"]), "detection": best_hit}
        )

    events = []
    for cluster in group_records(club_frames, CLUB_CLUSTER_GAP_S):
        best_frame = min(
            cluster,
            key=lambda record: (
                detection_value(record["detection"], "distance"),
                -detection_value(record["detection"], "magnitude"),
            ),
        )
        events.append(best_frame)
    return events


def load_session(
    session_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    session_meta: dict[str, Any] = {}
    shots: dict[int, dict[str, Any]] = defaultdict(dict)
    if not session_path.exists():
        raise ValueError(f"Session file not found: {session_path}")
    if not session_path.is_file():
        raise ValueError(f"Session path is not a file: {session_path}")
    with session_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {session_path}: {error.msg}"
                ) from error
            if not isinstance(entry, dict):
                raise ValueError(f"Line {line_number} of {session_path} must be a JSON object.")
            entry_type = entry.get("type")
            if entry_type == "session_start":
                session_meta = entry
                continue
            shot_number = entry.get("shot_number")
            if entry_type in {"rolling_buffer_capture", "kld7_buffer", "shot_detected"}:
                if shot_number is None:
                    raise ValueError(
                        f"{session_path} line {line_number} is missing shot_number for {entry_type}"
                    )
                shot_number = _coerce_int(
                    shot_number,
                    f"{session_path} line {line_number} shot_number",
                )
            if shot_number is None:
                continue
            if entry_type == "rolling_buffer_capture":
                shots[shot_number]["capture"] = entry
            elif entry_type == "kld7_buffer":
                shots[shot_number]["buffer"] = entry
            elif entry_type == "shot_detected":
                shots[shot_number]["shot"] = entry
    return session_meta, dict(sorted(shots.items()))


def collect_tracking_inputs(
    frames: list[dict[str, Any]],
    club_time: float,
) -> tuple[list[Detection], list[dict[str, float]], list[Detection]]:
    detections: list[Detection] = []
    pre_event_hits: list[dict[str, float]] = []
    all_post_hits: list[Detection] = []

    for frame_index, frame in enumerate(frames):
        time_ms = (float(frame["timestamp"]) - club_time) * 1000.0
        for hit in frame.get("pdat") or []:
            distance_m = detection_value(hit, "distance")
            angle_deg = detection_value(hit, "angle")
            magnitude = detection_value(hit, "magnitude")
            speed_raw = detection_value(hit, "speed")

            if -PRE_IMPACT_WINDOW_MS <= time_ms < 0.0:
                pre_event_hits.append({"distance_m": distance_m, "angle_deg": angle_deg})

            if 0.0 <= time_ms <= POST_IMPACT_WINDOW_MS:
                if not DISTANCE_MIN_M <= distance_m <= DISTANCE_MAX_M:
                    continue
                detection = Detection(
                    frame_index=frame_index,
                    time_ms=round(time_ms, 1),
                    distance_m=distance_m,
                    angle_deg=angle_deg,
                    magnitude=magnitude,
                    speed_raw=speed_raw,
                )
                detections.append(detection)
                all_post_hits.append(detection)

    return detections, pre_event_hits, all_post_hits


def start_score(
    detection: Detection,
    pre_event_hits: list[dict[str, float]],
) -> float:
    novelty_penalty = sum(
        1
        for hit in pre_event_hits
        if abs(hit["distance_m"] - detection.distance_m) <= 0.3
        and abs(hit["angle_deg"] - detection.angle_deg) <= 8.0
    )

    score = 1.0
    score += min(1.6, max(0.0, (detection.magnitude - 1800.0) / 2200.0))

    if 0.0 <= detection.time_ms <= 120.0:
        score += 1.2
    elif detection.time_ms <= 180.0:
        score += 0.4
    else:
        score -= 1.0

    if detection.distance_m <= 2.8:
        score += 0.8
    elif detection.distance_m <= 3.6:
        score += 0.2
    else:
        score -= 0.5

    score -= 1.0 * novelty_penalty
    return score


def transition_score(previous: Detection, current: Detection) -> float | None:
    dt_ms = current.time_ms - previous.time_ms
    distance_gain = current.distance_m - previous.distance_m
    angle_delta = abs(current.angle_deg - previous.angle_deg)

    if not (TRANSITION_MIN_DT_MS <= dt_ms <= TRANSITION_MAX_DT_MS):
        return None
    if not (-0.05 <= distance_gain <= TRANSITION_MAX_DISTANCE_GAIN_M):
        return None
    if angle_delta > TRANSITION_MAX_ANGLE_DELTA_DEG:
        return None

    score = 2.0
    score += min(1.8, max(0.0, distance_gain) * 1.1)
    score -= angle_delta * 0.04
    score += min(0.8, max(0.0, (current.magnitude - 1800.0) / 3000.0))
    if distance_gain < 0.02:
        score -= 0.8
    return score


def collapse_same_frame(path: list[Detection]) -> list[Detection]:
    collapsed: list[Detection] = []
    for detection in path:
        if collapsed and collapsed[-1].frame_index == detection.frame_index:
            if detection.distance_m > collapsed[-1].distance_m:
                collapsed[-1] = detection
            continue
        collapsed.append(detection)
    return collapsed


def count_lingering_hits(
    path: list[Detection],
    frames: list[dict[str, Any]],
    club_time: float,
) -> int:
    end = path[-1]
    lingering = 0
    for frame in frames:
        time_ms = (float(frame["timestamp"]) - club_time) * 1000.0
        if time_ms <= end.time_ms + 90.0:
            continue
        for hit in frame.get("pdat") or []:
            if (
                abs(detection_value(hit, "distance") - end.distance_m) <= 0.35
                and abs(detection_value(hit, "angle") - end.angle_deg) <= 8.0
            ):
                lingering += 1
    return lingering


def compute_path_metrics(path: list[Detection]) -> PathMetrics:
    times = np.array([d.time_ms for d in path], dtype=float)
    distances = np.array([d.distance_m for d in path], dtype=float)
    angles = np.array([d.angle_deg for d in path], dtype=float)
    magnitudes = np.array([d.magnitude for d in path], dtype=float)

    if len(distances) == 1:
        monotonicity = 1.0
    else:
        monotonicity = float(np.mean(np.diff(distances) > 0.0))

    return PathMetrics(
        start_ms=float(times[0]),
        end_ms=float(times[-1]),
        duration_ms=float(times[-1] - times[0]),
        start_distance_m=float(distances[0]),
        end_distance_m=float(distances[-1]),
        distance_gain_m=float(distances[-1] - distances[0]),
        point_count=len(path),
        monotonicity=monotonicity,
        angle_span_deg=float(np.max(angles) - np.min(angles)),
        mean_angle_deg=float(np.mean(angles)),
        start_angle_deg=float(angles[0]),
        end_angle_deg=float(angles[-1]),
        max_magnitude=float(np.max(magnitudes)),
    )


def selection_score(
    track_score: float,
    metrics: PathMetrics,
    lingering_hits: int,
    rolling_ball_dt_ms: float | None,
) -> float:
    score = track_score
    score += 0.8 * min(metrics.distance_gain_m, 2.5)
    score += 0.7 * metrics.monotonicity
    score += 0.5 if metrics.point_count >= 3 else -0.6
    score += 0.4 if 30.0 <= metrics.duration_ms <= 220.0 else -0.4
    score += 0.5 if metrics.start_distance_m <= 2.5 else 0.0
    score -= 0.35 * max(0.0, metrics.start_ms - 120.0) / 40.0
    score -= 0.8 * max(0.0, metrics.start_distance_m - 3.8)
    score -= 0.6 * lingering_hits

    if rolling_ball_dt_ms is not None:
        score -= min(1.0, abs(metrics.start_ms - rolling_ball_dt_ms) / 160.0)
    return score


def extract_anchor_candidate(
    frames: list[dict[str, Any]],
    club_event: dict[str, Any],
    club_event_index: int,
    rolling_ball_dt_ms: float | None,
) -> AnchorCandidate:
    detections, pre_event_hits, all_post_hits = collect_tracking_inputs(
        frames, float(club_event["timestamp"])
    )
    if not detections:
        raise ValueError("No post-impact detections available for this anchor.")

    best_scores = [start_score(detection, pre_event_hits) for detection in detections]
    previous_index: list[int | None] = [None] * len(detections)

    for current_index, current in enumerate(detections):
        for prior_index in range(current_index):
            score = transition_score(detections[prior_index], current)
            if score is None:
                continue
            candidate_score = best_scores[prior_index] + score
            if candidate_score > best_scores[current_index]:
                best_scores[current_index] = candidate_score
                previous_index[current_index] = prior_index

    terminal_index = max(range(len(detections)), key=lambda index: best_scores[index])
    path: list[Detection] = []
    cursor: int | None = terminal_index
    while cursor is not None:
        path.append(detections[cursor])
        cursor = previous_index[cursor]
    path.reverse()
    path = collapse_same_frame(path)

    club_time = float(club_event["timestamp"])
    lingering_hits = count_lingering_hits(path, frames, club_time)
    track_score = best_scores[terminal_index] - 1.2 * lingering_hits
    metrics = compute_path_metrics(path)

    return AnchorCandidate(
        club_event_index=club_event_index,
        club_time=club_time,
        club_distance_m=detection_value(club_event["detection"], "distance"),
        club_magnitude=detection_value(club_event["detection"], "magnitude"),
        path=path,
        all_post_hits=all_post_hits,
        track_score=track_score,
        selection_score=selection_score(track_score, metrics, lingering_hits, rolling_ball_dt_ms),
        lingering_hits=lingering_hits,
        metrics=metrics,
    )


def classify_quality(metrics: PathMetrics, lingering_hits: int) -> str:
    if (
        metrics.point_count >= 4
        and metrics.distance_gain_m >= 1.2
        and metrics.start_ms <= 120.0
        and metrics.monotonicity >= 0.75
    ):
        return "strong"
    if (
        metrics.point_count >= 3
        and metrics.distance_gain_m >= 1.5
        and metrics.start_ms <= 120.0
        and metrics.monotonicity >= 0.75
        and lingering_hits <= 4
    ):
        return "strong"
    if metrics.point_count >= 2 and metrics.distance_gain_m >= 0.4 and metrics.monotonicity >= 0.5:
        return "partial"
    return "weak"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def analyze_shot(shot_number: int, shot_bundle: dict[str, Any]) -> ShotReview:
    buffer_entry = shot_bundle["buffer"]
    capture_entry = shot_bundle.get("capture", {})
    detected_entry = shot_bundle.get("shot", {})
    frames = sorted(
        _validate_frames(shot_number, buffer_entry),
        key=lambda frame: frame["timestamp"],
    )

    rolling_ball_dt_ms: float | None = None
    if "ball_timestamp_ms" in capture_entry and "club_timestamp_ms" in capture_entry:
        rolling_ball_dt_ms = float(capture_entry["ball_timestamp_ms"])
        rolling_ball_dt_ms -= float(capture_entry["club_timestamp_ms"])

    club_events = find_club_events(frames)
    if not club_events:
        raise ValueError(f"No club event found for shot {shot_number}.")

    anchor_candidates = [
        extract_anchor_candidate(frames, club_event, index, rolling_ball_dt_ms)
        for index, club_event in enumerate(club_events)
    ]
    anchor = max(anchor_candidates, key=lambda candidate: candidate.selection_score)

    ball_angle = buffer_entry.get("ball_angle") or {}
    sanity_check = ball_angle.get("sanity_check") or {}

    return ShotReview(
        shot_number=shot_number,
        club_label=str(detected_entry.get("club") or "unknown"),
        logged_ball_speed_mph=_optional_float(detected_entry.get("ball_speed_mph")),
        logged_club_speed_mph=_optional_float(detected_entry.get("club_speed_mph")),
        logged_launch_angle_deg=_optional_float(detected_entry.get("launch_angle_vertical")),
        logged_launch_confidence=_optional_float(detected_entry.get("launch_angle_confidence")),
        logged_ball_angle_frames=_optional_int(ball_angle.get("num_frames")),
        logged_ball_angle_accepted=_optional_bool(ball_angle.get("accepted")),
        expected_launch_deg=_optional_float(sanity_check.get("expected_launch_deg")),
        allowed_delta_deg=_optional_float(sanity_check.get("allowed_delta_deg")),
        observed_delta_deg=_optional_float(sanity_check.get("delta_deg")),
        rolling_ball_dt_ms=rolling_ball_dt_ms,
        anchor=anchor,
        quality=classify_quality(anchor.metrics, anchor.lingering_hits),
    )


def analyze_session(
    session_path: Path,
) -> tuple[dict[str, Any], list[ShotReview]]:
    session_meta, shots = load_session(session_path)
    reviewable_shots = []
    missing_buffer_shots = []

    for shot_number, shot_bundle in shots.items():
        if "buffer" not in shot_bundle:
            missing_buffer_shots.append(shot_number)
            continue
        reviewable_shots.append((shot_number, shot_bundle))

    if not reviewable_shots:
        if missing_buffer_shots:
            raise ValueError(
                f"{session_path} has shots but no kld7_buffer entries. "
                "This review workflow only works on session logs that include K-LD7 frame buffers."
            )
        raise ValueError(f"{session_path} contains no reviewable shots.")

    if missing_buffer_shots:
        session_meta["_review_warnings"] = [
            (
                "Skipped shots without kld7_buffer data: "
                + ", ".join(str(shot_number) for shot_number in missing_buffer_shots)
            )
        ]

    results = [
        analyze_shot(shot_number, shot_bundle) for shot_number, shot_bundle in reviewable_shots
    ]
    return session_meta, results
