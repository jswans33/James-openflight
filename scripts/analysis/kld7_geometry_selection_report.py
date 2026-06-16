#!/usr/bin/env python
"""Offline K-LD7 RADC frame-selection and range-assisted geometry report.

This tool is intentionally more verbose and more configurable than the live
selector. It is for answering "what frames could we have used?" from a session
JSONL that contains raw K-LD7 RADC payloads.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from openflight.kld7.geometry import (
    GEOM_BALL_ABOVE_RADAR_FT,
    GEOM_FLIGHT_T_MAX_S,
    GEOM_PAIR_SINGLE_FRAME_FALLBACK_RMSE_DEG,
    MPH_TO_FTS,
    fit_launch_angle_geometric,
    fit_launch_angle_single_frame_geometric,
)
from openflight.kld7.radc import (
    OPS_ANCHORED_PEAK_MIN_SNR,
    RADC_PAYLOAD_BYTES,
    _centroid_angle_for_peak,
    _find_peak_in_bands,
    _find_peak_near_expected_bin,
    _peak_neighborhood_indices,
    _phase_coherence_for_peak,
    ball_bin_range_from_speed,
    bin_to_velocity_kmh,
    circular_bin_distance,
    compute_fft_complex,
    expected_ball_bin_from_speed,
    parse_radc_payload,
    per_bin_angle_deg,
    spectrum_from_channel_ffts,
    to_complex_iq,
)


@dataclass(frozen=True)
class ReportConfig:
    orientation: str = "vertical"
    ball_distance_ft: float = 5.0
    mount_deg: float = 10.0
    angle_offset_deg: float = 2.5
    ball_above_radar_ft: float = GEOM_BALL_ABOVE_RADAR_FT
    fft_size: int = 2048
    max_speed_kmh: float = 100.0
    dc_mask_bins: int = 8
    speed_tolerance_mph: float = 25.0
    spectrum_source: str = "f1a"
    centroid_floor_frac: float = 0.5
    report_time_min_ms: float = -80.0
    report_time_max_ms: float = 170.0
    report_bin_error_max: int = 220
    ops_bin_outlier_tol: int = 25
    ops_anchored_peak_min_snr: float = OPS_ANCHORED_PEAK_MIN_SNR
    anchor_time_min_ms: float = 20.0
    anchor_time_max_ms: float = 100.0
    anchor_snr_min: float = 5.0
    anchor_bin_error_max: int = 50
    neighbor_time_min_ms: float = 0.0
    neighbor_time_max_ms: float = 125.0
    neighbor_snr_min: float = 3.0
    neighbor_bin_error_max: int = 80
    early_neighbor_time_max_ms: float = 20.0
    early_neighbor_bin_error_max: int = 175
    neighbor_frame_gap: int = 2
    min_rising_deg: float = 0.0
    max_selected_frames: int = 2
    clock_error_ms: float = 20.0
    shift_step_ms: float = 1.0
    sensitivity_ms: float = 10.0
    kld7_range_m: float = 5.0
    f1b_range_bias_ft: float = 0.0
    f1b_phase_mode: str = "f1b_minus_f1a"
    range_unwrap_bin_error_max: int = 80
    range_unwrap_snr_min: float = 3.0
    range_unwrap_backtrack_ft: float = 0.75

    @property
    def unambiguous_range_ft(self) -> float:
        return self.kld7_range_m * 3.28084


@dataclass
class FrameReport:
    shot_number: int
    frame_index: int
    timestamp: float
    t_ms: float
    expected_bin: int | None = None
    peak_bin: int | None = None
    peak_source: str = ""
    bin_error: int | None = None
    speed_mph: float | None = None
    speed_error_mph: float | None = None
    peak_magnitude: float | None = None
    noise_floor: float | None = None
    snr: float | None = None
    snr_db: float | None = None
    angle_peak_deg: float | None = None
    angle_centroid_deg: float | None = None
    bearing_deg: float | None = None
    elevation_deg: float | None = None
    phase_coherence: float | None = None
    peak_width_bins: int | None = None
    f1b_phase_rad: float | None = None
    f1b_range_raw_ft: float | None = None
    f1b_range_ft: float | None = None
    f1b_range_unwrapped_ft: float | None = None
    f1b_range_unwrap_count: int = 0
    f1b_same_bin_snr: float | None = None
    f1b_peak_bin: int | None = None
    f1b_peak_bin_error: int | None = None
    anchored_peak_bin: int | None = None
    anchored_bin_error: int | None = None
    anchored_speed_mph: float | None = None
    anchored_snr: float | None = None
    anchored_bearing_deg: float | None = None
    broad_peak_bin: int | None = None
    broad_bin_error: int | None = None
    broad_speed_mph: float | None = None
    broad_snr: float | None = None
    broad_bearing_deg: float | None = None
    peak_selectable: bool = True
    selection_role: str = ""
    status: str = "invalid"
    reasons: list[str] = field(default_factory=list)

    @property
    def selectable(self) -> bool:
        return (
            self.peak_selectable
            and self.peak_bin is not None
            and self.bin_error is not None
            and self.snr is not None
        )


@dataclass(frozen=True)
class FitResult:
    shift_ms: float
    launch_angle_deg: float | None
    bearing_rmse_deg: float | None
    range_rmse_ft: float | None
    range_residuals_ft: list[float]
    frame_count: int
    method: str

    @property
    def score(self) -> float:
        if self.launch_angle_deg is None:
            return math.inf
        if self.range_rmse_ft is not None:
            return self.range_rmse_ft
        if self.bearing_rmse_deg is not None:
            return self.bearing_rmse_deg
        return math.inf


@dataclass
class ShotReport:
    shot_number: int
    ball_speed_mph: float | None
    impact_timestamp: float | None
    impact_timestamp_source: str
    logged_launch_angle_deg: float | None
    logged_angle_source: str | None
    logged_ball_angle_deg: float | None
    logged_ball_angle_confidence: float | None
    logged_ball_angle_accepted: bool | None
    logged_ball_angle_selection_reason: str | None
    logged_ball_angle_num_frames: int | None
    logged_radc_selection_available: bool
    logged_radc_estimator: str | None
    logged_radc_selection_path: str | None
    logged_radc_selected_frame_indices: list[int]
    logged_radc_selected_t_ms: list[float]
    logged_radc_selected_bin_errors: list[int | None]
    logged_radc_fit_rmse_deg: float | None
    frame_count: int
    considered_frame_count: int
    selected_frame_indices: list[int]
    selection_method: str
    selection_notes: list[str]
    nominal_fit: FitResult | None
    best_range_fit: FitResult | None
    minus_sensitivity_fit: FitResult | None
    plus_sensitivity_fit: FitResult | None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_radc(frame: dict[str, Any]) -> bytes | None:
    raw = frame.get("radc")
    if isinstance(raw, bytes):
        return raw if len(raw) == RADC_PAYLOAD_BYTES else None
    radc_b64 = frame.get("radc_b64")
    if not isinstance(radc_b64, str):
        return None
    try:
        decoded = base64.b64decode(radc_b64, validate=True)
    except ValueError:
        return None
    return decoded if len(decoded) == RADC_PAYLOAD_BYTES else None


def _mph_from_bin(bin_index: int, config: ReportConfig) -> float:
    velocity_kmh = bin_to_velocity_kmh(bin_index, config.fft_size, config.max_speed_kmh)
    return (2.0 * config.max_speed_kmh + velocity_kmh) / 1.609


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _f1b_phase_range_ft(
    f1a_fft: np.ndarray,
    f1b_fft: np.ndarray,
    spectrum: np.ndarray,
    peak_bin: int,
    peak_val: float,
    peak_band: tuple[int, int] | None,
    config: ReportConfig,
) -> tuple[float, float, float]:
    indices = _peak_neighborhood_indices(
        spectrum,
        peak_bin,
        peak_val,
        peak_band,
        half_width=4,
        floor_frac=None,
    )
    weights = np.maximum(spectrum[indices], 0.0)
    if float(np.sum(weights)) <= 0:
        weights = np.ones_like(weights)
    if config.f1b_phase_mode == "f1a_minus_f1b":
        cross = np.sum(weights * f1a_fft[indices] * np.conj(f1b_fft[indices]))
    else:
        cross = np.sum(weights * f1b_fft[indices] * np.conj(f1a_fft[indices]))
    phase = float(np.angle(cross))
    raw_range_ft = (phase % (2.0 * math.pi)) / (2.0 * math.pi) * config.unambiguous_range_ft
    return phase, raw_range_ft, raw_range_ft - config.f1b_range_bias_ft


def _predicted_range_ft(
    launch_angle_deg: float,
    flight_time_s: float,
    ball_speed_mph: float,
    config: ReportConfig,
) -> float:
    v_fts = ball_speed_mph * MPH_TO_FTS
    alpha_rad = math.radians(launch_angle_deg)
    x_ft = config.ball_distance_ft + v_fts * math.cos(alpha_rad) * flight_time_s
    y_ft = config.ball_above_radar_ft + v_fts * math.sin(alpha_rad) * flight_time_s
    return math.hypot(x_ft, y_ft)


def _load_session_rows(
    session_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    shots: dict[int, dict[str, Any]] = {}
    buffers: dict[tuple[int, str], dict[str, Any]] = {}
    with session_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {session_path}: {error.msg}"
                ) from error
            row_type = row.get("type")
            shot_number = row.get("shot_number")
            if shot_number is None:
                continue
            shot_number = int(shot_number)
            if row_type == "shot_detected":
                shots[shot_number] = row
            elif row_type == "kld7_buffer":
                orientation = str(row.get("orientation") or "")
                if orientation:
                    buffers[(shot_number, orientation)] = row
    return shots, buffers


def _impact_timestamp(
    shot: dict[str, Any] | None,
    buffer_entry: dict[str, Any],
) -> tuple[float | None, str]:
    if shot:
        impact = _to_float(shot.get("impact_timestamp"))
        if impact is not None:
            return impact, "shot_detected.impact_timestamp"
    shot_timestamp = _to_float(buffer_entry.get("shot_timestamp"))
    if shot_timestamp is not None:
        return shot_timestamp, "kld7_buffer.shot_timestamp"
    return None, "missing"


def _coerce_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _coerce_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _coerce_optional_int_list(value: Any) -> list[int | None]:
    if not isinstance(value, list):
        return []
    result: list[int | None] = []
    for item in value:
        if item is None:
            result.append(None)
            continue
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            result.append(None)
    return result


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    return int(numeric) if numeric is not None else None


def _logged_ball_angle_selection(buffer_entry: dict[str, Any]) -> dict[str, Any]:
    ball_angle = buffer_entry.get("ball_angle")
    if not isinstance(ball_angle, dict):
        return {
            "logged_ball_angle_deg": None,
            "logged_ball_angle_confidence": None,
            "logged_ball_angle_accepted": None,
            "logged_ball_angle_selection_reason": None,
            "logged_ball_angle_num_frames": None,
            "logged_radc_selection_available": False,
            "logged_radc_estimator": None,
            "logged_radc_selection_path": None,
            "logged_radc_selected_frame_indices": [],
            "logged_radc_selected_t_ms": [],
            "logged_radc_selected_bin_errors": [],
            "logged_radc_fit_rmse_deg": None,
        }

    radc_selection = ball_angle.get("radc_selection")
    if not isinstance(radc_selection, dict):
        radc_selection = {}

    return {
        "logged_ball_angle_deg": _to_float(
            ball_angle.get("vertical_deg") or ball_angle.get("horizontal_deg")
        ),
        "logged_ball_angle_confidence": _to_float(ball_angle.get("confidence")),
        "logged_ball_angle_accepted": (
            bool(ball_angle.get("accepted")) if "accepted" in ball_angle else None
        ),
        "logged_ball_angle_selection_reason": ball_angle.get("selection_reason"),
        "logged_ball_angle_num_frames": _to_int(ball_angle.get("num_frames")),
        "logged_radc_selection_available": bool(radc_selection),
        "logged_radc_estimator": radc_selection.get("estimator"),
        "logged_radc_selection_path": radc_selection.get("selection_path"),
        "logged_radc_selected_frame_indices": _coerce_int_list(
            radc_selection.get("selected_frame_indices")
        ),
        "logged_radc_selected_t_ms": _coerce_float_list(radc_selection.get("selected_t_ms")),
        "logged_radc_selected_bin_errors": _coerce_optional_int_list(
            radc_selection.get("selected_bin_errors")
        ),
        "logged_radc_fit_rmse_deg": _to_float(radc_selection.get("geom_fit_rmse_deg")),
    }


def extract_frame_report(
    shot_number: int,
    frame_index: int,
    frame: dict[str, Any],
    impact_timestamp: float,
    ball_speed_mph: float,
    config: ReportConfig,
) -> FrameReport:
    timestamp = float(frame["timestamp"])
    report = FrameReport(
        shot_number=shot_number,
        frame_index=frame_index,
        timestamp=timestamp,
        t_ms=(timestamp - impact_timestamp) * 1000.0,
    )
    radc = _decode_radc(frame)
    if radc is None:
        report.reasons.append("missing_or_invalid_radc")
        return report

    try:
        channels = parse_radc_payload(radc)
    except ValueError:
        report.reasons.append("invalid_radc_payload")
        return report

    f1a_fft = compute_fft_complex(
        to_complex_iq(channels["f1a_i"], channels["f1a_q"]),
        fft_size=config.fft_size,
        dc_mask_bins=config.dc_mask_bins,
    )
    f2a_fft = compute_fft_complex(
        to_complex_iq(channels["f2a_i"], channels["f2a_q"]),
        fft_size=config.fft_size,
        dc_mask_bins=config.dc_mask_bins,
    )
    f1b_fft = compute_fft_complex(
        to_complex_iq(channels["f1b_i"], channels["f1b_q"]),
        fft_size=config.fft_size,
        dc_mask_bins=config.dc_mask_bins,
    )
    spectrum = spectrum_from_channel_ffts(
        f1a_fft,
        f2a_fft,
        f1b_fft,
        source=config.spectrum_source,
    )
    expected_bin = expected_ball_bin_from_speed(
        ball_speed_mph,
        config.fft_size,
        config.max_speed_kmh,
    )
    bands = ball_bin_range_from_speed(
        ball_speed_mph,
        config.speed_tolerance_mph,
        config.fft_size,
        config.max_speed_kmh,
    )
    anchored_peak_bin, anchored_peak_val, anchored_peak_band = _find_peak_near_expected_bin(
        spectrum,
        bands,
        expected_bin,
        config.ops_bin_outlier_tol,
        config.fft_size,
    )
    broad_peak_bin, broad_peak_val, broad_peak_band = _find_peak_in_bands(
        spectrum,
        tuple(bands),
    )
    positive = spectrum[spectrum > 0]
    noise_floor = float(np.median(positive)) if positive.size else 0.0
    report.expected_bin = expected_bin
    report.noise_floor = noise_floor
    angles = per_bin_angle_deg(f1a_fft, f2a_fft)

    def describe_peak(
        peak_bin: int | None,
        peak_val: float,
        peak_band: tuple[int, int] | None,
    ) -> dict[str, float | int | None] | None:
        if peak_bin is None or peak_val <= 0:
            return None
        centroid_angle, peak_width = _centroid_angle_for_peak(
            angles,
            spectrum,
            peak_bin,
            peak_val,
            peak_band,
            config.centroid_floor_frac,
        )
        phase_coherence = _phase_coherence_for_peak(
            f1a_fft,
            f2a_fft,
            spectrum,
            peak_bin,
            peak_val,
            peak_band,
            coherence_bins=4,
        )
        speed_mph = _mph_from_bin(peak_bin, config)
        snr = float(peak_val / noise_floor) if noise_floor > 0 else 0.0
        return {
            "peak_bin": int(peak_bin),
            "peak_val": float(peak_val),
            "bin_error": circular_bin_distance(peak_bin, expected_bin, config.fft_size),
            "speed_mph": speed_mph,
            "speed_error_mph": speed_mph - ball_speed_mph,
            "snr": snr,
            "snr_db": float(10.0 * math.log10(snr)) if snr > 0 else 0.0,
            "angle_peak_deg": float(angles[peak_bin]),
            "angle_centroid_deg": float(centroid_angle),
            "bearing_deg": float(centroid_angle + config.angle_offset_deg),
            "elevation_deg": float(centroid_angle + config.angle_offset_deg + config.mount_deg),
            "phase_coherence": phase_coherence,
            "peak_width_bins": int(peak_width),
        }

    anchored = describe_peak(anchored_peak_bin, anchored_peak_val, anchored_peak_band)
    broad = describe_peak(broad_peak_bin, broad_peak_val, broad_peak_band)
    if anchored is not None:
        report.anchored_peak_bin = int(anchored["peak_bin"])
        report.anchored_bin_error = int(anchored["bin_error"])
        report.anchored_speed_mph = float(anchored["speed_mph"])
        report.anchored_snr = float(anchored["snr"])
        report.anchored_bearing_deg = float(anchored["bearing_deg"])
    if broad is not None:
        report.broad_peak_bin = int(broad["peak_bin"])
        report.broad_bin_error = int(broad["bin_error"])
        report.broad_speed_mph = float(broad["speed_mph"])
        report.broad_snr = float(broad["snr"])
        report.broad_bearing_deg = float(broad["bearing_deg"])

    if anchored is None:
        report.reasons.append("no_ops_anchored_peak")
        return report

    peak_bin = int(anchored["peak_bin"])
    peak_val = float(anchored["peak_val"])
    peak_band = anchored_peak_band
    anchored_snr = float(anchored["snr"])
    if anchored_snr < config.neighbor_snr_min:
        peak_source = "ops_anchored_low_snr"
    elif anchored_snr >= config.ops_anchored_peak_min_snr:
        peak_source = "ops_anchored"
    else:
        peak_source = "ops_anchored_weak"
    f1b_phase, f1b_range_raw_ft, f1b_range_ft = _f1b_phase_range_ft(
        f1a_fft,
        f1b_fft,
        spectrum,
        peak_bin,
        peak_val,
        peak_band,
        config,
    )

    f1b_mag = np.abs(f1b_fft)
    f1b_peak_bin, _, _ = _find_peak_near_expected_bin(
        f1b_mag,
        bands,
        expected_bin,
        config.report_bin_error_max,
        config.fft_size,
    )
    f1b_positive = f1b_mag[f1b_mag > 0]
    f1b_noise = float(np.median(f1b_positive)) if f1b_positive.size else 0.0

    snr = anchored_snr
    report.peak_bin = int(peak_bin)
    report.peak_source = peak_source
    report.bin_error = int(anchored["bin_error"])
    report.speed_mph = float(anchored["speed_mph"])
    report.speed_error_mph = float(anchored["speed_error_mph"])
    report.peak_magnitude = float(peak_val)
    report.snr = snr
    report.snr_db = float(anchored["snr_db"])
    report.angle_peak_deg = float(anchored["angle_peak_deg"])
    report.angle_centroid_deg = float(anchored["angle_centroid_deg"])
    report.bearing_deg = float(anchored["bearing_deg"])
    report.elevation_deg = float(anchored["elevation_deg"])
    report.phase_coherence = (
        float(anchored["phase_coherence"]) if anchored["phase_coherence"] is not None else None
    )
    report.peak_width_bins = int(anchored["peak_width_bins"])
    report.f1b_phase_rad = f1b_phase
    report.f1b_range_raw_ft = f1b_range_raw_ft
    report.f1b_range_ft = f1b_range_ft
    report.f1b_same_bin_snr = float(f1b_mag[peak_bin] / f1b_noise) if f1b_noise > 0 else None
    report.f1b_peak_bin = int(f1b_peak_bin) if f1b_peak_bin is not None else None
    report.f1b_peak_bin_error = (
        circular_bin_distance(int(f1b_peak_bin), peak_bin, config.fft_size)
        if f1b_peak_bin is not None
        else None
    )
    if snr < config.neighbor_snr_min:
        report.peak_selectable = False
        report.reasons.append("ops_anchored_snr_too_low")
        return report
    report.status = "candidate"
    return report


def _anchor_failure_reasons(frame: FrameReport, config: ReportConfig) -> list[str]:
    reasons: list[str] = []
    if not frame.selectable:
        return list(frame.reasons) or ["not_selectable"]
    if not (config.anchor_time_min_ms <= frame.t_ms <= config.anchor_time_max_ms):
        reasons.append("outside_anchor_time_window")
    if frame.snr is not None and frame.snr < config.anchor_snr_min:
        reasons.append("anchor_snr_too_low")
    if frame.bin_error is not None and frame.bin_error > config.anchor_bin_error_max:
        reasons.append("anchor_bin_error_too_large")
    return reasons


def _neighbor_bin_limit(frame: FrameReport, config: ReportConfig) -> int:
    if frame.t_ms <= config.early_neighbor_time_max_ms:
        return config.early_neighbor_bin_error_max
    return config.neighbor_bin_error_max


def _neighbor_failure_reasons(
    frame: FrameReport,
    anchor: FrameReport,
    config: ReportConfig,
) -> list[str]:
    reasons: list[str] = []
    if not frame.selectable:
        return list(frame.reasons) or ["not_selectable"]
    if not (config.neighbor_time_min_ms <= frame.t_ms <= config.neighbor_time_max_ms):
        reasons.append("outside_neighbor_time_window")
    if abs(frame.frame_index - anchor.frame_index) > config.neighbor_frame_gap:
        reasons.append("not_adjacent_enough")
    if frame.snr is not None and frame.snr < config.neighbor_snr_min:
        reasons.append("neighbor_snr_too_low")
    if frame.bin_error is not None and frame.bin_error > _neighbor_bin_limit(frame, config):
        reasons.append("neighbor_bin_error_too_large")
    first, second = (frame, anchor) if frame.t_ms < anchor.t_ms else (anchor, frame)
    if (
        first.bearing_deg is not None
        and second.bearing_deg is not None
        and second.bearing_deg < first.bearing_deg + config.min_rising_deg
    ):
        reasons.append("not_rising")
    return reasons


def select_candidate_frames(
    frames: list[FrameReport],
    config: ReportConfig,
) -> tuple[list[FrameReport], list[str]]:
    notes: list[str] = []
    for frame in frames:
        frame.selection_role = ""
        if frame.selectable:
            frame.status = "candidate"

    primary_passed = [
        frame
        for frame in frames
        if frame.selectable
        and config.anchor_time_min_ms <= frame.t_ms <= config.anchor_time_max_ms
        and frame.bin_error is not None
        and frame.bin_error <= config.anchor_bin_error_max
        and frame.snr is not None
        and frame.snr >= config.neighbor_snr_min
    ]
    early_context = [
        frame
        for frame in frames
        if frame.selectable
        and config.neighbor_time_min_ms <= frame.t_ms < config.early_neighbor_time_max_ms
        and frame.bin_error is not None
        and frame.bin_error <= config.anchor_bin_error_max
        and frame.snr is not None
        and frame.snr >= config.neighbor_snr_min
    ]

    anchor_pool = [
        frame
        for frame in primary_passed
        if frame.peak_source != "ops_anchored_weak"
        and frame.snr is not None
        and frame.snr >= config.anchor_snr_min
    ]
    if not anchor_pool:
        notes.append("no_anchor_frame")
        for frame in frames:
            if frame.selectable:
                frame.status = "rejected"
                frame.reasons = list(
                    dict.fromkeys(frame.reasons + _anchor_failure_reasons(frame, config))
                )
        return [], notes

    anchor = sorted(
        anchor_pool,
        key=lambda frame: (
            frame.bin_error if frame.bin_error is not None else 99999,
            -(frame.snr or 0.0),
            -(-1.0 if frame.phase_coherence is None else frame.phase_coherence),
        ),
    )[0]
    anchor.selection_role = "anchor"

    def valid_rising_pair(
        first: FrameReport,
        second: FrameReport,
    ) -> tuple[FrameReport, FrameReport] | None:
        partner = first if second is anchor else second
        partner_snr = partner.snr or 0.0
        if partner.peak_source == "ops_anchored_weak":
            min_partner_snr = config.neighbor_snr_min
        else:
            min_partner_snr = max(config.neighbor_snr_min, 0.5 * (anchor.snr or 0.0))
        if partner_snr < min_partner_snr:
            partner.reasons = list(dict.fromkeys(partner.reasons + ["neighbor_snr_too_low"]))
            return None
        if (
            first.bearing_deg is None
            or second.bearing_deg is None
            or second.bearing_deg < first.bearing_deg + config.min_rising_deg
        ):
            partner.reasons = list(dict.fromkeys(partner.reasons + ["not_rising"]))
            return None
        return first, second

    def find_neighbor_pair(pool: list[FrameReport]) -> tuple[FrameReport, FrameReport] | None:
        by_frame = sorted(pool, key=lambda frame: frame.frame_index)
        try:
            anchor_pos = by_frame.index(anchor)
        except ValueError:
            return None
        if anchor_pos > 0:
            pair = valid_rising_pair(by_frame[anchor_pos - 1], anchor)
            if pair is not None:
                return pair
        if anchor_pos + 1 < len(by_frame):
            pair = valid_rising_pair(anchor, by_frame[anchor_pos + 1])
            if pair is not None:
                return pair
        return None

    pair = find_neighbor_pair(primary_passed)
    if pair is None and early_context:
        pair = find_neighbor_pair(primary_passed + early_context)

    selected = sorted(pair if pair is not None else [anchor], key=lambda frame: frame.t_ms)
    if len(selected) == 1:
        notes.append("anchor_only_no_rising_neighbor")
    else:
        notes.append(f"selected_{len(selected)}_frames")
    for frame in selected:
        if frame is not anchor:
            frame.selection_role = "neighbor"
        frame.status = "selected"
        frame.reasons = []
    selected_ids = {id(frame) for frame in selected}
    for frame in frames:
        if id(frame) in selected_ids or not frame.selectable:
            continue
        frame.status = "rejected"
        frame.reasons = list(
            dict.fromkeys(
                frame.reasons
                + _anchor_failure_reasons(frame, config)
                + _neighbor_failure_reasons(frame, anchor, config)
            )
        )
    return selected, notes


def _broad_high_snr_frame(
    frame: FrameReport,
    ball_speed_mph: float,
    config: ReportConfig,
) -> FrameReport:
    """Return a frame view whose primary peak is the broad strongest in-band peak."""
    out = replace(frame, reasons=list(frame.reasons))
    out.selection_role = ""
    out.status = "invalid"
    out.peak_selectable = True
    out.f1b_phase_rad = None
    out.f1b_range_raw_ft = None
    out.f1b_range_ft = None
    out.f1b_range_unwrapped_ft = None
    out.f1b_range_unwrap_count = 0
    out.f1b_same_bin_snr = None
    out.f1b_peak_bin = None
    out.f1b_peak_bin_error = None

    if (
        frame.broad_peak_bin is None
        or frame.broad_bin_error is None
        or frame.broad_speed_mph is None
        or frame.broad_snr is None
        or frame.broad_bearing_deg is None
    ):
        out.peak_selectable = False
        out.reasons = list(dict.fromkeys(out.reasons + ["no_broad_peak"]))
        return out

    out.peak_bin = frame.broad_peak_bin
    out.peak_source = "broad_high_snr"
    out.bin_error = frame.broad_bin_error
    out.speed_mph = frame.broad_speed_mph
    out.speed_error_mph = frame.broad_speed_mph - ball_speed_mph
    out.peak_magnitude = None
    out.snr = frame.broad_snr
    out.snr_db = float(10.0 * math.log10(frame.broad_snr)) if frame.broad_snr > 0 else 0.0
    out.angle_peak_deg = None
    out.angle_centroid_deg = frame.broad_bearing_deg - config.angle_offset_deg
    out.bearing_deg = frame.broad_bearing_deg
    out.elevation_deg = frame.broad_bearing_deg + config.mount_deg
    out.phase_coherence = None
    out.peak_width_bins = None
    out.reasons = []

    if frame.broad_snr < config.neighbor_snr_min:
        out.peak_selectable = False
        out.reasons.append("broad_snr_too_low")
        return out

    out.status = "candidate"
    return out


def broad_high_snr_frames(
    frames: list[FrameReport],
    ball_speed_mph: float,
    config: ReportConfig,
) -> list[FrameReport]:
    """Return frame rows for the older broad/high-SNR exploratory selector."""
    return [_broad_high_snr_frame(frame, ball_speed_mph, config) for frame in frames]


def select_high_snr_candidate_frames(
    frames: list[FrameReport],
    config: ReportConfig,
) -> tuple[list[FrameReport], list[str]]:
    """Select broad/high-SNR frames without using OPS bin error as the primary gate."""
    notes: list[str] = []
    for frame in frames:
        frame.selection_role = ""
        if frame.selectable:
            frame.status = "candidate"

    candidates = [
        frame
        for frame in frames
        if frame.selectable
        and config.neighbor_time_min_ms <= frame.t_ms <= config.neighbor_time_max_ms
        and frame.snr is not None
        and frame.snr >= config.neighbor_snr_min
    ]
    anchor_pool = [
        frame
        for frame in candidates
        if config.anchor_time_min_ms <= frame.t_ms <= config.anchor_time_max_ms
        and frame.snr is not None
        and frame.snr >= config.anchor_snr_min
    ]
    if not anchor_pool:
        notes.append("no_high_snr_anchor_frame")
        for frame in frames:
            if frame.selectable:
                frame.status = "rejected"
                frame.reasons = list(
                    dict.fromkeys(frame.reasons + ["outside_high_snr_anchor_selection"])
                )
        return [], notes

    anchor = sorted(
        anchor_pool,
        key=lambda frame: (
            -(frame.snr or 0.0),
            abs(frame.t_ms - (config.anchor_time_min_ms + config.anchor_time_max_ms) / 2.0),
            frame.bin_error if frame.bin_error is not None else 99999,
        ),
    )[0]
    anchor.selection_role = "anchor"

    valid_pairs: list[tuple[float, FrameReport, FrameReport]] = []
    for partner in candidates:
        if partner is anchor:
            continue
        if abs(partner.frame_index - anchor.frame_index) > config.neighbor_frame_gap:
            continue
        first, second = (partner, anchor) if partner.t_ms < anchor.t_ms else (anchor, partner)
        if first.bearing_deg is None or second.bearing_deg is None:
            partner.reasons = list(dict.fromkeys(partner.reasons + ["missing_bearing"]))
            continue
        if second.bearing_deg < first.bearing_deg + config.min_rising_deg:
            partner.reasons = list(dict.fromkeys(partner.reasons + ["not_rising"]))
            continue
        valid_pairs.append((min(first.snr or 0.0, second.snr or 0.0), first, second))

    if valid_pairs:
        _, first, second = sorted(
            valid_pairs,
            key=lambda item: (
                -item[0],
                abs(item[1].frame_index - item[2].frame_index),
                item[1].frame_index,
            ),
        )[0]
        selected = [first, second]
        notes.append("selected_2_frames_high_snr")
    else:
        selected = [anchor]
        notes.append("anchor_only_high_snr_no_rising_neighbor")

    selected = sorted(selected, key=lambda frame: frame.t_ms)
    for frame in selected:
        if frame is not anchor:
            frame.selection_role = "neighbor"
        frame.status = "selected"
        frame.reasons = []
    selected_ids = {id(frame) for frame in selected}
    for frame in frames:
        if id(frame) in selected_ids or not frame.selectable:
            continue
        frame.status = "rejected"
        if not frame.reasons:
            frame.reasons = ["not_selected_by_high_snr_selector"]
    return selected, notes


def _range_for_fit(frame: FrameReport) -> float | None:
    if frame.f1b_range_unwrapped_ft is not None:
        return frame.f1b_range_unwrapped_ft
    return frame.f1b_range_ft


def unwrap_f1b_ranges(frames: list[FrameReport], config: ReportConfig) -> None:
    """Add a modulo-unwrapped F1B range estimate to eligible frame rows.

    RADC FSK range is phase-derived, so it is expected to wrap at the
    unambiguous range for the active K-LD7 range setting. This is still an
    analysis assumption: only unwrap frames that are close enough to the OPS
    Doppler bin and have enough SNR to plausibly be the same ball target.
    """
    period = config.unambiguous_range_ft
    previous_unwrapped: float | None = None
    for frame in sorted(frames, key=lambda candidate: candidate.t_ms):
        frame.f1b_range_unwrapped_ft = frame.f1b_range_ft
        frame.f1b_range_unwrap_count = 0
        if (
            frame.f1b_range_ft is None
            or frame.bin_error is None
            or frame.bin_error > config.range_unwrap_bin_error_max
            or frame.snr is None
            or frame.snr < config.range_unwrap_snr_min
        ):
            continue

        unwrapped = frame.f1b_range_ft
        wraps = 0
        if frame.t_ms >= config.anchor_time_min_ms:
            while unwrapped < config.ball_distance_ft - config.range_unwrap_backtrack_ft:
                unwrapped += period
                wraps += 1
        if previous_unwrapped is not None:
            while unwrapped < previous_unwrapped - config.range_unwrap_backtrack_ft:
                unwrapped += period
                wraps += 1
        frame.f1b_range_unwrapped_ft = unwrapped
        frame.f1b_range_unwrap_count = wraps
        previous_unwrapped = unwrapped


def fit_selected_frames(
    selected: list[FrameReport],
    ball_speed_mph: float,
    shift_ms: float,
    config: ReportConfig,
) -> FitResult:
    per_frame = []
    for frame in selected:
        if frame.bearing_deg is None or frame.snr is None:
            continue
        t_s = (frame.t_ms + shift_ms) / 1000.0
        per_frame.append((t_s, frame.bearing_deg, max(frame.snr * frame.snr, 1.0)))

    launch_angle: float | None = None
    bearing_rmse: float | None = None
    method = "none"
    if len(per_frame) >= 2:
        geom = fit_launch_angle_geometric(
            per_frame,
            ball_speed_mph,
            config.ball_distance_ft,
            config.mount_deg,
            config.ball_above_radar_ft,
        )
        if geom is not None:
            launch_angle, bearing_rmse, _ = geom
            method = "geometry_2plus_frame"
    elif len(per_frame) == 1:
        single = fit_launch_angle_single_frame_geometric(
            per_frame[0],
            ball_speed_mph,
            config.ball_distance_ft,
            config.mount_deg,
            config.ball_above_radar_ft,
        )
        if single is not None:
            launch_angle, bearing_rmse = single
            method = "geometry_single_frame"

    residuals: list[float] = []
    if launch_angle is not None:
        for frame in selected:
            range_ft = _range_for_fit(frame)
            if range_ft is None:
                continue
            t_s = (frame.t_ms + shift_ms) / 1000.0
            if t_s <= 0:
                continue
            predicted = _predicted_range_ft(launch_angle, t_s, ball_speed_mph, config)
            residuals.append(predicted - range_ft)
    range_rmse = (
        math.sqrt(sum(resid * resid for resid in residuals) / len(residuals)) if residuals else None
    )
    return FitResult(
        shift_ms=round(float(shift_ms), 3),
        launch_angle_deg=launch_angle,
        bearing_rmse_deg=bearing_rmse,
        range_rmse_ft=range_rmse,
        range_residuals_ft=residuals,
        frame_count=len(per_frame),
        method=method,
    )


def apply_high_rmse_single_frame_fallback(
    selected: list[FrameReport],
    ball_speed_mph: float,
    config: ReportConfig,
    notes: list[str],
) -> list[FrameReport]:
    if len(selected) < 2:
        return selected
    nominal = fit_selected_frames(selected, ball_speed_mph, 0.0, config)
    if (
        nominal.bearing_rmse_deg is None
        or nominal.bearing_rmse_deg <= GEOM_PAIR_SINGLE_FRAME_FALLBACK_RMSE_DEG
    ):
        return selected

    strong_in_flight = [
        frame
        for frame in selected
        if frame.peak_source != "ops_anchored_weak"
        and 0.0 < frame.t_ms / 1000.0 <= GEOM_FLIGHT_T_MAX_S
    ]
    if not strong_in_flight:
        notes.append("high_rmse_pair_no_single_frame_fallback")
        return selected

    single = min(
        strong_in_flight,
        key=lambda frame: (
            frame.t_ms,
            frame.bin_error if frame.bin_error is not None else 99999,
            -(frame.snr or 0.0),
        ),
    )
    notes.append(f"high_rmse_pair_fallback_to_single_frame({nominal.bearing_rmse_deg:.2f}deg)")
    for frame in selected:
        if frame is single:
            frame.selection_role = "anchor"
            frame.status = "selected"
            frame.reasons = []
        else:
            frame.selection_role = ""
            frame.status = "rejected"
            frame.reasons = list(dict.fromkeys(frame.reasons + ["high_rmse_pair_fallback"]))
    return [single]


def best_range_shift_fit(
    selected: list[FrameReport],
    ball_speed_mph: float,
    config: ReportConfig,
) -> FitResult | None:
    if not selected:
        return None
    step = max(config.shift_step_ms, 0.1)
    shifts = np.arange(-config.clock_error_ms, config.clock_error_ms + step / 2.0, step)
    fits = [fit_selected_frames(selected, ball_speed_mph, float(shift), config) for shift in shifts]
    valid = [fit for fit in fits if fit.launch_angle_deg is not None]
    if not valid:
        return None
    return min(valid, key=lambda fit: (fit.score, abs(fit.shift_ms)))


def analyze_shot(
    shot_number: int,
    shot: dict[str, Any] | None,
    buffer_entry: dict[str, Any],
    config: ReportConfig,
) -> tuple[ShotReport, list[FrameReport]]:
    ball_speed = _to_float((shot or {}).get("ball_speed_mph"))
    impact_ts, impact_source = _impact_timestamp(shot, buffer_entry)
    logged_selection = _logged_ball_angle_selection(buffer_entry)
    raw_frames = buffer_entry.get("frames") or []
    if ball_speed is None or impact_ts is None:
        report = ShotReport(
            shot_number=shot_number,
            ball_speed_mph=ball_speed,
            impact_timestamp=impact_ts,
            impact_timestamp_source=impact_source,
            logged_launch_angle_deg=_to_float((shot or {}).get("launch_angle_vertical")),
            logged_angle_source=(shot or {}).get("launch_angle_vertical_source"),
            **logged_selection,
            frame_count=len(raw_frames),
            considered_frame_count=0,
            selected_frame_indices=[],
            selection_method="missing_ball_speed_or_impact_timestamp",
            selection_notes=[],
            nominal_fit=None,
            best_range_fit=None,
            minus_sensitivity_fit=None,
            plus_sensitivity_fit=None,
        )
        return report, []

    frames: list[FrameReport] = []
    for frame_index, frame in enumerate(raw_frames):
        timestamp = _to_float(frame.get("timestamp"))
        if timestamp is None:
            continue
        t_ms = (timestamp - impact_ts) * 1000.0
        if not (config.report_time_min_ms <= t_ms <= config.report_time_max_ms):
            continue
        frames.append(
            extract_frame_report(
                shot_number,
                frame_index,
                frame,
                impact_ts,
                ball_speed,
                config,
            )
        )

    unwrap_f1b_ranges(frames, config)
    selected, notes = select_candidate_frames(frames, config)
    selected = apply_high_rmse_single_frame_fallback(
        selected,
        ball_speed,
        config,
        notes,
    )
    nominal = fit_selected_frames(selected, ball_speed, 0.0, config) if selected else None
    best = best_range_shift_fit(selected, ball_speed, config)
    minus = (
        fit_selected_frames(selected, ball_speed, -config.sensitivity_ms, config)
        if selected
        else None
    )
    plus = (
        fit_selected_frames(selected, ball_speed, config.sensitivity_ms, config)
        if selected
        else None
    )
    method = "no_selection"
    if selected:
        method = "anchor_only" if len(selected) == 1 else f"{len(selected)}_frame_selection"
    report = ShotReport(
        shot_number=shot_number,
        ball_speed_mph=ball_speed,
        impact_timestamp=impact_ts,
        impact_timestamp_source=impact_source,
        logged_launch_angle_deg=_to_float((shot or {}).get("launch_angle_vertical")),
        logged_angle_source=(shot or {}).get("launch_angle_vertical_source"),
        **logged_selection,
        frame_count=len(raw_frames),
        considered_frame_count=len(frames),
        selected_frame_indices=[frame.frame_index for frame in selected],
        selection_method=method,
        selection_notes=notes,
        nominal_fit=nominal,
        best_range_fit=best,
        minus_sensitivity_fit=minus,
        plus_sensitivity_fit=plus,
    )
    return report, frames


def analyze_high_snr_variant(
    live_report: ShotReport,
    live_frames: list[FrameReport],
    config: ReportConfig,
) -> tuple[ShotReport, list[FrameReport]]:
    if live_report.ball_speed_mph is None or live_report.impact_timestamp is None:
        return (
            replace(
                live_report,
                selected_frame_indices=[],
                selection_method="missing_ball_speed_or_impact_timestamp",
                selection_notes=[],
                nominal_fit=None,
                best_range_fit=None,
                minus_sensitivity_fit=None,
                plus_sensitivity_fit=None,
            ),
            [],
        )

    frames = broad_high_snr_frames(live_frames, live_report.ball_speed_mph, config)
    selected, notes = select_high_snr_candidate_frames(frames, config)
    selected = apply_high_rmse_single_frame_fallback(
        selected,
        live_report.ball_speed_mph,
        config,
        notes,
    )
    nominal = (
        fit_selected_frames(selected, live_report.ball_speed_mph, 0.0, config) if selected else None
    )
    best = best_range_shift_fit(selected, live_report.ball_speed_mph, config)
    minus = (
        fit_selected_frames(
            selected,
            live_report.ball_speed_mph,
            -config.sensitivity_ms,
            config,
        )
        if selected
        else None
    )
    plus = (
        fit_selected_frames(
            selected,
            live_report.ball_speed_mph,
            config.sensitivity_ms,
            config,
        )
        if selected
        else None
    )
    method = "no_selection"
    if selected:
        method = "anchor_only_high_snr" if len(selected) == 1 else "2_frame_high_snr"
    return (
        replace(
            live_report,
            selected_frame_indices=[frame.frame_index for frame in selected],
            selection_method=method,
            selection_notes=notes,
            nominal_fit=nominal,
            best_range_fit=best,
            minus_sensitivity_fit=minus,
            plus_sensitivity_fit=plus,
        ),
        frames,
    )


def _fit_dict(prefix: str, fit: FitResult | None) -> dict[str, Any]:
    if fit is None:
        return {
            f"{prefix}_shift_ms": None,
            f"{prefix}_launch_angle_deg": None,
            f"{prefix}_bearing_rmse_deg": None,
            f"{prefix}_range_rmse_ft": None,
            f"{prefix}_method": None,
        }
    return {
        f"{prefix}_shift_ms": fit.shift_ms,
        f"{prefix}_launch_angle_deg": fit.launch_angle_deg,
        f"{prefix}_bearing_rmse_deg": fit.bearing_rmse_deg,
        f"{prefix}_range_rmse_ft": fit.range_rmse_ft,
        f"{prefix}_method": fit.method,
    }


def _join_values(values: list[Any]) -> str:
    return " ".join("" if value is None else str(value) for value in values)


def _logged_radc_csv_value(report: ShotReport, value: Any) -> Any:
    if report.logged_radc_selection_available:
        return value
    if report.logged_ball_angle_deg is not None or report.logged_ball_angle_num_frames is not None:
        return "not_logged_in_session"
    return ""


def _shot_csv_row(report: ShotReport) -> dict[str, Any]:
    logged_radc_selected_frame_indices = _join_values(report.logged_radc_selected_frame_indices)
    logged_radc_selected_t_ms = _join_values(report.logged_radc_selected_t_ms)
    logged_radc_selected_bin_errors = _join_values(report.logged_radc_selected_bin_errors)
    row: dict[str, Any] = {
        "shot_number": report.shot_number,
        "ball_speed_mph": report.ball_speed_mph,
        "logged_launch_angle_deg": report.logged_launch_angle_deg,
        "logged_angle_source": report.logged_angle_source,
        "logged_ball_angle_deg": report.logged_ball_angle_deg,
        "logged_ball_angle_confidence": report.logged_ball_angle_confidence,
        "logged_ball_angle_accepted": report.logged_ball_angle_accepted,
        "logged_ball_angle_selection_reason": report.logged_ball_angle_selection_reason,
        "logged_ball_angle_num_frames": report.logged_ball_angle_num_frames,
        "logged_radc_selection_available": report.logged_radc_selection_available,
        "logged_radc_estimator": _logged_radc_csv_value(report, report.logged_radc_estimator),
        "logged_radc_selection_path": _logged_radc_csv_value(
            report, report.logged_radc_selection_path
        ),
        "logged_radc_selected_frame_indices": _logged_radc_csv_value(
            report, logged_radc_selected_frame_indices
        ),
        "logged_radc_selected_t_ms": _logged_radc_csv_value(report, logged_radc_selected_t_ms),
        "logged_radc_selected_bin_errors": _logged_radc_csv_value(
            report, logged_radc_selected_bin_errors
        ),
        "logged_radc_fit_rmse_deg": _logged_radc_csv_value(report, report.logged_radc_fit_rmse_deg),
        "impact_timestamp_source": report.impact_timestamp_source,
        "frame_count": report.frame_count,
        "considered_frame_count": report.considered_frame_count,
        "selected_frame_indices": " ".join(map(str, report.selected_frame_indices)),
        "selection_method": report.selection_method,
        "selection_notes": ";".join(report.selection_notes),
    }
    row.update(_fit_dict("nominal", report.nominal_fit))
    row.update(_fit_dict("range_best", report.best_range_fit))
    row.update(_fit_dict("minus_10ms", report.minus_sensitivity_fit))
    row.update(_fit_dict("plus_10ms", report.plus_sensitivity_fit))
    return row


def _frame_csv_row(frame: FrameReport) -> dict[str, Any]:
    row = asdict(frame)
    row["reasons"] = ";".join(frame.reasons)
    return row


def _fieldnames_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = _fieldnames_for_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    session_path: Path,
    output_dir: Path,
    config: ReportConfig,
    shot_reports: list[ShotReport],
    frame_reports: list[FrameReport],
    live_shot_reports: list[ShotReport],
    live_frame_reports: list[FrameReport],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / "shots.csv", [_shot_csv_row(report) for report in shot_reports])
    _write_csv(output_dir / "frames.csv", [_frame_csv_row(frame) for frame in frame_reports])
    _write_csv(
        output_dir / "shots_live.csv",
        [_shot_csv_row(report) for report in live_shot_reports],
    )
    _write_csv(
        output_dir / "frames_live.csv",
        [_frame_csv_row(frame) for frame in live_frame_reports],
    )

    lines = [
        f"# K-LD7 Geometry Selection Report: `{session_path.name}`",
        "",
        "Generated files: `shots.csv`, `frames.csv`, `shots_live.csv`, "
        "`frames_live.csv`, `config.json`.",
        "",
        "`frames.csv`/`shots.csv` use the broad/high-SNR exploratory selector. "
        "`frames_live.csv`/`shots_live.csv` use the current OPS-bin/live-style replay.",
        "",
        "`Logged KLD7 frames` is populated only for sessions that recorded "
        "`ball_angle.radc_selection.selected_frame_indices`; older sessions keep "
        "the logged KLD7 angle and frame count but not exact frame IDs.",
        "",
        "| Shot | Ball mph | Logged KLD7 | Logged KLD7 frames | Replay selected | Method | Nominal | Range-best | Notes |",
        "|---:|---:|---:|---|---|---|---:|---:|---|",
    ]
    for report in shot_reports:
        logged_kld7 = _format_float(report.logged_ball_angle_deg, 1)
        if logged_kld7 and report.logged_ball_angle_num_frames is not None:
            logged_kld7 = f"{logged_kld7} ({report.logged_ball_angle_num_frames}f)"
        elif report.logged_ball_angle_num_frames is not None:
            logged_kld7 = f"({report.logged_ball_angle_num_frames}f)"
        logged_frames = _join_values(report.logged_radc_selected_frame_indices)
        if not logged_frames and report.logged_ball_angle_num_frames is not None:
            logged_frames = "not logged"
        nominal = _format_float(
            report.nominal_fit.launch_angle_deg if report.nominal_fit else None,
            1,
        )
        best = _format_float(
            report.best_range_fit.launch_angle_deg if report.best_range_fit else None,
            1,
        )
        best_shift = f" @ {report.best_range_fit.shift_ms:+.0f}ms" if report.best_range_fit else ""
        lines.append(
            "| "
            f"{report.shot_number} | "
            f"{_format_float(report.ball_speed_mph, 1)} | "
            f"{logged_kld7} | "
            f"{logged_frames} | "
            f"{' '.join(map(str, report.selected_frame_indices))} | "
            f"{report.selection_method} | "
            f"{nominal} | "
            f"{best}{best_shift} | "
            f"{'; '.join(report.selection_notes)} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_session(
    session_path: Path,
    config: ReportConfig,
) -> tuple[list[ShotReport], list[FrameReport], list[ShotReport], list[FrameReport]]:
    if not session_path.exists():
        raise ValueError(f"Session file not found: {session_path}")
    shots, buffers = _load_session_rows(session_path)
    shot_reports: list[ShotReport] = []
    frame_reports: list[FrameReport] = []
    live_shot_reports: list[ShotReport] = []
    live_frame_reports: list[FrameReport] = []
    for shot_number in sorted(set(shots) | {key[0] for key in buffers}):
        buffer_entry = buffers.get((shot_number, config.orientation))
        if buffer_entry is None:
            continue
        live_shot_report, live_frames = analyze_shot(
            shot_number,
            shots.get(shot_number),
            buffer_entry,
            config,
        )
        shot_report, frames = analyze_high_snr_variant(
            live_shot_report,
            live_frames,
            config,
        )
        shot_reports.append(shot_report)
        frame_reports.extend(frames)
        live_shot_reports.append(live_shot_report)
        live_frame_reports.extend(live_frames)
    if not shot_reports:
        raise ValueError(
            f"{session_path} has no {config.orientation!r} kld7_buffer rows to analyze"
        )
    return shot_reports, frame_reports, live_shot_reports, live_frame_reports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline K-LD7 RADC frame-selection and geometry report."
    )
    parser.add_argument("session", type=Path, help="OpenFlight session JSONL path")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--orientation", default="vertical", choices=["vertical", "horizontal"])
    parser.add_argument("--ball-distance-ft", type=float, default=5.0)
    parser.add_argument("--mount-deg", type=float, default=10.0)
    parser.add_argument("--angle-offset-deg", type=float, default=2.5)
    parser.add_argument("--ball-above-radar-ft", type=float, default=GEOM_BALL_ABOVE_RADAR_FT)
    parser.add_argument("--speed-tolerance-mph", type=float, default=25.0)
    parser.add_argument("--spectrum-source", default="f1a")
    parser.add_argument("--dc-mask-bins", type=int, default=8)
    parser.add_argument("--centroid-floor-frac", type=float, default=0.5)
    parser.add_argument("--report-time-min-ms", type=float, default=-80.0)
    parser.add_argument("--report-time-max-ms", type=float, default=170.0)
    parser.add_argument("--report-bin-error-max", type=int, default=220)
    parser.add_argument("--ops-bin-outlier-tol", type=int, default=25)
    parser.add_argument(
        "--ops-anchored-peak-min-snr", type=float, default=OPS_ANCHORED_PEAK_MIN_SNR
    )
    parser.add_argument("--anchor-time-min-ms", type=float, default=20.0)
    parser.add_argument("--anchor-time-max-ms", type=float, default=100.0)
    parser.add_argument("--anchor-snr-min", type=float, default=5.0)
    parser.add_argument("--anchor-bin-error-max", type=int, default=50)
    parser.add_argument("--neighbor-time-min-ms", type=float, default=0.0)
    parser.add_argument("--neighbor-time-max-ms", type=float, default=125.0)
    parser.add_argument("--neighbor-snr-min", type=float, default=3.0)
    parser.add_argument("--neighbor-bin-error-max", type=int, default=80)
    parser.add_argument("--early-neighbor-time-max-ms", type=float, default=20.0)
    parser.add_argument("--early-neighbor-bin-error-max", type=int, default=175)
    parser.add_argument("--neighbor-frame-gap", type=int, default=2)
    parser.add_argument("--min-rising-deg", type=float, default=0.0)
    parser.add_argument("--max-selected-frames", type=int, default=2)
    parser.add_argument("--clock-error-ms", type=float, default=20.0)
    parser.add_argument("--shift-step-ms", type=float, default=1.0)
    parser.add_argument("--sensitivity-ms", type=float, default=10.0)
    parser.add_argument("--kld7-range-m", type=float, default=5.0)
    parser.add_argument("--f1b-range-bias-ft", type=float, default=0.0)
    parser.add_argument(
        "--f1b-phase-mode",
        choices=["f1b_minus_f1a", "f1a_minus_f1b"],
        default="f1b_minus_f1a",
    )
    parser.add_argument("--range-unwrap-bin-error-max", type=int, default=80)
    parser.add_argument("--range-unwrap-snr-min", type=float, default=3.0)
    parser.add_argument("--range-unwrap-backtrack-ft", type=float, default=0.75)
    return parser


def config_from_args(args: argparse.Namespace) -> ReportConfig:
    return ReportConfig(
        orientation=args.orientation,
        ball_distance_ft=args.ball_distance_ft,
        mount_deg=args.mount_deg,
        angle_offset_deg=args.angle_offset_deg,
        ball_above_radar_ft=args.ball_above_radar_ft,
        speed_tolerance_mph=args.speed_tolerance_mph,
        spectrum_source=args.spectrum_source,
        dc_mask_bins=args.dc_mask_bins,
        centroid_floor_frac=args.centroid_floor_frac,
        report_time_min_ms=args.report_time_min_ms,
        report_time_max_ms=args.report_time_max_ms,
        report_bin_error_max=args.report_bin_error_max,
        ops_bin_outlier_tol=args.ops_bin_outlier_tol,
        ops_anchored_peak_min_snr=args.ops_anchored_peak_min_snr,
        anchor_time_min_ms=args.anchor_time_min_ms,
        anchor_time_max_ms=args.anchor_time_max_ms,
        anchor_snr_min=args.anchor_snr_min,
        anchor_bin_error_max=args.anchor_bin_error_max,
        neighbor_time_min_ms=args.neighbor_time_min_ms,
        neighbor_time_max_ms=args.neighbor_time_max_ms,
        neighbor_snr_min=args.neighbor_snr_min,
        neighbor_bin_error_max=args.neighbor_bin_error_max,
        early_neighbor_time_max_ms=args.early_neighbor_time_max_ms,
        early_neighbor_bin_error_max=args.early_neighbor_bin_error_max,
        neighbor_frame_gap=args.neighbor_frame_gap,
        min_rising_deg=args.min_rising_deg,
        max_selected_frames=args.max_selected_frames,
        clock_error_ms=args.clock_error_ms,
        shift_step_ms=args.shift_step_ms,
        sensitivity_ms=args.sensitivity_ms,
        kld7_range_m=args.kld7_range_m,
        f1b_range_bias_ft=args.f1b_range_bias_ft,
        f1b_phase_mode=args.f1b_phase_mode,
        range_unwrap_bin_error_max=args.range_unwrap_bin_error_max,
        range_unwrap_snr_min=args.range_unwrap_snr_min,
        range_unwrap_backtrack_ft=args.range_unwrap_backtrack_ft,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    output_dir = args.output_dir or (
        args.session.parent / f"{args.session.stem}_kld7_geometry_report"
    )
    shot_reports, frame_reports, live_shot_reports, live_frame_reports = analyze_session(
        args.session,
        config,
    )
    write_reports(
        args.session,
        output_dir,
        config,
        shot_reports,
        frame_reports,
        live_shot_reports,
        live_frame_reports,
    )
    selected = sum(1 for report in shot_reports if report.selected_frame_indices)
    multi = sum(1 for report in shot_reports if len(report.selected_frame_indices) >= 2)
    live_selected = sum(1 for report in live_shot_reports if report.selected_frame_indices)
    live_multi = sum(1 for report in live_shot_reports if len(report.selected_frame_indices) >= 2)
    print(
        f"Analyzed {len(shot_reports)} shots "
        f"(high-SNR: {selected} selected, {multi} with 2+ frames; "
        f"live-style: {live_selected} selected, {live_multi} with 2+ frames)"
    )
    print(f"Report written to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
