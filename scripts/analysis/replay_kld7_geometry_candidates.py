#!/usr/bin/env python3
"""Replay vertical K-LD7 raw ADC candidates through geometry ranking.

This analysis mode intentionally treats TrackMan as an evaluation label, not
as an input to candidate selection. It asks: if we enumerate physically
plausible RADC candidates near the OPS ball-speed bin and impact time, which
candidate or candidate pair would a geometry-only ranker choose?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from replay_kld7_trackman import TrackmanTarget, load_buffers, load_targets  # noqa: E402

from openflight.kld7.geometry import (  # noqa: E402
    fit_launch_angle_geometric,
    fit_launch_angle_single_frame_geometric,
)
from openflight.kld7.radc import (  # noqa: E402
    ball_bin_range_from_speed,
    circular_bin_distance,
    compute_fft_complex,
    expected_ball_bin_from_speed,
    parse_radc_payload,
    per_bin_angle_deg,
    spectrum_from_channel_ffts,
    to_complex_iq,
)


@dataclass(frozen=True)
class GeometryCandidate:
    shot_number: int
    club: str
    frame_index: int
    dt_ms: float
    bin_index: int
    expected_bin: int
    bin_error: int
    snr: float
    snr_db: float
    bearing_deg: float
    single_launch_deg: float
    single_resid_deg: float
    band_rank: int
    local_peak: bool
    local_prominence: float
    trackman_angle_deg: float
    single_abs_error_deg: float


@dataclass(frozen=True)
class GeometryPick:
    shot_number: int
    club: str
    trackman_angle_deg: float
    estimator: str
    launch_angle_deg: float | None
    abs_error_deg: float | None
    score: float | None
    frame_count: int
    frame_indices: str
    dt_ms: str
    bins: str
    bin_errors: str
    snrs: str
    fit_rmse_deg: float | None
    single_resid_deg: float | None
    reason: str


@dataclass(frozen=True)
class ScoreWeights:
    single_resid: float
    pair_rmse: float
    bin_error: float
    time: float
    band_rank: float
    local_peak_penalty: float
    snr_reward: float
    short_span_penalty: float


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_window_ms(raw: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected START,END")
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("window start must be <= end")
    return parts[0], parts[1]


def _band_bins(bands: list[tuple[int, int]]) -> np.ndarray:
    parts = [np.arange(lo, hi, dtype=np.int64) for lo, hi in bands if lo < hi]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts)


def _local_peak(spec: np.ndarray, bin_index: int, radius: int = 2) -> tuple[bool, float]:
    lo = max(0, bin_index - radius)
    hi = min(len(spec), bin_index + radius + 1)
    local_max = float(np.max(spec[lo:hi])) if hi > lo else 0.0
    value = float(spec[bin_index])
    if local_max <= 0.0:
        return False, 0.0
    return value >= local_max, value / local_max


def enumerate_candidates(
    target: TrackmanTarget,
    frames: list[dict[str, Any]],
    *,
    shot_timestamp: float,
    near_window_ms: tuple[float, float],
    speed_tolerance_mph: float,
    max_bin_error: int,
    min_snr: float,
    angle_offset_deg: float,
    mount_deg: float,
    distance_ft: float,
    fft_size: int,
) -> list[GeometryCandidate]:
    expected_bin = expected_ball_bin_from_speed(target.ball_speed_mph, fft_size=fft_size)
    bands = ball_bin_range_from_speed(
        target.ball_speed_mph,
        speed_tolerance_mph,
        fft_size=fft_size,
    )
    band_bins = _band_bins(bands)
    if band_bins.size == 0:
        return []

    candidates: list[GeometryCandidate] = []
    for frame_index, frame in enumerate(frames):
        timestamp = _to_float(frame.get("timestamp"))
        if timestamp is None:
            continue
        dt_ms = (timestamp - float(shot_timestamp)) * 1000.0
        if dt_ms < near_window_ms[0] or dt_ms > near_window_ms[1]:
            continue

        radc_raw = frame.get("radc")
        if radc_raw is None:
            continue
        try:
            channels = parse_radc_payload(radc_raw) if isinstance(radc_raw, bytes) else radc_raw
        except (KeyError, TypeError, ValueError):
            continue

        f1a_iq = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
        f2a_iq = to_complex_iq(channels["f2a_i"], channels["f2a_q"])
        f1b_iq = (
            to_complex_iq(channels["f1b_i"], channels["f1b_q"])
            if "f1b_i" in channels and "f1b_q" in channels
            else None
        )
        f1a_fft = compute_fft_complex(f1a_iq, fft_size=fft_size)
        f2a_fft = compute_fft_complex(f2a_iq, fft_size=fft_size)
        f1b_fft = compute_fft_complex(f1b_iq, fft_size=fft_size) if f1b_iq is not None else None
        spec = spectrum_from_channel_ffts(f1a_fft, f2a_fft, f1b_fft, source="f1a")
        positive = spec[spec > 0]
        if positive.size == 0:
            continue
        noise_floor = float(np.median(positive))
        if noise_floor <= 0.0:
            continue

        snrs = spec[band_bins] / noise_floor
        order = np.argsort(-snrs)
        band_rank_by_bin = {int(band_bins[idx]): rank + 1 for rank, idx in enumerate(order)}
        angles = per_bin_angle_deg(f1a_fft, f2a_fft)

        for bin_index_raw, snr_raw in zip(band_bins, snrs, strict=True):
            bin_index = int(bin_index_raw)
            snr = float(snr_raw)
            if snr < min_snr:
                continue
            bin_error = circular_bin_distance(bin_index, expected_bin, fft_size=fft_size)
            if bin_error > max_bin_error:
                continue

            bearing = float(angles[bin_index] + angle_offset_deg)
            single = fit_launch_angle_single_frame_geometric(
                (dt_ms / 1000.0, bearing, snr * snr),
                target.ball_speed_mph,
                distance_ft,
                mount_deg,
            )
            if single is None:
                continue
            launch, resid = single
            is_local_peak, local_prominence = _local_peak(spec, bin_index)
            candidates.append(
                GeometryCandidate(
                    shot_number=target.shot_number,
                    club=target.club,
                    frame_index=frame_index,
                    dt_ms=dt_ms,
                    bin_index=bin_index,
                    expected_bin=expected_bin,
                    bin_error=bin_error,
                    snr=snr,
                    snr_db=10.0 * math.log10(snr) if snr > 0.0 else 0.0,
                    bearing_deg=bearing,
                    single_launch_deg=launch,
                    single_resid_deg=resid,
                    band_rank=band_rank_by_bin[bin_index],
                    local_peak=is_local_peak,
                    local_prominence=local_prominence,
                    trackman_angle_deg=target.trackman_angle_deg,
                    single_abs_error_deg=abs(launch - target.trackman_angle_deg),
                )
            )
    return candidates


def _candidate_pre_score(
    candidate: GeometryCandidate,
    *,
    target_time_ms: float,
    weights: ScoreWeights,
) -> float:
    time_penalty = abs(candidate.dt_ms - target_time_ms) / 50.0
    bin_penalty = candidate.bin_error / 25.0
    snr_reward = min(math.log2(max(candidate.snr, 1.0)), 5.0) / 5.0
    rank_penalty = min(candidate.band_rank - 1, 50) / 25.0
    local_penalty = 0.0 if candidate.local_peak else weights.local_peak_penalty
    return (
        candidate.single_resid_deg * weights.single_resid
        + bin_penalty * weights.bin_error
        + time_penalty * weights.time
        + rank_penalty * weights.band_rank
        + local_penalty
        - snr_reward * weights.snr_reward
    )


def rank_single(
    candidates: list[GeometryCandidate],
    *,
    target_time_ms: float,
    weights: ScoreWeights,
    launch_min_deg: float | None,
    launch_max_deg: float | None,
) -> GeometryPick | None:
    filtered = [
        candidate
        for candidate in candidates
        if (launch_min_deg is None or candidate.single_launch_deg >= launch_min_deg)
        and (launch_max_deg is None or candidate.single_launch_deg <= launch_max_deg)
    ]
    if not filtered:
        return None
    best = min(
        filtered,
        key=lambda cand: _candidate_pre_score(cand, target_time_ms=target_time_ms, weights=weights),
    )
    score = _candidate_pre_score(best, target_time_ms=target_time_ms, weights=weights)
    return GeometryPick(
        shot_number=best.shot_number,
        club=best.club,
        trackman_angle_deg=best.trackman_angle_deg,
        estimator="single",
        launch_angle_deg=round(best.single_launch_deg, 3),
        abs_error_deg=abs(best.single_launch_deg - best.trackman_angle_deg),
        score=score,
        frame_count=1,
        frame_indices=str([best.frame_index]),
        dt_ms=json.dumps([round(best.dt_ms, 1)]),
        bins=str([best.bin_index]),
        bin_errors=str([best.bin_error]),
        snrs=json.dumps([round(best.snr, 3)]),
        fit_rmse_deg=None,
        single_resid_deg=best.single_resid_deg,
        reason="ok",
    )


def _pair_score(
    first: GeometryCandidate,
    second: GeometryCandidate,
    *,
    fit_rmse: float,
    target_time_ms: float,
    weights: ScoreWeights,
) -> float:
    avg_bin = (first.bin_error + second.bin_error) / 2.0
    avg_rank = (min(first.band_rank, 50) + min(second.band_rank, 50)) / 2.0
    min_snr = min(first.snr, second.snr)
    local_penalty = (0.0 if first.local_peak else weights.local_peak_penalty) + (
        0.0 if second.local_peak else weights.local_peak_penalty
    )
    time_penalty = (
        abs(first.dt_ms - target_time_ms) + abs(second.dt_ms - target_time_ms)
    ) / 100.0
    span_ms = abs(second.dt_ms - first.dt_ms)
    span_penalty = weights.short_span_penalty if span_ms < 8.0 else 0.0
    snr_reward = min(math.log2(max(min_snr, 1.0)), 5.0) / 5.0
    return (
        fit_rmse * weights.pair_rmse
        + (avg_bin / 25.0) * weights.bin_error
        + (avg_rank / 25.0) * weights.band_rank
        + time_penalty * weights.time
        + span_penalty
        + local_penalty
        - snr_reward * weights.snr_reward
    )


def rank_pair(
    candidates: list[GeometryCandidate],
    *,
    ball_speed_mph: float,
    mount_deg: float,
    distance_ft: float,
    target_time_ms: float,
    max_pairs_per_frame: int,
    weights: ScoreWeights,
    launch_min_deg: float | None,
    launch_max_deg: float | None,
) -> tuple[GeometryPick | None, list[dict[str, Any]]]:
    by_frame: dict[int, list[GeometryCandidate]] = {}
    for candidate in candidates:
        by_frame.setdefault(candidate.frame_index, []).append(candidate)
    trimmed: list[GeometryCandidate] = []
    for frame_candidates in by_frame.values():
        trimmed.extend(
            sorted(
                frame_candidates,
                key=lambda cand: _candidate_pre_score(
                    cand,
                    target_time_ms=target_time_ms,
                    weights=weights,
                ),
            )[:max_pairs_per_frame]
        )

    pair_rows: list[dict[str, Any]] = []
    best_pick: GeometryPick | None = None
    best_score = math.inf
    ordered = sorted(trimmed, key=lambda cand: (cand.dt_ms, cand.frame_index, cand.bin_index))
    for left_idx, first in enumerate(ordered):
        for second in ordered[left_idx + 1 :]:
            if second.frame_index == first.frame_index:
                continue
            if second.dt_ms <= first.dt_ms:
                continue
            fit = fit_launch_angle_geometric(
                [
                    (first.dt_ms / 1000.0, first.bearing_deg, first.snr * first.snr),
                    (second.dt_ms / 1000.0, second.bearing_deg, second.snr * second.snr),
                ],
                ball_speed_mph,
                distance_ft,
                mount_deg,
            )
            if fit is None:
                continue
            launch, rmse, _ = fit
            if launch_min_deg is not None and launch < launch_min_deg:
                continue
            if launch_max_deg is not None and launch > launch_max_deg:
                continue
            score = _pair_score(
                first,
                second,
                fit_rmse=rmse,
                target_time_ms=target_time_ms,
                weights=weights,
            )
            row = {
                "shot_number": first.shot_number,
                "club": first.club,
                "trackman_angle_deg": first.trackman_angle_deg,
                "launch_angle_deg": launch,
                "abs_error_deg": abs(launch - first.trackman_angle_deg),
                "score": score,
                "fit_rmse_deg": rmse,
                "frame_indices": [first.frame_index, second.frame_index],
                "dt_ms": [first.dt_ms, second.dt_ms],
                "bins": [first.bin_index, second.bin_index],
                "bin_errors": [first.bin_error, second.bin_error],
                "snrs": [first.snr, second.snr],
                "bearings": [first.bearing_deg, second.bearing_deg],
                "local_peaks": [first.local_peak, second.local_peak],
                "band_ranks": [first.band_rank, second.band_rank],
            }
            pair_rows.append(row)
            if score < best_score:
                best_score = score
                best_pick = GeometryPick(
                    shot_number=first.shot_number,
                    club=first.club,
                    trackman_angle_deg=first.trackman_angle_deg,
                    estimator="pair",
                    launch_angle_deg=round(launch, 3),
                    abs_error_deg=abs(launch - first.trackman_angle_deg),
                    score=score,
                    frame_count=2,
                    frame_indices=str(row["frame_indices"]),
                    dt_ms=json.dumps([round(v, 1) for v in row["dt_ms"]]),
                    bins=str(row["bins"]),
                    bin_errors=str(row["bin_errors"]),
                    snrs=json.dumps([round(v, 3) for v in row["snrs"]]),
                    fit_rmse_deg=rmse,
                    single_resid_deg=None,
                    reason="ok",
                )
    return best_pick, sorted(pair_rows, key=lambda row: row["score"])


def _empty_pick(target: TrackmanTarget, reason: str) -> GeometryPick:
    return GeometryPick(
        shot_number=target.shot_number,
        club=target.club,
        trackman_angle_deg=target.trackman_angle_deg,
        estimator="none",
        launch_angle_deg=None,
        abs_error_deg=None,
        score=None,
        frame_count=0,
        frame_indices="[]",
        dt_ms="[]",
        bins="[]",
        bin_errors="[]",
        snrs="[]",
        fit_rmse_deg=None,
        single_resid_deg=None,
        reason=reason,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[GeometryPick]) -> dict[str, Any]:
    detected = [row for row in rows if row.abs_error_deg is not None]
    errors = [float(row.abs_error_deg) for row in detected]
    by_estimator: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        by_estimator[row.estimator] = by_estimator.get(row.estimator, 0) + 1
        by_reason[row.reason] = by_reason.get(row.reason, 0) + 1
    return {
        "targets": len(rows),
        "detected": len(detected),
        "detection_rate": len(detected) / len(rows) if rows else 0.0,
        "mae": statistics.fmean(errors) if errors else None,
        "median_abs_error": statistics.median(errors) if errors else None,
        "p90_abs_error": sorted(errors)[round(0.9 * (len(errors) - 1))] if errors else None,
        "max_abs_error": max(errors) if errors else None,
        "within_1_deg": sum(error <= 1.0 for error in errors),
        "within_2_deg": sum(error <= 2.0 for error in errors),
        "within_5_deg": sum(error <= 5.0 for error in errors),
        "estimator_counts": dict(sorted(by_estimator.items())),
        "reason_counts": dict(sorted(by_reason.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openflight", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--rows-output", required=True, type=Path)
    parser.add_argument("--candidates-output", required=True, type=Path)
    parser.add_argument("--pairs-output", required=True, type=Path)
    parser.add_argument("--detail-shot", type=int, default=None)
    parser.add_argument("--near-window-ms", type=_parse_window_ms, default=(0.0, 150.0))
    parser.add_argument("--speed-tolerance", type=float, default=10.0)
    parser.add_argument("--max-bin-error", type=int, default=25)
    parser.add_argument("--min-snr", type=float, default=2.0)
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--mount-deg", type=float, default=10.0)
    parser.add_argument("--distance-ft", type=float, default=5.0)
    parser.add_argument("--target-time-ms", type=float, default=55.0)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--max-pairs-per-frame", type=int, default=8)
    parser.add_argument("--single-resid-weight", type=float, default=3.0)
    parser.add_argument("--pair-rmse-weight", type=float, default=6.0)
    parser.add_argument("--bin-error-weight", type=float, default=1.0)
    parser.add_argument("--time-weight", type=float, default=1.0)
    parser.add_argument("--band-rank-weight", type=float, default=1.0)
    parser.add_argument("--local-peak-penalty", type=float, default=0.75)
    parser.add_argument("--snr-reward-weight", type=float, default=1.0)
    parser.add_argument("--short-span-penalty", type=float, default=1.0)
    parser.add_argument("--launch-min-deg", type=float, default=None)
    parser.add_argument("--launch-max-deg", type=float, default=None)
    args = parser.parse_args()
    weights = ScoreWeights(
        single_resid=args.single_resid_weight,
        pair_rmse=args.pair_rmse_weight,
        bin_error=args.bin_error_weight,
        time=args.time_weight,
        band_rank=args.band_rank_weight,
        local_peak_penalty=args.local_peak_penalty,
        snr_reward=args.snr_reward_weight,
        short_span_penalty=args.short_span_penalty,
    )

    targets = [
        target
        for target in load_targets(args.comparison, axis="vertical")
        if target.orientation == "vertical"
    ]
    buffers = load_buffers(args.openflight)
    shot_timestamps: dict[int, float] = {}
    with args.openflight.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("type") != "kld7_buffer" or entry.get("orientation") != "vertical":
                continue
            shot_number = int(entry.get("shot_number") or -1)
            timestamp = _to_float(entry.get("shot_timestamp"))
            if shot_number >= 1 and timestamp is not None:
                shot_timestamps[shot_number] = timestamp

    rows: list[GeometryPick] = []
    candidate_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for target in targets:
        buffer = buffers.get((target.shot_number, "vertical"))
        if not buffer:
            rows.append(_empty_pick(target, "missing_buffer"))
            continue

        shot_timestamp = shot_timestamps.get(target.shot_number)
        if shot_timestamp is None:
            rows.append(_empty_pick(target, "missing_shot_timestamp"))
            continue

        candidates = enumerate_candidates(
            target,
            buffer,
            shot_timestamp=shot_timestamp,
            near_window_ms=args.near_window_ms,
            speed_tolerance_mph=args.speed_tolerance,
            max_bin_error=args.max_bin_error,
            min_snr=args.min_snr,
            angle_offset_deg=args.angle_offset,
            mount_deg=args.mount_deg,
            distance_ft=args.distance_ft,
            fft_size=args.fft_size,
        )
        if args.detail_shot is None or target.shot_number == args.detail_shot:
            candidate_rows.extend(asdict(candidate) for candidate in candidates)

        single_pick = rank_single(
            candidates,
            target_time_ms=args.target_time_ms,
            weights=weights,
            launch_min_deg=args.launch_min_deg,
            launch_max_deg=args.launch_max_deg,
        )
        pair_pick, pairs = rank_pair(
            candidates,
            ball_speed_mph=target.ball_speed_mph,
            mount_deg=args.mount_deg,
            distance_ft=args.distance_ft,
            target_time_ms=args.target_time_ms,
            max_pairs_per_frame=args.max_pairs_per_frame,
            weights=weights,
            launch_min_deg=args.launch_min_deg,
            launch_max_deg=args.launch_max_deg,
        )
        if args.detail_shot is None or target.shot_number == args.detail_shot:
            pair_rows.extend(pairs[:200])

        pick = pair_pick or single_pick
        rows.append(pick if pick is not None else _empty_pick(target, "no_geometry_candidate"))

    _write_csv(args.rows_output, [asdict(row) for row in rows])
    _write_csv(args.candidates_output, candidate_rows)
    _write_csv(args.pairs_output, pair_rows)

    payload = {
        "params": {
            "near_window_ms": list(args.near_window_ms),
            "speed_tolerance_mph": args.speed_tolerance,
            "max_bin_error": args.max_bin_error,
            "min_snr": args.min_snr,
            "angle_offset_deg": args.angle_offset,
            "mount_deg": args.mount_deg,
            "distance_ft": args.distance_ft,
            "target_time_ms": args.target_time_ms,
            "max_pairs_per_frame": args.max_pairs_per_frame,
            "launch_min_deg": args.launch_min_deg,
            "launch_max_deg": args.launch_max_deg,
            "score_weights": asdict(weights),
        },
        "summary": _summary(rows),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
