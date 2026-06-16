#!/usr/bin/env python3
"""Frame-level raw ADC diagnostics for K-LD7 RADC captures.

Usage:
    uv run --no-project --with numpy --with pyserial python scripts/analysis/diagnose_kld7_raw_adc.py \
        session_logs/kld7_radc_20260428_123644.pkl

    uv run --no-project --with numpy --with pyserial python scripts/analysis/diagnose_kld7_raw_adc.py \
        capture.pkl --ball-speed-mph 112.4 --orientation horizontal
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from openflight.kld7.radc import (  # noqa: E402
    RADCFrameDiagnostics,
    radc_capture_diagnostics,
)

CHANNELS = ("f1a_i", "f1a_q", "f2a_i", "f2a_q", "f1b_i", "f1b_q")


def load_capture(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected pickle to contain a dict, got {type(data).__name__}")
    return data


def frames_from_capture(data: dict) -> list[dict]:
    frames = data.get("frames")
    if isinstance(frames, list):
        return frames
    captures = data.get("captures")
    if isinstance(captures, list):
        return captures
    return []


def infer_ball_speed_mph(data: dict) -> float | None:
    for group_key in ("ops243_shots", "ops243_captures"):
        for item in data.get(group_key) or []:
            if not isinstance(item, dict):
                continue
            speed = item.get("ball_speed_mph")
            if speed is None:
                continue
            try:
                speed_f = float(speed)
            except (TypeError, ValueError):
                continue
            if speed_f > 0:
                return speed_f
    return None


def _fmt(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    return value


def diagnostics_row(diag: RADCFrameDiagnostics) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_index": diag.frame_index,
        "timestamp": _fmt(diag.timestamp),
        "valid_payload": diag.valid_payload,
        "reason": diag.reason or "",
        "target_bands": ";".join(f"{lo}:{hi}" for lo, hi in diag.target_bands),
        "expected_bin": diag.expected_bin if diag.expected_bin is not None else "",
        "peak_bin": diag.peak_bin if diag.peak_bin is not None else "",
        "bin_error": diag.bin_error if diag.bin_error is not None else "",
        "peak_velocity_kmh": _fmt(diag.peak_velocity_kmh),
        "peak_ball_speed_mph": _fmt(diag.peak_ball_speed_mph),
        "speed_error_mph": _fmt(diag.speed_error_mph),
        "peak_magnitude": _fmt(diag.peak_magnitude),
        "noise_floor": _fmt(diag.noise_floor),
        "snr_linear": _fmt(diag.snr_linear),
        "snr_db": _fmt(diag.snr_db),
        "angle_peak_deg": _fmt(diag.angle_peak_deg),
        "angle_centroid_deg": _fmt(diag.angle_centroid_deg),
        "phase_coherence": _fmt(diag.phase_coherence),
        "peak_width_bins": diag.peak_width_bins,
        "warnings": ";".join(diag.warnings),
    }

    for name in CHANNELS:
        stats = diag.channel_stats.get(name)
        if stats is None:
            row[f"{name}_mean"] = ""
            row[f"{name}_std"] = ""
            row[f"{name}_min"] = ""
            row[f"{name}_max"] = ""
            row[f"{name}_clip_low_frac"] = ""
            row[f"{name}_clip_high_frac"] = ""
            continue
        row[f"{name}_mean"] = _fmt(stats.mean)
        row[f"{name}_std"] = _fmt(stats.std)
        row[f"{name}_min"] = stats.min_code
        row[f"{name}_max"] = stats.max_code
        row[f"{name}_clip_low_frac"] = _fmt(stats.clipped_low_frac)
        row[f"{name}_clip_high_frac"] = _fmt(stats.clipped_high_frac)

    for name in ("f1a", "f2a"):
        stats = diag.iq_stats.get(name) or {}
        row[f"{name}_i_std"] = _fmt(stats.get("i_std"))
        row[f"{name}_q_std"] = _fmt(stats.get("q_std"))
        row[f"{name}_q_to_i_std_ratio"] = _fmt(stats.get("q_to_i_std_ratio"))
        row[f"{name}_iq_correlation"] = _fmt(stats.get("iq_correlation"))

    return row


def write_diagnostics_csv(path: Path, diagnostics: list[RADCFrameDiagnostics]) -> None:
    rows = [diagnostics_row(diag) for diag in diagnostics]
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = list(diagnostics_row(
            RADCFrameDiagnostics(
                frame_index=0,
                timestamp=None,
                has_radc=False,
                valid_payload=False,
                reason="empty",
                target_bands=(),
                expected_bin=None,
                peak_bin=None,
                peak_velocity_kmh=None,
                peak_ball_speed_mph=None,
                speed_error_mph=None,
                peak_magnitude=0.0,
                noise_floor=0.0,
                snr_linear=0.0,
                snr_db=0.0,
                bin_error=None,
                angle_peak_deg=None,
                angle_centroid_deg=None,
                phase_coherence=None,
                peak_width_bins=0,
                channel_stats={},
                iq_stats={},
                warnings=(),
            ),
        ).keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ops_shot_windows(
    data: dict,
    frames: list[dict],
    ms_before: float,
    ms_after: float,
) -> list[dict[str, object]]:
    shots = data.get("ops243_shots") or []
    out: list[dict[str, object]] = []
    for idx, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            continue
        shot_ts = shot.get("timestamp")
        if shot_ts is None:
            continue
        try:
            shot_time = float(shot_ts)
        except (TypeError, ValueError):
            continue
        start = shot_time - ms_before / 1000.0
        end = shot_time + ms_after / 1000.0
        window_frames = [
            frame for frame in frames
            if frame.get("radc") is not None
            and frame.get("timestamp") is not None
            and start <= float(frame["timestamp"]) <= end
        ]
        out.append({
            "shot_index": idx,
            "timestamp": shot_time,
            "ball_speed_mph": shot.get("ball_speed_mph"),
            "club_speed_mph": shot.get("club_speed_mph"),
            "frames": window_frames,
        })
    return out


def write_shot_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "shot_index",
        "timestamp",
        "ball_speed_mph",
        "club_speed_mph",
        "frame_count",
        "peak_frame_count",
        "expected_bin",
        "target_bands",
        "median_snr_db",
        "max_snr_db",
        "median_phase_coherence",
        "median_abs_bin_error",
        "median_abs_speed_error_mph",
        "top_peak_bins",
        "warnings_by_type",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose raw ADC quality in K-LD7 RADC .pkl captures.",
    )
    parser.add_argument("capture", type=Path, help="Path to K-LD7 RADC .pkl capture")
    parser.add_argument("--ball-speed-mph", type=float, default=None)
    parser.add_argument("--speed-tolerance-mph", type=float, default=10.0)
    parser.add_argument("--orientation", choices=("vertical", "horizontal"), default=None)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--max-speed-kmh", type=float, default=100.0)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--shot-windows", action="store_true")
    parser.add_argument("--shot-ms-before", type=float, default=1200.0)
    parser.add_argument("--shot-ms-after", type=float, default=400.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data = load_capture(args.capture)
    frames = frames_from_capture(data)
    if args.limit_frames is not None:
        frames = frames[: args.limit_frames]
    if not frames:
        raise SystemExit("No frames found in capture.")

    metadata = data.get("metadata") or {}
    orientation = args.orientation or metadata.get("orientation")
    ball_speed_mph = args.ball_speed_mph
    if ball_speed_mph is None:
        ball_speed_mph = infer_ball_speed_mph(data)

    output_dir = args.output_dir or args.capture.parent / f"raw_adc_diagnostics_{args.capture.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics, summary = radc_capture_diagnostics(
        frames,
        fft_size=args.fft_size,
        max_speed_kmh=args.max_speed_kmh,
        ops243_ball_speed_mph=ball_speed_mph,
        speed_tolerance_mph=args.speed_tolerance_mph,
        orientation=orientation,
    )

    csv_path = output_dir / "radc_frame_diagnostics.csv"
    json_path = output_dir / "radc_summary.json"
    write_diagnostics_csv(csv_path, diagnostics)
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    shot_summary_rows: list[dict[str, object]] = []
    if args.shot_windows:
        shot_dir = output_dir / "shots"
        shot_dir.mkdir(parents=True, exist_ok=True)
        for shot in ops_shot_windows(
            data,
            frames,
            ms_before=args.shot_ms_before,
            ms_after=args.shot_ms_after,
        ):
            shot_index = int(shot["shot_index"])
            shot_frames = shot["frames"]
            shot_speed = shot.get("ball_speed_mph")
            try:
                shot_speed_f = float(shot_speed) if shot_speed is not None else None
            except (TypeError, ValueError):
                shot_speed_f = None

            shot_diagnostics, shot_summary = radc_capture_diagnostics(
                shot_frames,
                fft_size=args.fft_size,
                max_speed_kmh=args.max_speed_kmh,
                ops243_ball_speed_mph=shot_speed_f,
                speed_tolerance_mph=args.speed_tolerance_mph,
                orientation=orientation,
            )
            stem = f"shot_{shot_index:02d}"
            write_diagnostics_csv(shot_dir / f"{stem}_frame_diagnostics.csv", shot_diagnostics)
            with (shot_dir / f"{stem}_summary.json").open("w") as f:
                json.dump(shot_summary, f, indent=2, sort_keys=True)

            shot_summary_rows.append({
                "shot_index": shot_index,
                "timestamp": shot.get("timestamp"),
                "ball_speed_mph": shot_speed_f,
                "club_speed_mph": shot.get("club_speed_mph"),
                "frame_count": shot_summary["frame_count"],
                "peak_frame_count": shot_summary["peak_frame_count"],
                "expected_bin": shot_summary["expected_bin"],
                "target_bands": json.dumps(shot_summary["target_bands"]),
                "median_snr_db": shot_summary["median_snr_db"],
                "max_snr_db": shot_summary["max_snr_db"],
                "median_phase_coherence": shot_summary["median_phase_coherence"],
                "median_abs_bin_error": shot_summary["median_abs_bin_error"],
                "median_abs_speed_error_mph": shot_summary["median_abs_speed_error_mph"],
                "top_peak_bins": json.dumps(shot_summary["peak_bin_histogram_top"]),
                "warnings_by_type": json.dumps(shot_summary["warnings_by_type"]),
            })

        write_shot_summary_csv(output_dir / "shot_summaries.csv", shot_summary_rows)

    print("=" * 64)
    print(f"Raw ADC diagnostics: {args.capture.name}")
    print("=" * 64)
    print(f"Frames:       {summary['frame_count']}")
    print(f"RADC frames:  {summary['radc_frame_count']}")
    print(f"Peak frames:  {summary['peak_frame_count']}")
    print(f"Orientation:  {orientation or 'unknown'}")
    print(f"Ball speed:   {ball_speed_mph:.1f} mph" if ball_speed_mph else "Ball speed:   none")
    print(f"Target band:  {summary['target_bands']}")
    print(f"Expected bin: {summary['expected_bin']}")
    print(f"Median SNR:   {summary['median_snr_db']}")
    print(f"Max SNR:      {summary['max_snr_db']}")
    print(f"Median coh.:  {summary['median_phase_coherence']}")
    print(f"Median bin err: {summary['median_abs_bin_error']}")
    print(f"Median speed err: {summary['median_abs_speed_error_mph']}")
    print(f"Top peak bins: {summary['peak_bin_histogram_top']}")
    print(f"Warnings:     {summary['warnings_by_type']}")
    if args.shot_windows:
        print(f"Shot windows: {len(shot_summary_rows)}")
    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    if args.shot_windows:
        print(f"Shot CSV: {output_dir / 'shot_summaries.csv'}")


if __name__ == "__main__":
    main()
