#!/usr/bin/env python3
"""Analyze K-LD7 RADC captures — FFT, CFAR detection, comparison with PDAT.

Usage:
    uv run --no-project --with numpy --with matplotlib python scripts/analyze_kld7_radc.py capture.pkl
    uv run --no-project --with numpy --with matplotlib python scripts/analyze_kld7_radc.py capture.pkl --shot-windows
    uv run --no-project --with numpy --with matplotlib python scripts/analyze_kld7_radc.py capture.pkl --csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from kld7_radc_lib import (
    ADC_MIDPOINT,
    process_radc_frame,
    compare_radc_vs_pdat,
    compute_spectrum,
    to_complex_iq,
    parse_radc_payload,
)


def load_capture(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_active_windows(frames: list[dict], min_pdat_targets: int = 3, context_frames: int = 10):
    """Find frame ranges with activity (likely swings)."""
    windows = []
    in_window = False
    start = 0

    for i, frame in enumerate(frames):
        n_pdat = len(frame.get("pdat") or [])
        if n_pdat >= min_pdat_targets:
            if not in_window:
                start = max(0, i - context_frames)
                in_window = True
        else:
            if in_window:
                end = min(len(frames), i + context_frames)
                windows.append((start, end))
                in_window = False

    if in_window:
        windows.append((start, len(frames)))

    # Merge overlapping windows
    merged = []
    for window in windows:
        if merged and window[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], window[1]))
        else:
            merged.append(window)

    return merged


def plot_spectrogram(frames: list[dict], title: str, output_path: Path, fft_size: int = 2048):
    """Plot a time-frequency spectrogram from RADC frames."""
    spectra = []
    timestamps = []

    for frame in frames:
        radc = frame.get("radc")
        if radc is None:
            continue
        if isinstance(radc, bytes):
            channels = parse_radc_payload(radc)
        else:
            channels = radc
        iq = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
        spec = compute_spectrum(iq, fft_size=fft_size)
        spectra.append(10.0 * np.log10(spec + 1e-10))
        timestamps.append(frame["timestamp"])

    if not spectra:
        print(f"  No RADC frames for spectrogram: {title}")
        return

    spectrogram = np.array(spectra).T  # (fft_size, n_frames)
    t0 = timestamps[0]
    time_axis = np.array(timestamps) - t0

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        spectrogram[: fft_size // 2, :],
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], 0, fft_size // 2],
        cmap="viridis",
        vmin=np.percentile(spectrogram, 20),
        vmax=np.percentile(spectrogram, 99),
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("FFT bin (0 = DC, higher = faster)")
    fig.colorbar(ax.images[0], label="Power (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_detection_timeline(
    frames: list[dict],
    title: str,
    output_path: Path,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    cfar_threshold: float = 8.0,
):
    """Plot RADC detections vs PDAT detections over time."""
    radc_times = []
    radc_velocities = []
    radc_snrs = []
    pdat_times = []
    pdat_speeds = []
    pdat_mags = []

    t0 = frames[0]["timestamp"] if frames else 0

    for i, frame in enumerate(frames):
        ts = frame["timestamp"] - t0
        detections = process_radc_frame(
            frame, frame_index=i, fft_size=fft_size, max_speed_kmh=max_speed_kmh,
            cfar_threshold=cfar_threshold,
        )
        for d in detections:
            radc_times.append(ts)
            radc_velocities.append(d.velocity_kmh)
            radc_snrs.append(d.snr_db)

        for p in frame.get("pdat") or []:
            if p:
                pdat_times.append(ts)
                pdat_speeds.append(p.get("speed", 0))
                pdat_mags.append(p.get("magnitude", 0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if radc_times:
        sc = ax1.scatter(radc_times, radc_velocities, c=radc_snrs, s=10, cmap="hot", alpha=0.7)
        fig.colorbar(sc, ax=ax1, label="SNR (dB)")
    ax1.set_ylabel("RADC velocity (km/h)")
    ax1.set_title(f"{title} — RADC detections (our FFT + CFAR)")
    ax1.grid(True, alpha=0.25)

    if pdat_times:
        sc2 = ax2.scatter(pdat_times, pdat_speeds, c=pdat_mags, s=10, cmap="viridis", alpha=0.7)
        fig.colorbar(sc2, ax=ax2, label="Magnitude")
    ax2.set_ylabel("PDAT speed (km/h)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Module PDAT detections (built-in detector)")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_comparison_csv(
    frames: list[dict],
    output_path: Path,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    cfar_threshold: float = 8.0,
):
    """Write per-frame RADC vs PDAT comparison to CSV."""
    t0 = frames[0]["timestamp"] if frames else 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "time_s", "radc_count", "pdat_count",
            "radc_max_velocity_kmh", "pdat_max_speed_kmh",
            "radc_max_magnitude", "pdat_max_magnitude",
        ])
        writer.writeheader()

        for i, frame in enumerate(frames):
            detections = process_radc_frame(
                frame, frame_index=i, fft_size=fft_size, max_speed_kmh=max_speed_kmh,
                cfar_threshold=cfar_threshold,
            )
            comparison = compare_radc_vs_pdat(detections, frame.get("pdat") or [])
            comparison["frame"] = i
            comparison["time_s"] = round(frame["timestamp"] - t0, 3)
            writer.writerow(comparison)


def main():
    parser = argparse.ArgumentParser(description="Analyze K-LD7 RADC captures.")
    parser.add_argument("capture", type=Path, help="Path to RADC .pkl capture file")
    parser.add_argument("--shot-windows", action="store_true", help="Only plot active windows")
    parser.add_argument("--csv", action="store_true", help="Export per-frame comparison CSV")
    parser.add_argument("--fft-size", type=int, default=2048, help="FFT size (default: 2048)")
    parser.add_argument("--cfar-threshold", type=float, default=8.0, help="CFAR threshold factor")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots")
    args = parser.parse_args()

    data = load_capture(args.capture)
    meta = data["metadata"]
    frames = data["frames"]

    output_dir = args.output_dir or args.capture.parent / f"radc_analysis_{args.capture.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  K-LD7 RADC Analysis: {args.capture.name}")
    print("=" * 60)
    print(f"  Frames:     {len(frames)}")
    radc_frames = sum(1 for f in frames if f.get("radc"))
    print(f"  RADC:       {radc_frames} frames")
    print(f"  FFT size:   {args.fft_size}")
    print(f"  CFAR:       {args.cfar_threshold}x threshold")
    print(f"  Output:     {output_dir}")
    print()

    if args.shot_windows:
        windows = find_active_windows(frames)
        print(f"  Active windows: {len(windows)}")
        for i, (start, end) in enumerate(windows):
            window_frames = frames[start:end]
            t0 = window_frames[0]["timestamp"]
            duration = window_frames[-1]["timestamp"] - t0
            print(f"    Window {i+1}: frames {start}-{end} ({duration:.1f}s)")

            plot_spectrogram(
                window_frames,
                f"Window {i+1} (frames {start}-{end})",
                output_dir / f"spectrogram_window_{i+1:02d}.png",
                fft_size=args.fft_size,
            )
            plot_detection_timeline(
                window_frames,
                f"Window {i+1}",
                output_dir / f"detections_window_{i+1:02d}.png",
                fft_size=args.fft_size,
                cfar_threshold=args.cfar_threshold,
            )
    else:
        plot_spectrogram(
            frames,
            f"Full capture: {args.capture.name}",
            output_dir / "spectrogram_full.png",
            fft_size=args.fft_size,
        )
        plot_detection_timeline(
            frames,
            args.capture.name,
            output_dir / "detections_full.png",
            fft_size=args.fft_size,
            cfar_threshold=args.cfar_threshold,
        )

    if args.csv:
        csv_path = output_dir / "radc_vs_pdat.csv"
        write_comparison_csv(
            frames, csv_path,
            fft_size=args.fft_size,
            cfar_threshold=args.cfar_threshold,
        )
        print(f"  CSV: {csv_path}")

    # Summary stats
    total_radc_detections = 0
    total_pdat_detections = 0
    for i, frame in enumerate(frames):
        dets = process_radc_frame(
            frame, frame_index=i, fft_size=args.fft_size,
            cfar_threshold=args.cfar_threshold,
        )
        total_radc_detections += len(dets)
        total_pdat_detections += len(frame.get("pdat") or [])

    print()
    print(f"  RADC detections (our FFT+CFAR): {total_radc_detections}")
    print(f"  PDAT detections (module):       {total_pdat_detections}")
    print(f"  Ratio: {total_radc_detections / max(total_pdat_detections, 1):.1f}x")
    print()
    print(f"  Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
