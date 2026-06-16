#!/usr/bin/env python3
"""Render an annotated single-frame figure that contrasts the legacy
single-bin peak-pick angle estimate with the new magnitude²-weighted
multi-bin centroid (Zhang et al., Sensors 2016) used by
`extract_launch_angle`.

Designed as a self-contained illustration for explaining the change. It
loads a captured RADC pickle, picks the highest-SNR frame from the
specified shot, and draws:

  - top panel: magnitude spectrum, with the centroid window highlighted,
    a vertical line at the legacy single-bin peak, and a dot at the
    centroid bin location (for reference).
  - bottom panel: per-bin angle, with horizontal lines at:
      * the legacy single-bin angle (old)
      * the magnitude²-weighted centroid angle (new)
    and the same centroid window shaded.

Usage:
    uv run python scripts/analysis/illustrate_centroid_vs_peak.py \
        --pkl session_logs/kld7_radc_20260428_124851.pkl \
        --shot 3 \
        --out session_logs/centroid_illustration.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_radc():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from openflight.kld7 import radc  # noqa: E402
    return radc


def _pick_highest_snr_frame(group: dict, radc, band_tol_mph: float = 12.0) -> dict:
    """Find the highest-SNR frame inside the OPS-anchored ball band.

    Returns a dict with frame_index, peak_bin, peak_val, peak_snr, and
    ball_band_lo/hi (the OPS-anchored band used for the search).
    """
    ball_mph = group["ball_speed_mph"]
    if ball_mph is None:
        raise SystemExit("shot has no OPS243 ball speed; cannot anchor band")
    bands = radc.ball_bin_range_from_speed(
        ball_mph, band_tol_mph, 2048, 100.0,
    )

    best_meta: dict = {}
    best_snr = -np.inf
    for fi, frame in enumerate(group["frames"]):
        rb = frame.get("radc")
        if rb is None:
            continue
        ch = radc.parse_radc_payload(rb)
        iq1 = radc.to_complex_iq(ch["f1a_i"], ch["f1a_q"])
        spec = radc.compute_spectrum(iq1, fft_size=2048)
        # Find global peak across all sub-bands (handles wrap)
        peak_bin = -1
        peak_val = 0.0
        for sub_lo, sub_hi in bands:
            sub = spec[sub_lo:sub_hi]
            if sub.size == 0:
                continue
            sub_max = float(sub.max())
            if sub_max > peak_val:
                peak_val = sub_max
                peak_bin = sub_lo + int(np.argmax(sub))
        if peak_val <= 0 or peak_bin < 0:
            continue
        full_pos = spec[spec > 0]
        noise = float(np.median(full_pos)) if full_pos.size else 0.0
        snr = peak_val / max(noise, 1e-9)
        if snr > best_snr:
            best_snr = snr
            best_meta = {
                "frame_index": fi,
                "peak_bin": peak_bin,
                "peak_val": peak_val,
                "peak_snr": snr,
                "ball_bands": bands,
            }
    if not best_meta:
        raise SystemExit("no frames with RADC payload found")
    return best_meta


def _ops243_expected_bin(radc, ball_speed_mph: float, fft_size: int = 2048,
                         max_speed_kmh: float = 100.0) -> int:
    ball_kmh = ball_speed_mph * 1.609
    aliased = ball_kmh % (2.0 * max_speed_kmh)
    if aliased > max_speed_kmh:
        aliased -= 2.0 * max_speed_kmh
    return radc._velocity_to_bin(aliased, fft_size, max_speed_kmh)


def _load_groups(pkl_path: Path) -> list[dict]:
    """Load capture and group frames around each OPS243 shot — mirrors
    `group_frames_around_shots` in `diagnose_horizontal_angle.py`.
    """
    with pkl_path.open("rb") as f:
        capture = pickle.load(f)
    shots = capture.get("ops243_shots") or []
    frames = capture.get("frames") or []
    ms_before, ms_after = 1500.0, 700.0
    out = []
    for s in shots:
        t = s.get("timestamp")
        if t is None:
            continue
        sel = [
            fr for fr in frames
            if fr.get("radc") is not None
            and fr.get("timestamp") is not None
            and (t - ms_before / 1000.0) <= fr["timestamp"] <= (t + ms_after / 1000.0)
        ]
        out.append({
            "shot_timestamp": t,
            "ball_speed_mph": s.get("ball_speed_mph"),
            "frames": sel,
        })
    return out


def render(pkl_path: Path, shot_idx: int, out_path: Path,
           floor_frac: float = 0.5,
           zoom_pad_bins: int = 80) -> None:
    radc = _import_radc()
    groups = _load_groups(pkl_path)
    if shot_idx < 1 or shot_idx > len(groups):
        raise SystemExit(f"shot {shot_idx} not in capture (have {len(groups)})")
    group = groups[shot_idx - 1]

    bd = _pick_highest_snr_frame(group, radc)
    frame = group["frames"][bd["frame_index"]]
    rb = frame["radc"]
    ch = radc.parse_radc_payload(rb)
    iq1 = radc.to_complex_iq(ch["f1a_i"], ch["f1a_q"])
    iq2 = radc.to_complex_iq(ch["f2a_i"], ch["f2a_q"])
    spec = radc.compute_spectrum(iq1, fft_size=2048)
    fft1 = radc.compute_fft_complex(iq1, fft_size=2048)
    fft2 = radc.compute_fft_complex(iq2, fft_size=2048)
    angles = radc.per_bin_angle_deg(fft1, fft2)

    peak_bin = bd["peak_bin"]
    peak_val = float(spec[peak_bin])
    bands = bd["ball_bands"]
    # Sub-band that contains the peak (clip centroid neighborhood to it)
    sub_for_peak = next(
        (sub for sub in bands if sub[0] <= peak_bin < sub[1]),
        (0, 2048),
    )
    b_lo, b_hi = sub_for_peak

    # Centroid window (replicates extract_launch_angle math)
    lo_n = max(b_lo, peak_bin - radc.CENTROID_SEARCH_BINS)
    hi_n = min(b_hi, peak_bin + radc.CENTROID_SEARCH_BINS + 1)
    neigh = spec[lo_n:hi_n]
    threshold = peak_val * floor_frac
    neigh_mask = neigh >= threshold
    neigh_indices = np.flatnonzero(neigh_mask) + lo_n
    neigh_w = neigh[neigh_mask] ** 2
    centroid_angle = float(
        np.sum(angles[neigh_indices] * neigh_w) / float(neigh_w.sum())
    )
    legacy_angle = float(angles[peak_bin])

    # Plot zoomed onto the peak so the win/lose is visible
    z_lo = max(0, peak_bin - zoom_pad_bins)
    z_hi = min(len(spec), peak_bin + zoom_pad_bins)
    bins = np.arange(len(spec))
    mag_db = 20.0 * np.log10(np.maximum(spec, 1e-3))
    threshold_db = 20.0 * np.log10(max(threshold, 1e-3))

    ball_mph = group["ball_speed_mph"]
    expected_bin = _ops243_expected_bin(radc, ball_mph) if ball_mph else None

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )

    # --- Top: magnitude ---
    axes[0].plot(bins, mag_db, color="#1565C0", linewidth=1.0, zorder=2)
    axes[0].axvspan(neigh_indices.min(), neigh_indices.max() + 1,
                    color="#FFD54F", alpha=0.55, zorder=1,
                    label=f"NEW: centroid window "
                          f"(±{radc.CENTROID_SEARCH_BINS} bins, ≥{int(floor_frac*100)}% of peak)")
    axes[0].axhline(threshold_db, color="#F57C00", linestyle=":", linewidth=1.2,
                    alpha=0.8, label=f"floor = {int(floor_frac*100)}% of peak")
    axes[0].axvline(peak_bin, color="#C62828", linestyle="-", linewidth=2.0,
                    zorder=3,
                    label=f"OLD: single peak bin {peak_bin}")
    axes[0].scatter(neigh_indices, mag_db[neigh_indices],
                    s=18, color="#2E7D32", zorder=4,
                    label=f"bins included in centroid ({len(neigh_indices)})")
    if expected_bin is not None and z_lo <= expected_bin <= z_hi:
        axes[0].axvline(expected_bin, color="#455A64", linestyle="--",
                        linewidth=1.2, alpha=0.7,
                        label=f"OPS243 expected bin ({ball_mph:.1f} mph)")
    axes[0].set_ylabel("magnitude (dB)")
    axes[0].set_title(
        f"shot {shot_idx} — single-bin peak vs. magnitude²-weighted centroid  "
        f"(ball {ball_mph:.1f} mph)"
    )
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.92)
    axes[0].grid(alpha=0.3)

    # --- Bottom: per-bin angle ---
    axes[1].plot(bins, angles, color="#7B1FA2", linewidth=0.9, alpha=0.9)
    axes[1].axvspan(neigh_indices.min(), neigh_indices.max() + 1,
                    color="#FFD54F", alpha=0.55, zorder=1)
    axes[1].axvline(peak_bin, color="#C62828", linestyle="-", linewidth=2.0,
                    zorder=3)
    axes[1].scatter(neigh_indices, angles[neigh_indices],
                    s=22, color="#2E7D32", zorder=4)

    axes[1].axhline(
        legacy_angle, color="#C62828", linestyle="-", linewidth=2.0,
        alpha=0.85,
        label=f"OLD angle (single bin): {legacy_angle:+.2f}°",
    )
    axes[1].axhline(
        centroid_angle, color="#1B5E20", linestyle="-", linewidth=2.5,
        alpha=0.95,
        label=f"NEW angle (centroid):   {centroid_angle:+.2f}°",
    )
    axes[1].set_xlabel("FFT bin")
    axes[1].set_ylabel("per-bin angle (deg)")

    # Annotate diff
    delta = centroid_angle - legacy_angle
    axes[1].annotate(
        f"Δ = {delta:+.2f}°",
        xy=(peak_bin, (legacy_angle + centroid_angle) / 2),
        xytext=(peak_bin + 18, (legacy_angle + centroid_angle) / 2),
        fontsize=11, color="#1B5E20", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#1B5E20", linewidth=1.2),
        verticalalignment="center",
    )
    axes[1].legend(loc="lower left", fontsize=9, framealpha=0.92)
    axes[1].grid(alpha=0.3)

    # Center on the peak
    axes[1].set_xlim(z_lo, z_hi)
    # Y-range that comfortably includes both estimates and the local angle traces
    local_angles = angles[z_lo:z_hi]
    y_lo = min(float(local_angles.min()), legacy_angle, centroid_angle) - 5
    y_hi = max(float(local_angles.max()), legacy_angle, centroid_angle) + 5
    axes[1].set_ylim(y_lo, y_hi)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  legacy single-bin angle : {legacy_angle:+.2f}°  (bin {peak_bin})")
    print(f"  new centroid angle      : {centroid_angle:+.2f}°  "
          f"({len(neigh_indices)} bins, "
          f"{neigh_indices.min()}..{neigh_indices.max()})")
    print(f"  delta                   : {centroid_angle - legacy_angle:+.2f}°")


def render_spectrogram(pkl_path: Path, shot_idx: int, out_path: Path,
                       floor_frac: float = 0.5,
                       zoom_pad_bins: int = 80) -> None:
    """Annotated spectrogram (frame × bin heatmap) for the same shot.

    Shows the ball as a streak across consecutive frames, with:
      - per-frame OLD single-bin pick (red dots)
      - per-frame NEW centroid bin (green dots, weighted-mean bin)
      - centroid window for the highest-SNR frame (yellow band)
      - the OPS243-expected bin (dashed)
      - the highest-SNR frame highlighted on the y-axis
    """
    radc = _import_radc()
    groups = _load_groups(pkl_path)
    if shot_idx < 1 or shot_idx > len(groups):
        raise SystemExit(f"shot {shot_idx} not in capture (have {len(groups)})")
    group = groups[shot_idx - 1]
    frames = group["frames"]
    if not frames:
        raise SystemExit("no RADC frames in window for this shot")

    bd = _pick_highest_snr_frame(group, radc)
    bands = bd["ball_bands"]
    best_frame = bd["frame_index"]
    best_peak = bd["peak_bin"]

    fft_size = 2048
    n = len(frames)
    spec_grid = np.full((n, fft_size), 1e-3, dtype=np.float32)
    per_frame_peak: list[tuple[int, int]] = []      # (frame, peak_bin)
    per_frame_centroid: list[tuple[int, float]] = []  # (frame, centroid_bin)

    for fi, frame in enumerate(frames):
        rb = frame.get("radc")
        if rb is None:
            continue
        ch = radc.parse_radc_payload(rb)
        iq = radc.to_complex_iq(ch["f1a_i"], ch["f1a_q"])
        spec = radc.compute_spectrum(iq, fft_size=fft_size)
        spec_grid[fi] = spec
        # Find global peak across all sub-bands
        peak_bin = -1
        peak_val = 0.0
        for sub_lo, sub_hi in bands:
            sub = spec[sub_lo:sub_hi]
            if sub.size == 0:
                continue
            sub_max = float(sub.max())
            if sub_max > peak_val:
                peak_val = sub_max
                peak_bin = sub_lo + int(np.argmax(sub))
        if peak_val <= 0 or peak_bin < 0:
            continue
        per_frame_peak.append((fi, peak_bin))

        # Centroid bin (clip to the sub-band containing the peak)
        sub_for_peak = next(
            (sub for sub in bands if sub[0] <= peak_bin < sub[1]),
            (0, fft_size),
        )
        b_lo, b_hi = sub_for_peak
        lo_n = max(b_lo, peak_bin - radc.CENTROID_SEARCH_BINS)
        hi_n = min(b_hi, peak_bin + radc.CENTROID_SEARCH_BINS + 1)
        neigh = spec[lo_n:hi_n]
        mask = neigh >= peak_val * floor_frac
        if mask.any():
            idx = np.flatnonzero(mask) + lo_n
            w = neigh[mask] ** 2
            cbin = float(np.sum(idx * w) / float(w.sum()))
            per_frame_centroid.append((fi, cbin))

    # Window for the highest-SNR frame (used to shade the band)
    sub_for_best = next(
        (sub for sub in bands if sub[0] <= best_peak < sub[1]),
        (0, fft_size),
    )
    best_b_lo, best_b_hi = sub_for_best
    lo_b = max(best_b_lo, best_peak - radc.CENTROID_SEARCH_BINS)
    hi_b = min(best_b_hi, best_peak + radc.CENTROID_SEARCH_BINS)

    ball_mph = group["ball_speed_mph"]
    expected_bin = _ops243_expected_bin(radc, ball_mph) if ball_mph else None

    # Color scaling
    mag_db = 20.0 * np.log10(np.maximum(spec_grid, 1e-3))
    valid = mag_db[mag_db > 0]
    floor_db = float(np.percentile(valid, 50)) if valid.size else 0.0
    ceil_db = float(np.percentile(mag_db, 99.5))

    # Zoom window on x-axis
    z_lo = max(0, best_peak - zoom_pad_bins)
    z_hi = min(fft_size, best_peak + zoom_pad_bins)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    im = ax.imshow(
        mag_db, aspect="auto", origin="lower", cmap="magma",
        vmin=floor_db, vmax=ceil_db,
        extent=(0, fft_size, 0, n), interpolation="nearest",
    )

    # Centroid window for the highest-SNR frame
    ax.axvspan(lo_b, hi_b + 1, color="#FFD54F", alpha=0.22, lw=0,
               label=f"NEW: centroid window for highest-SNR frame "
                     f"(±{radc.CENTROID_SEARCH_BINS} bins)")

    # OPS243 expected bin
    if expected_bin is not None and z_lo <= expected_bin <= z_hi:
        ax.axvline(expected_bin, color="#B0BEC5", linestyle="--", linewidth=1.4,
                   alpha=0.85,
                   label=f"OPS243 expected bin ({ball_mph:.1f} mph)")

    # Highlight the highest-SNR frame row
    ax.axhspan(best_frame, best_frame + 1,
               facecolor="none", edgecolor="#FFFFFF",
               linewidth=1.2, alpha=0.7)
    ax.text(z_lo + 2, best_frame + 0.5, "highest SNR",
            color="white", fontsize=8, va="center")

    # Per-frame OLD picks (single-bin peak)
    if per_frame_peak:
        pf = np.array(per_frame_peak)
        ax.scatter(pf[:, 1], pf[:, 0] + 0.5, s=42, marker="o",
                   color="#C62828", edgecolor="white", linewidth=0.6,
                   label="OLD: per-frame single-bin peak", zorder=4)

    # Per-frame NEW centroid bin
    if per_frame_centroid:
        cf = np.array(per_frame_centroid)
        ax.scatter(cf[:, 1], cf[:, 0] + 0.5, s=42, marker="X",
                   color="#1B5E20", edgecolor="white", linewidth=0.6,
                   label="NEW: per-frame centroid bin", zorder=5)

    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(0, n)
    ax.set_xlabel("FFT bin")
    ax.set_ylabel("frame index in window")
    ax.set_title(
        f"shot {shot_idx} spectrogram — ball streak across frames  "
        f"(ball {ball_mph:.1f} mph)"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.colorbar(im, ax=ax, label="magnitude (dB)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  highest-SNR frame index : {best_frame}  (peak bin {best_peak})")
    print(f"  per-frame OLD picks     : {len(per_frame_peak)}")
    print(f"  per-frame NEW centroids : {len(per_frame_centroid)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True, type=Path,
                        help="Capture pickle (kld7_radc_*.pkl)")
    parser.add_argument("--shot", type=int, default=3,
                        help="1-based shot index in the capture (default 3)")
    parser.add_argument("--out", type=Path,
                        default=Path("session_logs") / "centroid_illustration.png",
                        help="Output PNG path")
    parser.add_argument("--mode", choices=["spectrum", "spectrogram", "both"],
                        default="spectrum",
                        help="Which illustration to render (default: spectrum)")
    parser.add_argument("--floor-frac", type=float, default=0.5,
                        help="Magnitude floor as fraction of peak (default 0.5)")
    parser.add_argument("--zoom-pad", type=int, default=80,
                        help="Bins to show on each side of the peak (default 80)")
    args = parser.parse_args()
    if args.mode in ("spectrum", "both"):
        spec_out = (
            args.out if args.mode == "spectrum"
            else args.out.with_name(args.out.stem + "_spectrum" + args.out.suffix)
        )
        render(args.pkl, args.shot, spec_out,
               floor_frac=args.floor_frac, zoom_pad_bins=args.zoom_pad)
    if args.mode in ("spectrogram", "both"):
        sgram_out = (
            args.out if args.mode == "spectrogram"
            else args.out.with_name(args.out.stem + "_spectrogram" + args.out.suffix)
        )
        render_spectrogram(args.pkl, args.shot, sgram_out,
                           floor_frac=args.floor_frac,
                           zoom_pad_bins=args.zoom_pad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
