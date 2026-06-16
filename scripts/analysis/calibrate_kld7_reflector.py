#!/usr/bin/env python3
"""Live K-LD7 angle reader for static-reflector calibration.

Designed for ground-truth calibration: place a metal reflector (corner
reflector, tin can, metal flag) at a known geometric angle from the radar,
and this script reports what the radar measures. Sweep the reflector
through several known angles to characterize the radar's actual transfer
function (slope + offset between measured and true angle).

Does NOT use OPS243 anchoring. Instead, finds the strongest peak in the
spectrum at each frame and reports its angle via phase interferometry —
which is what you want when there's a single dominant reflector in the
scene rather than a moving golf ball.

Important: the K-LD7 is a Doppler radar. A purely-static reflector won't
show up because DC bins get masked. **You need to give the reflector
slight motion** — wave it slowly by hand, hang it as a pendulum, or
mount it on something that vibrates at a few Hz. The angle measurement
is independent of the magnitude of the motion.

Usage
-----
Live monitoring (move reflector around, watch angle update):
    uv run --no-project --with numpy --with pyserial --with kld7 \\
        python scripts/analysis/calibrate_kld7_reflector.py

Timed capture at one known position (e.g. reflector at known 15° elevation):
    uv run --no-project --with numpy --with pyserial --with kld7 \\
        python scripts/analysis/calibrate_kld7_reflector.py \\
        --capture-seconds 30 \\
        --label "15deg_5m" \\
        --output ~/openflight_sessions/calib_15deg.json

For a full calibration session, run multiple timed captures at different
known angles, then fit measured-vs-true angles offline.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import sys
import time
from collections import deque
from pathlib import Path

# Make openflight imports work from any worktree layout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np

from openflight.kld7.radc import (  # noqa: E402
    parse_radc_payload,
    to_complex_iq,
    compute_spectrum,
    compute_fft_complex,
    per_bin_angle_deg,
    bin_to_velocity_kmh,
    DC_MASK_BINS,
)
from openflight.kld7.serial_io import connect_with_recovery  # noqa: E402


FFT_SIZE = 2048
MAX_SPEED_KMH = 100.0  # RSPI=3
SAMPLE_RATE_HZ = 8800  # ~256 samples in ~29 ms per K-LD7 datasheet


def setup_radar(port: str | None, range_setting: int, baud: int):
    """Open the K-LD7 serial port and configure it for static-reflector capture.

    Returns the kld7 radar object. Caller is responsible for radar.close()
    when done.
    """
    if not port:
        # Try common udev symlink first
        candidates = [
            "/dev/kld7_vertical",
            "/dev/kld7_horizontal",
        ]
        for c in candidates:
            if Path(c).exists():
                port = c
                break
    if not port:
        print("[calibrate] No port specified and no /dev/kld7_* symlink found.",
              file=sys.stderr)
        print("[calibrate] Pass --port /dev/ttyUSB0 (or similar).", file=sys.stderr)
        sys.exit(2)

    print(f"[calibrate] Opening {port} at {baud} baud…")
    radar = connect_with_recovery(port, baudrate=baud, log=print)

    # Configure for max angle resolution. Range setting affects FFT bin
    # spacing in velocity space; the angle measurement uses phase
    # interferometry across the two Rx channels so it's largely
    # independent of range / speed range choices.
    p = radar.params
    range_codes = {5: 0, 10: 1, 30: 2, 100: 3}
    p.RRAI = range_codes.get(range_setting, 0)
    p.RSPI = 3      # max speed (100 km/h)
    p.RBFR = 0      # base frequency: 24.05 GHz
    p.DEDI = 2
    p.THOF = 10
    p.TRFT = 1
    p.MIAN = -90
    p.MAAN = 90
    p.MIRA = 0
    p.MARA = 100
    p.MISP = 0
    p.MASP = 100
    p.VISU = 0

    print(f"[calibrate] Configured: range={range_setting}m, speed=100km/h, RBFR=0")
    return radar


def extract_strongest_peak(payload: bytes) -> dict | None:
    """For one RADC frame, find the strongest peak in the F1a spectrum
    (excluding DC) and report angle/velocity/SNR for it."""
    if len(payload) != 3072:
        return None
    try:
        channels = parse_radc_payload(payload)
    except ValueError:
        return None
    f1a_iq = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
    f2a_iq = to_complex_iq(channels["f2a_i"], channels["f2a_q"])
    spec = compute_spectrum(f1a_iq, fft_size=FFT_SIZE)

    # Find strongest peak across the whole spectrum (excluding DC)
    # DC_MASK_BINS already zeros out bins near DC in compute_spectrum
    peak_bin = int(np.argmax(spec))
    peak_val = float(spec[peak_bin])
    if peak_val <= 0:
        return None

    positive = spec[spec > 0]
    noise_floor = float(np.median(positive)) if positive.size else 0.0
    snr = peak_val / noise_floor if noise_floor > 0 else 0.0
    snr_db = 10.0 * math.log10(snr) if snr > 0 else 0.0

    f1a_fft = compute_fft_complex(f1a_iq, fft_size=FFT_SIZE)
    f2a_fft = compute_fft_complex(f2a_iq, fft_size=FFT_SIZE)
    angles = per_bin_angle_deg(f1a_fft, f2a_fft)
    angle_deg = float(angles[peak_bin])
    velocity_kmh = bin_to_velocity_kmh(peak_bin, FFT_SIZE, MAX_SPEED_KMH)

    return {
        "peak_bin": peak_bin,
        "angle_deg": angle_deg,
        "velocity_kmh": velocity_kmh,
        "snr_linear": snr,
        "snr_db": snr_db,
        "peak_magnitude": peak_val,
        "noise_floor": noise_floor,
    }


def fmt_row(reading: dict) -> str:
    return (
        f"peak_bin={reading['peak_bin']:4d}  "
        f"v={reading['velocity_kmh']:+6.1f} km/h  "
        f"angle={reading['angle_deg']:+6.2f}°  "
        f"SNR={reading['snr_linear']:6.1f}× ({reading['snr_db']:5.1f} dB)"
    )


def summarize(angles: list[float], snrs: list[float], label: str = "") -> dict:
    if not angles:
        return {"n": 0}
    n = len(angles)
    mean = statistics.mean(angles)
    median = statistics.median(angles)
    std = statistics.pstdev(angles) if n >= 2 else 0.0
    out = {
        "n": n,
        "mean_angle_deg": mean,
        "median_angle_deg": median,
        "stdev_angle_deg": std,
        "min_angle_deg": min(angles),
        "max_angle_deg": max(angles),
        "mean_snr_linear": statistics.mean(snrs),
        "median_snr_linear": statistics.median(snrs),
    }
    if label:
        out["label"] = label
    return out


def print_summary(s: dict, label: str = "") -> None:
    if not s.get("n"):
        print(f"  {label}: no samples")
        return
    print(f"  {label}n={s['n']}  "
          f"mean={s['mean_angle_deg']:+.2f}°  "
          f"median={s['median_angle_deg']:+.2f}°  "
          f"σ={s['stdev_angle_deg']:.2f}°  "
          f"range=[{s['min_angle_deg']:+.2f}, {s['max_angle_deg']:+.2f}]  "
          f"SNR_med={s['median_snr_linear']:.1f}×")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", default=None,
                        help="Serial port (default: auto-detect /dev/kld7_vertical)")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--range", type=int, default=5, choices=[5, 10, 30, 100],
                        dest="range_m", help="K-LD7 max range setting in meters")
    parser.add_argument("--capture-seconds", type=float, default=None,
                        help="Capture for N seconds, summarize, exit. "
                             "If omitted: run live monitor until Ctrl-C.")
    parser.add_argument("--label", default="",
                        help="Label written into the saved JSON (e.g. '15deg_5m')")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save per-frame readings + summary to this JSON file")
    parser.add_argument("--min-snr-linear", type=float, default=3.0,
                        help="Skip frames where the dominant peak is too weak "
                             "(default 3.0× = ~5 dB)")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print every Nth frame in live mode (default 10)")
    parser.add_argument("--start-delay", type=float, default=0.0,
                        help="Stream live (showing readings + countdown) for N "
                             "seconds before the capture window begins, so you "
                             "can get the reflector into position and stable.")
    args = parser.parse_args()

    radar = setup_radar(args.port, args.range_m, args.baud)
    try:
        from kld7 import FrameCode  # type: ignore
    except ImportError:
        print("[calibrate] kld7 package not installed. Reinstall the project: "
              "uv pip install -e '.[ui]'",
              file=sys.stderr)
        return 2

    print("\n[calibrate] Streaming RADC. ", end="")
    if args.capture_seconds:
        print(f"Capturing {args.capture_seconds:.1f}s, then summarizing.")
    else:
        print("Live monitor — Ctrl-C to exit.")
    print()

    readings: list[dict] = []
    rolling_angles: deque[float] = deque(maxlen=30)
    stream_start = time.time()
    capture_start = stream_start + max(args.start_delay, 0.0)
    in_warmup = args.start_delay > 0.0
    last_countdown = 0.0
    frame_count = 0

    if in_warmup:
        print(f"[calibrate] Warmup: get the reflector into position. "
              f"Capture window begins in {args.start_delay:.0f}s...\n")

    def handle_signal(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)

    try:
        for code, payload in radar.stream_frames(FrameCode.RADC, max_count=-1):
            now = time.time()

            # Warmup phase: show live readings + countdown, but don't record.
            if now < capture_start:
                if code != "RADC" or not isinstance(payload, bytes):
                    continue
                r = extract_strongest_peak(payload)
                if r is None or r["snr_linear"] < args.min_snr_linear:
                    continue
                rolling_angles.append(r["angle_deg"])
                if now - last_countdown >= 1.0:
                    last_countdown = now
                    remaining = capture_start - now
                    roll_mean = statistics.mean(rolling_angles) if rolling_angles else 0.0
                    roll_std = (statistics.pstdev(rolling_angles)
                                if len(rolling_angles) >= 2 else 0.0)
                    print(f"  [warmup {remaining:4.1f}s] angle={r['angle_deg']:+6.2f}°  "
                          f"SNR={r['snr_linear']:7.1f}×   "
                          f"rolling μ={roll_mean:+.2f}° σ={roll_std:.2f}°")
                continue

            # First frame after warmup: announce and reset rolling stats.
            if in_warmup:
                in_warmup = False
                rolling_angles.clear()
                if args.capture_seconds:
                    print(f"\n[calibrate] >>> Capturing now for "
                          f"{args.capture_seconds:.0f}s <<<\n")
                else:
                    print("\n[calibrate] >>> Capture started — Ctrl-C to stop <<<\n")

            # Capture phase.
            if args.capture_seconds and (now - capture_start) >= args.capture_seconds:
                break
            if code != "RADC" or not isinstance(payload, bytes):
                continue
            r = extract_strongest_peak(payload)
            if r is None:
                continue
            if r["snr_linear"] < args.min_snr_linear:
                continue
            r["t_s"] = round(now - capture_start, 4)
            readings.append(r)
            rolling_angles.append(r["angle_deg"])
            frame_count += 1

            if args.capture_seconds is None and frame_count % args.print_every == 0:
                # Live monitor: print every Nth frame with rolling stats
                roll_mean = statistics.mean(rolling_angles)
                roll_std = statistics.pstdev(rolling_angles) if len(rolling_angles) >= 2 else 0.0
                print(f"  t={r['t_s']:6.2f}s  {fmt_row(r)}  "
                      f"  rolling[{len(rolling_angles)}]: μ={roll_mean:+.2f}°  σ={roll_std:.2f}°")

    except KeyboardInterrupt:
        print("\n[calibrate] Interrupted.")
    finally:
        try:
            radar.close()
        except Exception:
            pass

    elapsed = time.time() - capture_start
    angles = [r["angle_deg"] for r in readings]
    snrs = [r["snr_linear"] for r in readings]

    print(f"\n[calibrate] Capture done: {len(readings)} frames in {elapsed:.1f}s")
    summary = summarize(angles, snrs, label=args.label)
    print_summary(summary, label=f"{args.label or 'all'}: " if args.label else "")

    # Also separate "high SNR" subset (likely the reflector specifically)
    high_snr_pairs = [(r["angle_deg"], r["snr_linear"])
                      for r in readings if r["snr_linear"] >= 10.0]
    if high_snr_pairs:
        hi_angles, hi_snrs = zip(*high_snr_pairs)
        hi_summary = summarize(list(hi_angles), list(hi_snrs))
        print_summary(hi_summary, label="SNR >= 10x: ")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "label": args.label,
            "range_m": args.range_m,
            "elapsed_s": elapsed,
            "summary": summary,
            "high_snr_summary": (summarize(list(hi_angles), list(hi_snrs))
                                 if high_snr_pairs else None),
            "readings": readings,
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"[calibrate] Wrote {len(readings)} readings to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
