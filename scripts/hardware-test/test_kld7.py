#!/usr/bin/env python3
"""
K-LD7 radar module test script for angle/distance data gathering.

Connects to the K-LD7 EVAL board via USB serial, configures for golf
use (short range, max speed, both directions), streams raw target
detection (PDAT) and optionally FFT data, and saves everything to a
.pkl file.

The K-LD7 measures horizontal angle, distance, speed, and magnitude
per target. Speed maxes out at 100 km/h (62 mph) — below golf ball
speeds — so speed data will alias. Angle and distance data should
still be valid.

WARNING: The datasheet explicitly states that targets moving faster
than the configured max speed produce WRONG speed values (aliasing),
not just saturated ones. Speed data from K-LD7 is unreliable for golf.
OPS243 handles speed — K-LD7 is for angle and distance only.

At the 100 km/h speed setting, frame duration is ~29ms (~34 fps).
A golf ball transits the 5m detection zone in ~30ms, so expect only
1-2 frames per shot. PDAT (raw detections) is more useful than TDAT
(tracked target) since the tracking filter may not lock on in time.

Prerequisites:
    uv pip install -e '.[ui]'   # kld7 ships as a base dependency

Usage:
    # Basic capture (auto-detect port, Ctrl+C to stop)
    python scripts/test_kld7.py

    # Specify port and range
    python scripts/test_kld7.py --port /dev/ttyUSB0 --range 10

    # Skip FFT data (smaller output)
    python scripts/test_kld7.py --no-fft

    # Limit to 500 frames
    python scripts/test_kld7.py -n 500

To analyze afterwards:
    python -c "import pickle; d=pickle.load(open('file.pkl','rb')); print(d['metadata'])"

Datasheets:
    docs/K-LD7_Datasheet.pdf (module)
    docs/K-LD7-EVAL_Datasheet.pdf (eval board)
"""

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from kld7 import KLD7, FrameCode, KLD7Exception
except ImportError:
    print("Error: kld7 package not installed. Reinstall the project: uv pip install -e '.[ui]'")
    sys.exit(1)

from serial.tools.list_ports import comports


# K-LD7 parameter presets for golf
RANGE_SETTINGS = {5: 0, 10: 1, 30: 2, 100: 3}
SPEED_SETTINGS = {12: 0, 25: 1, 50: 2, 100: 3}


def find_kld7_port():
    """Auto-detect K-LD7 EVAL board USB serial port."""
    for port in comports():
        desc = (port.description or "").lower()
        mfg = (port.manufacturer or "").lower()
        # EVAL board uses FTDI or CP210x USB-to-serial
        if any(kw in desc for kw in ["ftdi", "cp210", "usb-serial", "uart"]):
            return port.device
        if any(kw in mfg for kw in ["ftdi", "silicon labs"]):
            return port.device
    return None


def configure_for_golf(radar, range_m=5, speed_kmh=100):
    """Configure K-LD7 parameters for golf ball detection."""
    params = radar.params

    # Range and speed
    params.RRAI = RANGE_SETTINGS.get(range_m, 0)
    params.RSPI = SPEED_SETTINGS.get(speed_kmh, 3)

    # Detection direction: both approaching and receding
    params.DEDI = 2

    # Low threshold for max sensitivity
    params.THOF = 10

    # Fast detection tracking filter (transient targets)
    params.TRFT = 1

    # Full angle range
    params.MIAN = -90
    params.MAAN = 90

    # Full distance range
    params.MIRA = 0
    params.MARA = 100

    # Full speed range
    params.MISP = 0
    params.MASP = 100

    # No vibration suppression (we want raw signal)
    params.VISU = 0

    # Frame duration at 100 km/h: ~29ms (~34 fps)
    frame_ms = {12: 115, 25: 58, 50: 38, 100: 29}.get(speed_kmh, 29)

    print(f"  Range: {range_m}m (RRAI={params.RRAI})")
    print(f"  Max speed: {speed_kmh} km/h (RSPI={params.RSPI})")
    print(f"  Frame rate: ~{1000 // frame_ms} fps ({frame_ms}ms/frame)")
    print(f"  Direction: both (DEDI={params.DEDI})")
    print(f"  Threshold: {params.THOF} dB")
    print(f"  Tracking: fast detection (TRFT={params.TRFT})")
    print()
    print("  NOTE: Golf balls exceed max speed — speed values will alias.")
    print("  Angle and distance data should still be valid.")


def read_all_params(radar):
    """Read all K-LD7 parameters into a dict for saving."""
    param_names = [
        "RBFR", "RSPI", "RRAI", "THOF", "TRFT", "VISU",
        "MIRA", "MARA", "MIAN", "MAAN", "MISP", "MASP",
        "DEDI", "RATH", "ANTH", "SPTH", "DIG1", "DIG2",
        "DIG3", "HOLD", "MIDE", "MIDS",
    ]
    params = {}
    for name in param_names:
        try:
            params[name] = getattr(radar.params, name)
        except Exception:
            params[name] = None
    return params


def target_to_dict(target):
    """Convert a Target namedtuple to a serializable dict."""
    if target is None:
        return None
    return {
        "distance": target.distance,
        "speed": target.speed,
        "angle": target.angle,
        "magnitude": target.magnitude,
    }


def main():
    parser = argparse.ArgumentParser(
        description="K-LD7 radar test script for angle/distance data gathering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python scripts/test_kld7.py
    python scripts/test_kld7.py --port /dev/ttyUSB0 --range 10
    python scripts/test_kld7.py --no-fft -n 500
        """,
    )
    parser.add_argument(
        "--port", help="Serial port (auto-detect if not specified)"
    )
    parser.add_argument(
        "--range", "-r", type=int, default=5, choices=[5, 10, 30, 100],
        help="Detection range in meters (default: 5)"
    )
    parser.add_argument(
        "--speed", "-s", type=int, default=100, choices=[12, 25, 50, 100],
        help="Max speed in km/h (default: 100)"
    )
    parser.add_argument(
        "--baud", "-b", type=int, default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--no-fft", action="store_true",
        help="Skip raw FFT frames (smaller output, faster)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .pkl file (default: auto in ~/openflight_sessions/)"
    )
    parser.add_argument(
        "-n", "--max-frames", type=int, default=0,
        help="Stop after N frames (default: 0 = unlimited)"
    )
    parser.add_argument(
        "--club", type=str, default=None,
        help="Club used for this session (e.g. 'driver', '7iron', 'PW')"
    )
    parser.add_argument(
        "--shots", type=int, default=None,
        help="Expected number of shots (saved to metadata for ground truth)"
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Session notes (e.g. 'hitting into net at 3m')"
    )
    args = parser.parse_args()

    # Find port
    port = args.port
    if not port:
        port = find_kld7_port()
        if not port:
            print("Error: No K-LD7 EVAL board detected. Specify --port manually.")
            print("Available ports:")
            for p in comports():
                print(f"  {p.device}: {p.description} [{p.manufacturer}]")
            sys.exit(1)

    # Output file
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path.home() / "openflight_sessions"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"kld7_capture_{timestamp}.pkl"

    print("=" * 60)
    print("  K-LD7 Radar Test (Angle + Distance Data Gathering)")
    print("=" * 60)
    print()
    print(f"  Port:    {port}")
    print(f"  Baud:    {args.baud}")
    print(f"  Output:  {output_path}")
    print(f"  FFT:     {'disabled' if args.no_fft else 'enabled'}")
    if args.max_frames > 0:
        print(f"  Max:     {args.max_frames} frames")
    print()

    # Connect
    print("Connecting to K-LD7...")
    try:
        radar = KLD7(port, baudrate=args.baud)
    except (KLD7Exception, Exception) as e:
        print(f"Error connecting: {e}")
        sys.exit(1)
    print(f"  Connected: {radar}")
    print()

    # Configure
    print("Configuring for golf...")
    configure_for_golf(radar, range_m=args.range, speed_kmh=args.speed)
    all_params = read_all_params(radar)
    print()

    # Determine frame codes
    if args.no_fft:
        frame_codes = FrameCode.TDAT | FrameCode.PDAT
    else:
        frame_codes = FrameCode.TDAT | FrameCode.PDAT | FrameCode.RFFT

    # Metadata
    metadata = {
        "module": "K-LD7",
        "port": port,
        "baud_rate": args.baud,
        "capture_start": datetime.now().isoformat(),
        "range_m": args.range,
        "speed_kmh": args.speed,
        "fft_enabled": not args.no_fft,
        "params": all_params,
        "club": args.club,
        "expected_shots": args.shots,
        "notes": args.notes,
    }

    frames = []
    frame_count = 0
    detection_count = 0

    print("-" * 60)
    print("Streaming target data (Ctrl+C to stop)")
    print("-" * 60)
    print()
    print(f"  {'#':>5s}  {'dist(m)':>8s}  {'speed(km/h)':>12s}  {'angle':>7s}  {'mag':>5s}  {'targets':>7s}")
    print(f"  {'-----':>5s}  {'--------':>8s}  {'------------':>12s}  {'-------':>7s}  {'-----':>5s}  {'-------':>7s}")

    # Track which frame codes we expect per cycle to detect frame boundaries.
    # A new TDAT after we've already seen one means new frame cycle.
    try:
        current_frame = {"timestamp": time.time()}
        seen_in_frame = set()

        for code, payload in radar.stream_frames(
            frame_codes,
            max_count=args.max_frames if args.max_frames > 0 else -1,
        ):
            # Detect frame boundary: if we see a code we already saw in this
            # cycle, the previous frame is complete.
            if code in seen_in_frame:
                frames.append(current_frame)
                if "pdat" not in current_frame:
                    print()
                current_frame = {"timestamp": time.time()}
                seen_in_frame = set()

            seen_in_frame.add(code)

            if code == "TDAT":
                current_frame["tdat"] = target_to_dict(payload)

                frame_count += 1

                # Print live data
                if payload is not None:
                    detection_count += 1
                    print(
                        f"  {frame_count:5d}  {payload.distance:8.2f}  "
                        f"{payload.speed:12.1f}  {payload.angle:6.1f}\u00b0  "
                        f"{payload.magnitude:5.0f}",
                        end="",
                    )
                else:
                    print(f"  {frame_count:5d}  {'---':>8s}  {'---':>12s}  {'---':>7s}  {'---':>5s}", end="")

            elif code == "PDAT":
                current_frame["pdat"] = [target_to_dict(t) for t in payload] if payload else []
                n_targets = len(payload) if payload else 0
                print(f"  {n_targets:7d}")

            elif code == "RFFT":
                current_frame["rfft"] = payload

        # Save final in-progress frame
        if seen_in_frame:
            frames.append(current_frame)

    except KeyboardInterrupt:
        print()

    except KLD7Exception as e:
        print(f"\nK-LD7 error: {e}")

    finally:
        print()
        print("=" * 60)
        print("  CAPTURE SUMMARY")
        print("=" * 60)
        print(f"  Total frames:      {frame_count}")
        print(f"  TDAT detections:   {detection_count}")
        if frame_count > 0:
            print(f"  TDAT det. rate:    {detection_count / frame_count * 100:.0f}%")

            # TDAT stats (tracked target — may miss fast golf balls)
            angles = [f["tdat"]["angle"] for f in frames if f.get("tdat")]
            distances = [f["tdat"]["distance"] for f in frames if f.get("tdat")]

            if angles:
                print(f"  TDAT angle range:  {min(angles):.1f}\u00b0 to {max(angles):.1f}\u00b0")
                print(f"  TDAT dist range:   {min(distances):.2f}m to {max(distances):.2f}m")

            # PDAT stats (raw detections — more reliable for transient targets)
            pdat_targets = []
            for f in frames:
                if f.get("pdat"):
                    pdat_targets.extend(f["pdat"])
            pdat_with_data = [t for t in pdat_targets if t is not None]

            if pdat_with_data:
                pdat_angles = [t["angle"] for t in pdat_with_data]
                pdat_dists = [t["distance"] for t in pdat_with_data]
                pdat_frames = sum(1 for f in frames if f.get("pdat") and len(f["pdat"]) > 0)
                print(f"  PDAT detections:   {len(pdat_with_data)} targets in {pdat_frames} frames")
                print(f"  PDAT angle range:  {min(pdat_angles):.1f}\u00b0 to {max(pdat_angles):.1f}\u00b0")
                print(f"  PDAT dist range:   {min(pdat_dists):.2f}m to {max(pdat_dists):.2f}m")

        # Save
        if frames:
            metadata["capture_end"] = datetime.now().isoformat()
            metadata["total_frames"] = len(frames)
            metadata["detection_count"] = detection_count

            output_data = {
                "metadata": metadata,
                "frames": frames,
            }

            with open(output_path, "wb") as f:
                pickle.dump(output_data, f)

            print(f"  Output file:       {output_path}")
            print(f"  File size:         {output_path.stat().st_size / 1024:.1f} KB")
        else:
            print("\n  No frames captured.")

        print("=" * 60)

        # Close cleanly. The kld7 library's __del__ can error if the port
        # is already gone, so we close explicitly and suppress the destructor.
        try:
            radar.close()
        except Exception:
            pass
        # Prevent __del__ from erroring after we already closed
        try:
            radar._port = None
        except Exception:
            pass


if __name__ == "__main__":
    main()
