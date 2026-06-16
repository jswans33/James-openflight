#!/usr/bin/env python3
"""
Analyze K-LD7 radar capture files (.pkl) from test_kld7.py.

Prints summary stats, detection timeline, and optionally generates plots
of angle, distance, and magnitude over time.

Usage:
    # Summary only
    python scripts/analyze_kld7.py ~/openflight_sessions/kld7_capture_*.pkl

    # With plots (saves to same directory as .pkl)
    python scripts/analyze_kld7.py capture.pkl --plot

    # Show PDAT multi-target details
    python scripts/analyze_kld7.py capture.pkl --pdat

    # Filter to frames with detections only
    python scripts/analyze_kld7.py capture.pkl --detections-only

    # Print probable club-to-ball pairs
    python scripts/analyze_kld7.py capture.pkl --pair-shots
"""

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openflight.kld7.tracker import KLD7Tracker
from openflight.kld7.types import KLD7Frame


def load_capture(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_metadata(meta):
    print("=" * 60)
    print("  CAPTURE METADATA")
    print("=" * 60)
    for key, value in meta.items():
        if key == "params":
            print(f"  {key}:")
            for pk, pv in value.items():
                print(f"    {pk}: {pv}")
        else:
            print(f"  {key}: {value}")
    print()


def print_tdat_summary(frames):
    tdat_frames = [f for f in frames if f.get("tdat")]
    if not tdat_frames:
        print("  No TDAT detections.")
        return

    angles = [f["tdat"]["angle"] for f in tdat_frames]
    distances = [f["tdat"]["distance"] for f in tdat_frames]
    speeds = [f["tdat"]["speed"] for f in tdat_frames]
    magnitudes = [f["tdat"]["magnitude"] for f in tdat_frames]

    print(f"  TDAT detections:   {len(tdat_frames)} / {len(frames)} frames")
    print(f"  Angle:             {min(angles):.1f}\u00b0 to {max(angles):.1f}\u00b0  (mean {sum(angles)/len(angles):.1f}\u00b0)")
    print(f"  Distance:          {min(distances):.2f}m to {max(distances):.2f}m  (mean {sum(distances)/len(distances):.2f}m)")
    print(f"  Speed:             {min(speeds):.1f} to {max(speeds):.1f} km/h  (WARNING: aliased at golf speeds)")
    print(f"  Magnitude:         {min(magnitudes):.0f} to {max(magnitudes):.0f}  (mean {sum(magnitudes)/len(magnitudes):.0f})")


def print_pdat_summary(frames):
    all_targets = []
    frames_with_targets = 0
    for f in frames:
        pdat = f.get("pdat", [])
        targets = [t for t in pdat if t is not None]
        if targets:
            frames_with_targets += 1
            all_targets.extend(targets)

    if not all_targets:
        print("  No PDAT detections.")
        return

    angles = [t["angle"] for t in all_targets]
    distances = [t["distance"] for t in all_targets]
    magnitudes = [t["magnitude"] for t in all_targets]

    print(f"  PDAT detections:   {len(all_targets)} targets across {frames_with_targets} frames")
    print(f"  Angle:             {min(angles):.1f}\u00b0 to {max(angles):.1f}\u00b0  (mean {sum(angles)/len(angles):.1f}\u00b0)")
    print(f"  Distance:          {min(distances):.2f}m to {max(distances):.2f}m  (mean {sum(distances)/len(distances):.2f}m)")
    print(f"  Magnitude:         {min(magnitudes):.0f} to {max(magnitudes):.0f}  (mean {sum(magnitudes)/len(magnitudes):.0f})")


def print_timeline(frames, detections_only=False, show_pdat=False):
    print()
    print("-" * 80)
    if show_pdat:
        print(f"  {'#':>5s}  {'time(s)':>8s}  {'dist(m)':>8s}  {'speed':>8s}  {'angle':>7s}  {'mag':>5s}  {'src':>4s}")
        print(f"  {'-----':>5s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'-------':>7s}  {'-----':>5s}  {'----':>4s}")
    else:
        print(f"  {'#':>5s}  {'time(s)':>8s}  {'dist(m)':>8s}  {'speed':>8s}  {'angle':>7s}  {'mag':>5s}")
        print(f"  {'-----':>5s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'-------':>7s}  {'-----':>5s}")
    print("-" * 80)

    if not frames:
        print("  (no frames)")
        return

    t0 = frames[0].get("timestamp", 0)

    for i, f in enumerate(frames):
        t = f.get("timestamp", 0) - t0
        tdat = f.get("tdat")
        pdat = f.get("pdat", [])

        has_detection = tdat is not None or any(t is not None for t in pdat)
        if detections_only and not has_detection:
            continue

        # Print TDAT line
        if tdat:
            line = (
                f"  {i:5d}  {t:8.3f}  {tdat['distance']:8.2f}  "
                f"{tdat['speed']:8.1f}  {tdat['angle']:6.1f}\u00b0  "
                f"{tdat['magnitude']:5.0f}"
            )
            if show_pdat:
                line += "  TDAT"
            print(line)
        elif not show_pdat:
            print(f"  {i:5d}  {t:8.3f}  {'---':>8s}  {'---':>8s}  {'---':>7s}  {'---':>5s}")

        # Print PDAT targets
        if show_pdat:
            targets = [pt for pt in pdat if pt is not None]
            for j, pt in enumerate(targets):
                print(
                    f"  {'':>5s}  {'':>8s}  {pt['distance']:8.2f}  "
                    f"{pt['speed']:8.1f}  {pt['angle']:6.1f}\u00b0  "
                    f"{pt['magnitude']:5.0f}  P{j}"
                )


def find_events(frames, min_gap_s=0.5):
    """Group consecutive detections into events (possible ball passes)."""
    events = []
    current_event = []

    for i, f in enumerate(frames):
        has_tdat = f.get("tdat") is not None
        has_pdat = any(t is not None for t in f.get("pdat", []))

        if has_tdat or has_pdat:
            if current_event:
                prev_t = frames[current_event[-1]].get("timestamp", 0)
                curr_t = f.get("timestamp", 0)
                if curr_t - prev_t > min_gap_s:
                    events.append(current_event)
                    current_event = []
            current_event.append(i)
        else:
            if current_event:
                events.append(current_event)
                current_event = []

    if current_event:
        events.append(current_event)

    return events


def print_events(frames, events):
    if not events:
        print("  No detection events found.")
        return

    t0 = frames[0].get("timestamp", 0)

    print()
    print("=" * 60)
    print(f"  DETECTION EVENTS ({len(events)} found)")
    print("=" * 60)
    print(f"  {'Event':>5s}  {'time(s)':>8s}  {'frames':>6s}  {'dur(ms)':>8s}  {'angle':>7s}  {'dist(m)':>8s}  {'mag':>5s}")
    print(f"  {'-----':>5s}  {'--------':>8s}  {'------':>6s}  {'--------':>8s}  {'-------':>7s}  {'--------':>8s}  {'-----':>5s}")

    for ei, event_idxs in enumerate(events):
        event_frames = [frames[i] for i in event_idxs]
        t_start = event_frames[0].get("timestamp", 0) - t0
        t_end = event_frames[-1].get("timestamp", 0) - t0
        dur_ms = (t_end - t_start) * 1000

        # Collect all detections (TDAT + PDAT)
        angles = []
        distances = []
        magnitudes = []
        for f in event_frames:
            if f.get("tdat"):
                angles.append(f["tdat"]["angle"])
                distances.append(f["tdat"]["distance"])
                magnitudes.append(f["tdat"]["magnitude"])
            for pt in f.get("pdat", []):
                if pt is not None:
                    angles.append(pt["angle"])
                    distances.append(pt["distance"])
                    magnitudes.append(pt["magnitude"])

        avg_angle = sum(angles) / len(angles) if angles else 0
        avg_dist = sum(distances) / len(distances) if distances else 0
        max_mag = max(magnitudes) if magnitudes else 0

        print(
            f"  {ei+1:5d}  {t_start:8.3f}  {len(event_idxs):6d}  "
            f"{dur_ms:8.1f}  {avg_angle:6.1f}\u00b0  {avg_dist:8.2f}  {max_mag:5.0f}"
        )


def build_tracker(frames, orientation="vertical"):
    """Load capture frames into an in-memory tracker for offline analysis."""
    tracker = KLD7Tracker.__new__(KLD7Tracker)
    tracker.orientation = orientation
    tracker.buffer_seconds = 2.0
    tracker.max_buffer_frames = max(len(frames), 70)
    tracker._init_ring_buffer()
    for frame in frames:
        tracker._add_frame(KLD7Frame(
            timestamp=frame["timestamp"],
            tdat=frame.get("tdat"),
            pdat=frame.get("pdat", []),
        ))
    return tracker


def print_probable_shots(frames, orientation="vertical"):
    """Print likely club-to-ball pairings for a long capture."""
    tracker = build_tracker(frames, orientation=orientation)
    probable_shots = tracker.find_probable_shots()

    print()
    print("=" * 60)
    print(f"  PROBABLE SHOTS ({len(probable_shots)} found)")
    print("=" * 60)
    if not probable_shots:
        print("  No club-to-ball pairs found.")
        return

    t0 = frames[0].get("timestamp", 0)
    print(
        f"  {'#':>5s}  {'club(s)':>8s}  {'ball(s)':>8s}  {'dt(ms)':>8s}  "
        f"{'club':>7s}  {'ball':>7s}  {'dist(m)':>8s}  {'mag':>5s}  {'conf':>5s}"
    )
    print(
        f"  {'-----':>5s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  "
        f"{'-------':>7s}  {'-------':>7s}  {'--------':>8s}  {'-----':>5s}  {'-----':>5s}"
    )
    for idx, shot in enumerate(probable_shots, start=1):
        print(
            f"  {idx:5d}  {shot['club_time'] - t0:8.3f}  {shot['ball_time'] - t0:8.3f}  "
            f"{shot['dt_ms']:8.1f}  {shot['club_angle_deg']:6.1f}\u00b0  "
            f"{shot['ball_angle_deg']:6.1f}\u00b0  {shot['ball_distance_m']:8.2f}  "
            f"{shot['ball_magnitude']:5.0f}  {shot['ball_confidence']:5.2f}"
        )


def plot_capture(frames, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots. Install with: uv pip install matplotlib")
        return

    t0 = frames[0].get("timestamp", 0)

    # Collect TDAT data
    tdat_t, tdat_angle, tdat_dist, tdat_mag = [], [], [], []
    for f in frames:
        if f.get("tdat"):
            tdat_t.append(f["timestamp"] - t0)
            tdat_angle.append(f["tdat"]["angle"])
            tdat_dist.append(f["tdat"]["distance"])
            tdat_mag.append(f["tdat"]["magnitude"])

    # Collect PDAT data
    pdat_t, pdat_angle, pdat_dist, pdat_mag = [], [], [], []
    for f in frames:
        for pt in f.get("pdat", []):
            if pt is not None:
                pdat_t.append(f["timestamp"] - t0)
                pdat_angle.append(pt["angle"])
                pdat_dist.append(pt["distance"])
                pdat_mag.append(pt["magnitude"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"K-LD7 Capture — {output_path.stem}", fontsize=14)

    # Angle over time
    ax = axes[0]
    if pdat_t:
        ax.scatter(pdat_t, pdat_angle, s=8, alpha=0.4, label="PDAT", color="tab:blue")
    if tdat_t:
        ax.scatter(tdat_t, tdat_angle, s=20, alpha=0.8, label="TDAT", color="tab:orange", marker="x")
    ax.set_ylabel("Angle (\u00b0)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Distance over time
    ax = axes[1]
    if pdat_t:
        ax.scatter(pdat_t, pdat_dist, s=8, alpha=0.4, label="PDAT", color="tab:blue")
    if tdat_t:
        ax.scatter(tdat_t, tdat_dist, s=20, alpha=0.8, label="TDAT", color="tab:orange", marker="x")
    ax.set_ylabel("Distance (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Magnitude over time
    ax = axes[2]
    if pdat_t:
        ax.scatter(pdat_t, pdat_mag, s=8, alpha=0.4, label="PDAT", color="tab:blue")
    if tdat_t:
        ax.scatter(tdat_t, tdat_mag, s=20, alpha=0.8, label="TDAT", color="tab:orange", marker="x")
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze K-LD7 radar capture files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Path to .pkl capture file")
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    parser.add_argument("--pdat", action="store_true", help="Show PDAT multi-target details in timeline")
    parser.add_argument("--detections-only", action="store_true", help="Only show frames with detections")
    parser.add_argument("--timeline", action="store_true", help="Show full frame-by-frame timeline")
    parser.add_argument("--pair-shots", action="store_true", help="Show probable club-to-ball shot pairs")
    parser.add_argument("--min-gap", type=float, default=0.5, help="Min gap (seconds) between events (default: 0.5)")
    args = parser.parse_args()

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    data = load_capture(path)
    frames = data["frames"]
    meta = data["metadata"]

    print()
    print_metadata(meta)

    print("=" * 60)
    print("  DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Total frames:      {len(frames)}")
    if frames:
        duration = frames[-1].get("timestamp", 0) - frames[0].get("timestamp", 0)
        print(f"  Duration:          {duration:.1f}s")
        print(f"  Frame rate:        {len(frames) / duration:.1f} fps" if duration > 0 else "")
    print()
    print("  --- TDAT (tracked target) ---")
    print_tdat_summary(frames)
    print()
    print("  --- PDAT (raw detections) ---")
    print_pdat_summary(frames)

    # Find and print events
    events = find_events(frames, min_gap_s=args.min_gap)
    print_events(frames, events)

    if args.pair_shots:
        orientation = meta.get("orientation", "vertical")
        print_probable_shots(frames, orientation=orientation)

    # Timeline
    if args.timeline:
        print()
        print("=" * 60)
        print("  FRAME TIMELINE")
        print("=" * 60)
        print_timeline(frames, detections_only=args.detections_only, show_pdat=args.pdat)

    # Plots
    if args.plot:
        print()
        plot_capture(frames, path)

    print()


if __name__ == "__main__":
    main()
