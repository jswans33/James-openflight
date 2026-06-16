#!/usr/bin/env python3
"""
Capture raw I/Q data from rolling buffer mode via hardware trigger.

Waits for the sound trigger (SEN-14262 → HOST_INT) to fire, captures the
I/Q buffer dump, runs it through the processor, prints live results,
and saves all captures to a .pkl file for offline analysis.

The radar must already be in persistent rolling buffer mode (see --setup
in test_rolling_buffer_persist.py). This script does NOT send GC — it
assumes the board was configured with A! and power cycled.

Usage:
    uv run python scripts/capture_iq.py
    uv run python scripts/capture_iq.py --pre-trigger 32 --sample-rate 30
    uv run python scripts/capture_iq.py -o my_captures.pkl

To analyze afterwards:
    uv run python src/analysis/analyze_capture.py ~/openflight_sessions/capture_*.pkl
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openflight.ops243 import OPS243Radar
from openflight.rolling_buffer.processor import RollingBufferProcessor


def capture_to_dict(capture, sample_rate_hz, capture_idx):
    """
    Convert an IQCapture to the dict format expected by analyze_capture.py.

    Builds complex_signal from raw I/Q for compatibility with the analysis
    pipeline (time domain plots, spectrogram, FFT).
    """
    i_samples = np.array(capture.i_samples, dtype=np.int16)
    q_samples = np.array(capture.q_samples, dtype=np.int16)

    # Center and scale (12-bit ADC, 3.3V reference)
    i_centered = i_samples.astype(np.float64) - np.mean(i_samples)
    q_centered = q_samples.astype(np.float64) - np.mean(q_samples)
    i_scaled = i_centered * (3.3 / 4096)
    q_scaled = q_centered * (3.3 / 4096)

    complex_signal = i_scaled + 1j * q_scaled

    # sample_time relative to capture start (index-based, seconds)
    sample_time = capture_idx * (len(i_samples) / sample_rate_hz)

    return {
        "sample_time": sample_time,
        "radar_sample_time": capture.sample_time,
        "radar_trigger_time": capture.trigger_time,
        "trigger_offset_ms": capture.trigger_offset_ms,
        "i_samples": i_samples,
        "q_samples": q_samples,
        "complex_signal": complex_signal,
        "capture_timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Capture I/Q data from rolling buffer mode via hardware trigger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Basic capture (Ctrl+C to stop and save)
    uv run python scripts/capture_iq.py

    # With custom pre-trigger and sample rate
    uv run python scripts/capture_iq.py --pre-trigger 32 --sample-rate 30

    # Analyze saved captures
    uv run python src/analysis/analyze_capture.py ~/openflight_sessions/capture_*.pkl
        """,
    )
    parser.add_argument(
        "-o", "--output", help="Output .pkl file path (default: auto-generated in ~/openflight_sessions)"
    )
    parser.add_argument(
        "--port", help="Serial port for radar (auto-detect if not specified)"
    )
    parser.add_argument(
        "--pre-trigger", "-p", type=int, default=16,
        help="Pre-trigger segments S#n, 0-32 (default: 16 = 50/50 split)"
    )
    parser.add_argument(
        "--buffer-split", "-b",
        choices=["balanced", "post-heavy", "pre-heavy"],
        help="Buffer split preset (overrides --pre-trigger): "
             "balanced=S#16 (50/50), post-heavy=S#12 (37/63), pre-heavy=S#24 (75/25)"
    )
    parser.add_argument(
        "--sample-rate", "-s", type=int, default=30,
        help="Sample rate in ksps (default: 30)"
    )
    parser.add_argument(
        "--timeout", "-t", type=float, default=60.0,
        help="Timeout waiting for each trigger in seconds (default: 60)"
    )
    parser.add_argument(
        "--max-captures", "-n", type=int, default=0,
        help="Stop after N captures (default: 0 = unlimited, Ctrl+C to stop)"
    )
    args = parser.parse_args()

    # Resolve buffer split preset
    if args.buffer_split:
        presets = {"balanced": 16, "post-heavy": 12, "pre-heavy": 24}
        args.pre_trigger = presets[args.buffer_split]

    # Output file
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path.home() / "openflight_sessions"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"capture_{timestamp}.pkl"

    sample_rate_hz = args.sample_rate * 1000
    segment_ms = 128 / sample_rate_hz * 1000
    pre_ms = args.pre_trigger * segment_ms
    post_ms = (32 - args.pre_trigger) * segment_ms
    total_ms = 4096 / sample_rate_hz * 1000

    print("=" * 60)
    print("  I/Q Capture Tool (Rolling Buffer + Hardware Trigger)")
    print("=" * 60)
    print()
    print(f"  Output:       {output_path}")
    print(f"  Sample rate:  S={args.sample_rate} ({sample_rate_hz} Hz)")
    print(f"  Pre-trigger:  S#{args.pre_trigger} ({pre_ms:.1f}ms pre / {post_ms:.1f}ms post)")
    print(f"  Buffer:       4096 samples = {total_ms:.1f}ms total")
    print(f"  Timeout:      {args.timeout}s per trigger")
    if args.max_captures > 0:
        print(f"  Max captures: {args.max_captures}")
    print()

    # Connect to radar
    print("Connecting to radar...")
    radar = OPS243Radar(port=args.port if args.port else None)
    radar.connect()
    print(f"  Connected on: {radar.port}")

    info = radar.get_info()
    print(f"  Firmware: {info.get('Version', 'unknown')}")
    print()

    # Arm rolling buffer — board should already be in GC mode (persistent).
    # We just need PA + S#n to start sampling.
    print(f"Arming rolling buffer (PA, S#{args.pre_trigger})...")
    radar.rearm_rolling_buffer(pre_trigger_segments=args.pre_trigger)
    print("  Armed — waiting for triggers")
    print()

    processor = RollingBufferProcessor(sample_rate=sample_rate_hz)

    captures = []
    metadata = {
        "radar_info": info,
        "capture_start": datetime.now().isoformat(),
        "sample_rate": sample_rate_hz,
        "sample_rate_ksps": args.sample_rate,
        "pre_trigger_segments": args.pre_trigger,
        "fft_size": 4096,
        "window_size": 128,
        "mode": "rolling_buffer",
    }

    capture_count = 0
    valid_count = 0

    print("-" * 60)
    print("Waiting for hardware trigger (sound -> HOST_INT)...")
    print("Hit a golf ball or clap near the sensor.")
    print("Ctrl+C to stop and save.")
    print("-" * 60)
    print()

    try:
        while True:
            if args.max_captures > 0 and capture_count >= args.max_captures:
                print(f"\nReached max captures ({args.max_captures}), stopping.")
                break

            # Wait for hardware trigger
            response = radar.wait_for_hardware_trigger(timeout=args.timeout)

            if not response:
                print(f"  [{capture_count + 1}] Timeout — no trigger in {args.timeout}s")
                continue

            capture_count += 1
            response_len = len(response)

            # Parse I/Q data
            capture = processor.parse_capture(response)
            if not capture:
                print(f"  #{capture_count}: PARSE FAILED ({response_len} bytes)")
                radar.rearm_rolling_buffer(pre_trigger_segments=args.pre_trigger)
                continue

            # Run full pipeline
            result = processor.process_capture(capture)

            # Print live results
            trigger_offset = capture.trigger_offset_ms
            if result:
                valid_count += 1
                spin_str = ""
                if result.spin and result.spin.spin_rpm > 0:
                    spin_str = (
                        f"  spin={result.spin.spin_rpm:.0f}rpm "
                        f"(snr={result.spin.snr:.1f}, {result.spin.quality})"
                    )
                elif result.spin and result.spin.rejection_reason:
                    spin_str = (
                        f"  spin=none ({result.spin.rejection_reason}, "
                        f"snr={result.spin.snr:.1f})"
                    )
                club_str = f"  club={result.club_speed_mph:.1f}mph" if result.club_speed_mph else ""
                smash_str = f"  smash={result.smash_factor:.2f}" if result.smash_factor else ""

                print(
                    f"  #{capture_count}: ball={result.ball_speed_mph:.1f}mph"
                    f"{club_str}{smash_str}{spin_str}"
                    f"  (trigger@{trigger_offset:.1f}ms, {response_len}B)"
                )
            else:
                timeline = processor.process_standard(capture)
                outbound = [r for r in timeline.readings if r.is_outbound]
                peak = max((r.speed_mph for r in outbound), default=0)
                print(
                    f"  #{capture_count}: NO SHOT (peak outbound={peak:.1f}mph, "
                    f"{len(outbound)} readings, trigger@{trigger_offset:.1f}ms)"
                )

            # Convert and store for pkl
            capture_dict = capture_to_dict(capture, sample_rate_hz, capture_count - 1)
            if result:
                capture_dict["ball_speed_mph"] = round(result.ball_speed_mph, 1)
                capture_dict["club_speed_mph"] = round(result.club_speed_mph, 1) if result.club_speed_mph else None
                capture_dict["smash_factor"] = round(result.smash_factor, 2) if result.smash_factor else None
                if result.spin:
                    capture_dict["spin_rpm"] = round(result.spin.spin_rpm)
                    capture_dict["spin_snr"] = round(result.spin.snr, 1)
                    capture_dict["spin_quality"] = result.spin.quality
                    capture_dict["spin_modulation_depth"] = result.spin.modulation_depth
                    capture_dict["spin_peak_freq_hz"] = result.spin.peak_freq_hz
                    capture_dict["spin_candidate_rpm"] = (
                        round(result.spin.peak_freq_hz * 60)
                        if result.spin.peak_freq_hz is not None else None
                    )
                    capture_dict["spin_seam_cycles"] = result.spin.seam_cycles
                    capture_dict["spin_at_lower_rail"] = result.spin.at_lower_rail
                    capture_dict["spin_at_upper_rail"] = result.spin.at_upper_rail
                    capture_dict["spin_rejection_reason"] = (
                        result.spin.rejection_reason
                    )
            captures.append(capture_dict)

            # Re-arm for next trigger
            radar.rearm_rolling_buffer(pre_trigger_segments=args.pre_trigger)

    except KeyboardInterrupt:
        print()

    finally:
        print()
        print("=" * 60)
        print("  CAPTURE SUMMARY")
        print("=" * 60)
        print(f"  Total triggers:  {capture_count}")
        print(f"  Valid captures:  {valid_count}")
        if capture_count > 0:
            print(f"  Success rate:    {valid_count / capture_count * 100:.0f}%")

        # Save to pickle
        if captures:
            metadata["capture_end"] = datetime.now().isoformat()
            metadata["total_captures"] = len(captures)
            metadata["valid_captures"] = valid_count

            output_data = {
                "metadata": metadata,
                "captures": captures,
            }

            with open(output_path, "wb") as f:
                pickle.dump(output_data, f)

            print(f"  Output file:     {output_path}")
            print(f"  File size:       {output_path.stat().st_size / 1024:.1f} KB")
            print()
            print("  To analyze:")
            print(f"    uv run python src/analysis/analyze_capture.py {output_path}")
        else:
            print("\n  No captures to save.")

        print("=" * 60)
        radar.disconnect()


if __name__ == "__main__":
    main()
