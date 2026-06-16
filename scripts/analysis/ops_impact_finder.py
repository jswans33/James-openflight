#!/usr/bin/env python3
"""OPS-based impact-time finder for K-LD7 trajectory fitting.

For each shot in a session JSONL, derive Pi-clock impact timestamp from
the OPS rolling-buffer capture and the first-byte timing.

Two sources are reconciled per shot:

  1. OPS-reported `ball_timestamp_ms` (the OPS firmware's onset detection
     of the ball signal in the rolling buffer, in milliseconds from the
     start of the buffer).
  2. Independent verification: STFT of the raw I/Q samples to find when
     the high-Doppler ball signal first crosses a threshold.

If the two agree to within ~5 ms, we trust ball_timestamp_ms and convert
to Pi-clock time:

    Pi_impact_ts = first_byte_ts - (buffer_duration_ms - ball_timestamp_ms) / 1000

(The OPS sends the first byte just after the last sample in the buffer.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 30_000
BUFFER_DURATION_MS = 4096 / SAMPLE_RATE_HZ * 1000.0  # ~136.5 ms


def load_first_byte_times(csv_path: Path, long_name: str) -> dict[int, float]:
    """Return {json_shot_no -> first_byte_ts_epoch}."""
    out: dict[int, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("session") != long_name:
                continue
            shot_no = row.get("json_shot_no")
            ts_str = row.get("first_byte_ts", "")
            try:
                out[int(shot_no)] = datetime.fromisoformat(ts_str).timestamp()
            except (TypeError, ValueError):
                pass
    return out


def stft_ball_onset_ms(
    i_samples: np.ndarray,
    q_samples: np.ndarray,
    ball_speed_mph: float,
    window_samples: int = 128,
    hop_samples: int = 32,
    min_doppler_hz: float = 2000.0,
) -> float | None:
    """Detect ball-signal onset from STFT of raw I/Q.

    Looks for the first window where a peak appears at Doppler > min_doppler_hz
    (well above static clutter) with magnitude > 3× the noise floor.

    Returns onset time in ms from start of buffer, or None if not found.
    """
    iq = i_samples.astype(np.float64) + 1j * q_samples.astype(np.float64)
    iq = iq - np.mean(iq)  # DC removal

    n_total = len(iq)
    n_windows = (n_total - window_samples) // hop_samples + 1

    # Doppler frequency bin spacing
    freq_resolution_hz = SAMPLE_RATE_HZ / window_samples
    min_bin = int(min_doppler_hz / freq_resolution_hz)

    noise_floors: list[float] = []
    peak_above_floor_idx: int | None = None

    for w in range(n_windows):
        start = w * hop_samples
        seg = iq[start: start + window_samples] * np.hanning(window_samples)
        spec = np.abs(np.fft.fft(seg))
        # Keep one half + treat outbound vs inbound; for golf the ball is
        # moving away, so positive Doppler. But OPS aliasing means it may
        # appear in either half. Scan both ends above min_bin.
        relevant = np.concatenate([
            spec[min_bin: window_samples // 2],
            spec[window_samples // 2 + 1: window_samples - min_bin],
        ])
        if len(relevant) == 0:
            continue
        peak = float(np.max(relevant))

        # Noise floor: take the median of the first few windows where
        # presumably no ball signal exists.
        if w < 5:
            noise_floors.append(peak)
            continue
        noise = float(np.median(noise_floors)) if noise_floors else 1.0
        if peak > 3.0 * noise and peak_above_floor_idx is None:
            peak_above_floor_idx = w
            break

    if peak_above_floor_idx is None:
        return None
    onset_sample = peak_above_floor_idx * hop_samples + window_samples // 2
    return onset_sample / SAMPLE_RATE_HZ * 1000.0


def find_rolling_buffer_captures(jsonl_path: Path) -> dict[int, dict]:
    """Return {shot_number -> rolling_buffer_capture entry}."""
    out: dict[int, dict] = {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "rolling_buffer_capture":
                continue
            shot_n = d.get("shot_number")
            if shot_n is None:
                continue
            out[int(shot_n)] = d
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--session-jsonl", type=Path,
        default=Path("/Users/john.pacino/openflight_sessions/session_20260523_143732_range.jsonl"),
    )
    p.add_argument(
        "--timing-csv", type=Path,
        default=Path("/Users/john.pacino/Desktop/openflight_sessions/_analysis_terminal_shot_matching/terminal_shot_timing_strict_matches.csv"),
    )
    p.add_argument("--long-name", default="20260523_143732_18deg_7iron_8shots")
    p.add_argument(
        "--first-byte-trigger-delay-ms", type=float, default=68.0,
        help="Delay between hardware trigger firing and OPS first-byte "
             "arriving on Pi (default 68ms = post-trigger buffer duration).",
    )
    args = p.parse_args()

    captures = find_rolling_buffer_captures(args.session_jsonl)
    first_byte_times = load_first_byte_times(args.timing_csv, args.long_name)

    if not captures:
        print(f"No rolling_buffer_capture entries in {args.session_jsonl}", file=sys.stderr)
        return

    print(f"{'shot':>4}  {'OPS ball_ts_ms':>14}  {'STFT onset_ms':>14}  "
          f"{'agreement_ms':>13}  {'trigger_off_ms':>14}  {'fb_ts (Pi)':>22}  "
          f"{'Pi_impact_ts':>22}  {'Pi_impact rel fb':>17}")
    print("-" * 145)

    for shot_n in sorted(captures.keys()):
        cap = captures[shot_n]
        ops_ball_ms = cap.get("ball_timestamp_ms")
        trigger_off_ms = cap.get("trigger_offset_ms")
        ball_speed = cap.get("ball_speed_mph") or 0.0
        i_samples = np.array(cap["i_samples"], dtype=np.int32)
        q_samples = np.array(cap["q_samples"], dtype=np.int32)

        stft_onset_ms = stft_ball_onset_ms(i_samples, q_samples, ball_speed)
        agreement = (
            stft_onset_ms - ops_ball_ms
            if stft_onset_ms is not None and ops_ball_ms is not None
            else None
        )

        fb_ts = first_byte_times.get(shot_n)
        pi_impact_ts: float | None = None
        rel_fb: float | None = None
        if fb_ts is not None and ops_ball_ms is not None:
            # Pi-clock impact:
            # = first_byte_ts - (post-trigger buffer duration)
            #     - (trigger_offset_ms - ball_timestamp_ms)/1000
            # The first term puts us at trigger, the second at impact.
            trigger_delay_s = args.first_byte_trigger_delay_ms / 1000.0
            pi_impact_ts = (
                fb_ts - trigger_delay_s
                - (trigger_off_ms - ops_ball_ms) / 1000.0
            )
            rel_fb = pi_impact_ts - fb_ts

        ops_str = f"{ops_ball_ms:.2f}" if ops_ball_ms is not None else "—"
        stft_str = f"{stft_onset_ms:.2f}" if stft_onset_ms is not None else "—"
        agree_str = f"{agreement:+.2f}" if agreement is not None else "—"
        trig_str = f"{trigger_off_ms:.2f}" if trigger_off_ms is not None else "—"
        fb_str = f"{fb_ts:.3f}" if fb_ts is not None else "—"
        pi_str = f"{pi_impact_ts:.3f}" if pi_impact_ts is not None else "—"
        rel_str = f"{rel_fb*1000:+.1f}ms" if rel_fb is not None else "—"

        print(f"{shot_n:>4}  {ops_str:>14}  {stft_str:>14}  {agree_str:>13}  "
              f"{trig_str:>14}  {fb_str:>22}  {pi_str:>22}  {rel_str:>17}")


if __name__ == "__main__":
    main()
