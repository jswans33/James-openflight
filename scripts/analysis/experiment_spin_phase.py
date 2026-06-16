#!/usr/bin/env python3
"""Test phase-based spin confirmation against TrackMan truth.

This is offline-only. It compares the production amplitude-envelope spin
candidate with two phase-derived witnesses:

- residual unwrapped phase
- residual instantaneous frequency

The goal is not to replace the envelope detector. It is to see whether phase
can confirm currently rejected low-SNR envelope candidates without introducing
large false positives.

Usage:
    uv run --no-sync python scripts/analysis/experiment_spin_phase.py \
        --openflight session_logs/session_20260511_120001_range.jsonl \
        --comparison session_logs/comparison_test2.csv \
        --output session_logs/spin_phase_experiment_test2.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_trackman as ct  # noqa: E402  pylint: disable=wrong-import-position
from experiment_spin_windows import (  # noqa: E402
    _club_enum,
    _current_window,
    _load_session_entries,
    _load_trackman_by_shot,
    _to_int,
)

from openflight.rolling_buffer.monitor import get_optimal_spin_for_ball_speed  # noqa: E402
from openflight.rolling_buffer.processor import RollingBufferProcessor  # noqa: E402
from openflight.rolling_buffer.types import IQCapture  # noqa: E402


@dataclass(frozen=True)
class PhaseSpinResult:
    """Phase-derived spin candidate."""

    method: str
    rpm: Optional[float]
    snr: float
    seam_cycles: Optional[float]
    at_lower_rail: bool
    at_upper_rail: bool
    reason: Optional[str] = None

    @property
    def reportable(self) -> bool:
        return bool(
            self.rpm is not None
            and self.snr >= 2.5
            and not self.at_lower_rail
            and not self.at_upper_rail
        )


def _post_onset_filtered_iq(
    processor: RollingBufferProcessor,
    capture: IQCapture,
    ball_speed_mph: float,
    ball_timestamp_ms: float,
) -> Optional[np.ndarray]:
    i_data = np.array(capture.i_samples, dtype=np.float64)
    q_data = np.array(capture.q_samples, dtype=np.float64)
    i_data -= np.mean(i_data)
    q_data -= np.mean(q_data)
    iq = i_data + 1j * q_data

    ball_speed_mps = ball_speed_mph / processor.MPS_TO_MPH
    ball_doppler_hz = 2 * ball_speed_mps / processor.WAVELENGTH_M
    nyquist = processor.SAMPLE_RATE / 2
    low = max((ball_doppler_hz - processor.SPIN_BANDPASS_BW_HZ) / nyquist, 0.001)
    high = min((ball_doppler_hz + processor.SPIN_BANDPASS_BW_HZ) / nyquist, 0.999)
    if low >= high:
        return None

    sos = butter(processor.SPIN_BANDPASS_ORDER, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, iq)
    start_sample = max(0, int(ball_timestamp_ms * processor.SAMPLE_RATE / 1000))
    return filtered[start_sample:]


def _candidate_from_signal(
    processor: RollingBufferProcessor,
    method: str,
    signal: np.ndarray,
    sample_rate_hz: float,
    expected_spin_rpm: Optional[float],
) -> PhaseSpinResult:
    if len(signal) < processor.SPIN_MIN_SAMPLES:
        return PhaseSpinResult(
            method=method,
            rpm=None,
            snr=0.0,
            seam_cycles=None,
            at_lower_rail=False,
            at_upper_rail=False,
            reason=f"Signal too short ({len(signal)} samples)",
        )

    centered = signal - np.mean(signal)
    if np.std(centered) < 1e-9:
        return PhaseSpinResult(
            method=method,
            rpm=None,
            snr=0.0,
            seam_cycles=None,
            at_lower_rail=False,
            at_upper_rail=False,
            reason="Signal variation too low",
        )

    windowed = centered * np.hanning(len(centered))
    fft_result = np.fft.fft(windowed, processor.SPIN_ENVELOPE_FFT_SIZE)
    freqs = np.fft.fftfreq(processor.SPIN_ENVELOPE_FFT_SIZE, d=1 / sample_rate_hz)
    half = processor.SPIN_ENVELOPE_FFT_SIZE // 2
    magnitude = np.abs(fft_result[1:half])
    freqs = freqs[1:half]
    valid_mask = (freqs >= processor.SPIN_MIN_SEAM_HZ) & (freqs <= processor.SPIN_MAX_SEAM_HZ)
    if not np.any(valid_mask):
        return PhaseSpinResult(
            method=method,
            rpm=None,
            snr=0.0,
            seam_cycles=None,
            at_lower_rail=False,
            at_upper_rail=False,
            reason="No valid seam frequencies",
        )

    valid_mag = magnitude[valid_mask].copy()
    valid_freqs = freqs[valid_mask]
    n_valid = len(valid_mag)
    leakage = min(processor.SPIN_DC_LEAKAGE_BINS, max(0, n_valid - 1))
    if leakage > 0:
        valid_mag[:leakage] = 0
    if not np.any(valid_mag > 0):
        return PhaseSpinResult(
            method=method,
            rpm=None,
            snr=0.0,
            seam_cycles=None,
            at_lower_rail=False,
            at_upper_rail=False,
            reason="No nonzero seam magnitudes",
        )

    peak_idx = _select_phase_peak(
        processor,
        valid_mag,
        valid_freqs,
        leakage,
        expected_spin_rpm,
    )
    noise_floor = np.median(valid_mag[valid_mag > 0])
    peak_freq = float(valid_freqs[peak_idx])
    rpm = peak_freq * 60
    snr = float(valid_mag[peak_idx] / noise_floor) if noise_floor > 0 else 0.0
    seam_cycles = peak_freq * len(centered) / sample_rate_hz
    return PhaseSpinResult(
        method=method,
        rpm=rpm,
        snr=snr,
        seam_cycles=seam_cycles,
        at_lower_rail=peak_idx < leakage + processor.SPIN_UPPER_RAIL_BINS,
        at_upper_rail=peak_idx >= n_valid - processor.SPIN_UPPER_RAIL_BINS,
        reason=None,
    )


def _select_phase_peak(
    processor: RollingBufferProcessor,
    valid_mag: np.ndarray,
    valid_freqs: np.ndarray,
    leakage_bins: int,
    expected_spin_rpm: Optional[float],
) -> int:
    """Use the same conservative prior shape as the envelope detector."""
    strongest_idx = int(np.argmax(valid_mag))
    if expected_spin_rpm is None or expected_spin_rpm <= 0:
        return strongest_idx

    peak_indices = set(find_peaks(valid_mag, distance=2)[0])
    peak_indices.add(strongest_idx)
    lower_rail_limit = leakage_bins + processor.SPIN_UPPER_RAIL_BINS
    upper_rail_start = len(valid_mag) - processor.SPIN_UPPER_RAIL_BINS
    strongest_rpm = float(valid_freqs[strongest_idx] * 60)
    strongest_error = abs(strongest_rpm - expected_spin_rpm) / expected_spin_rpm
    candidates = []
    for idx in peak_indices:
        if idx < lower_rail_limit or idx >= upper_rail_start:
            continue
        relative_mag = float(valid_mag[idx] / valid_mag[strongest_idx])
        if relative_mag < processor.SPIN_PRIOR_MIN_RELATIVE_MAG:
            continue
        rpm = float(valid_freqs[idx] * 60)
        relative_error = abs(rpm - expected_spin_rpm) / expected_spin_rpm
        if relative_error > processor.SPIN_PRIOR_MAX_RELATIVE_ERROR:
            continue
        candidates.append((relative_error, -relative_mag, int(idx)))

    if not candidates:
        return strongest_idx

    best_error, _, best_idx = min(candidates)
    if strongest_idx < lower_rail_limit or (
        strongest_error > processor.SPIN_PRIOR_STRONGEST_FAR_ERROR
        and best_error < strongest_error
    ):
        return best_idx
    return strongest_idx


def _phase_methods(
    processor: RollingBufferProcessor,
    filtered_post_onset: np.ndarray,
    start_sample: int,
    end_sample: int,
    expected_spin_rpm: Optional[float],
) -> list[PhaseSpinResult]:
    segment = filtered_post_onset[start_sample:end_sample]
    if len(segment) < processor.SPIN_MIN_SAMPLES:
        return []

    phase = np.unwrap(np.angle(segment))
    x = np.arange(len(phase), dtype=np.float64)
    slope, intercept = np.polyfit(x, phase, 1)
    phase_residual = phase - (slope * x + intercept)
    phase_result = _candidate_from_signal(
        processor,
        "phase_residual",
        phase_residual,
        processor.SAMPLE_RATE,
        expected_spin_rpm,
    )

    inst_freq = np.diff(phase) * processor.SAMPLE_RATE / (2 * np.pi)
    if len(inst_freq) >= processor.SPIN_MIN_SAMPLES:
        x_freq = np.arange(len(inst_freq), dtype=np.float64)
        slope_freq, intercept_freq = np.polyfit(x_freq, inst_freq, 1)
        freq_residual = inst_freq - (slope_freq * x_freq + intercept_freq)
        freq_result = _candidate_from_signal(
            processor,
            "instant_frequency",
            freq_residual,
            processor.SAMPLE_RATE,
            expected_spin_rpm,
        )
    else:
        freq_result = PhaseSpinResult(
            method="instant_frequency",
            rpm=None,
            snr=0.0,
            seam_cycles=None,
            at_lower_rail=False,
            at_upper_rail=False,
            reason=f"Signal too short ({len(inst_freq)} samples)",
        )

    return [phase_result, freq_result]


def _agreement_pct(envelope_rpm: Optional[float], phase_rpm: Optional[float]) -> Optional[float]:
    if envelope_rpm is None or phase_rpm is None or envelope_rpm <= 0:
        return None
    return abs(envelope_rpm - phase_rpm) / envelope_rpm * 100


def _envelope_candidate_rpm(spin: Any) -> Optional[float]:
    if spin is None or spin.peak_freq_hz is None:
        return None
    return spin.peak_freq_hz * 60


def _envelope_nonrail(spin: Any) -> bool:
    return bool(
        spin is not None
        and _envelope_candidate_rpm(spin) is not None
        and not spin.at_lower_rail
        and not spin.at_upper_rail
    )


def _envelope_reportable(spin: Any) -> bool:
    return bool(
        _envelope_nonrail(spin)
        and spin.spin_rpm is not None
        and spin.spin_rpm > 0
        and spin.snr >= 2.5
    )


def _combined_accept(spin: Any, phase: PhaseSpinResult) -> bool:
    agreement = _agreement_pct(_envelope_candidate_rpm(spin), phase.rpm)
    return bool(
        _envelope_nonrail(spin)
        and spin.snr >= 1.75
        and phase.rpm is not None
        and phase.snr >= 2.5
        and not phase.at_lower_rail
        and not phase.at_upper_rail
        and agreement is not None
        and agreement <= 10.0
    )


def _rows(
    shots: list[dict],
    captures: list[dict],
    trackman_by_shot: dict[int, dict[str, Any]],
    sample_rate_hz: int,
) -> list[dict[str, Any]]:
    processor = RollingBufferProcessor(sample_rate=sample_rate_hz)
    rows = []
    for shot_entry, capture_entry in zip(shots, captures):
        shot_data = shot_entry.get("data", shot_entry)
        shot_number = _to_int(shot_data.get("shot_number"))
        trackman = trackman_by_shot.get(shot_number or -1, {})
        if trackman.get("match_quality") != "good" or trackman.get("spin_tm") is None:
            continue

        normalized_club = ct.normalize_club(shot_data.get("club"))
        club = _club_enum(normalized_club)
        capture = IQCapture(
            sample_time=capture_entry.get("sample_time", 0),
            trigger_time=capture_entry.get("trigger_time", 0),
            i_samples=capture_entry["i_samples"],
            q_samples=capture_entry["q_samples"],
        )
        processed = processor.process_capture(
            capture,
            expected_spin_for_ball_speed=lambda ball_speed, club=club: (
                get_optimal_spin_for_ball_speed(ball_speed, club)
            ),
        )
        if not processed:
            continue

        filtered = _post_onset_filtered_iq(
            processor,
            capture,
            processed.ball_speed_mph,
            processed.ball_timestamp_ms,
        )
        if filtered is None:
            continue

        window = _current_window(processor, np.abs(filtered))
        expected_spin = get_optimal_spin_for_ball_speed(processed.ball_speed_mph, club)
        envelope_spin = processed.spin
        envelope_rpm = _envelope_candidate_rpm(envelope_spin)
        phase_results = _phase_methods(
            processor,
            filtered,
            window.start_sample,
            window.end_sample,
            expected_spin,
        )

        for phase in phase_results:
            agreement = _agreement_pct(envelope_rpm, phase.rpm)
            combined_accept = _combined_accept(envelope_spin, phase)
            accepted_rpm = envelope_rpm if combined_accept else None
            spin_tm = trackman["spin_tm"]
            rows.append({
                "shot_number": shot_number,
                "club": normalized_club,
                "method": phase.method,
                "ball_speed_of": round(processed.ball_speed_mph, 3),
                "ball_speed_tm": trackman.get("ball_speed_tm"),
                "expected_spin_rpm": round(expected_spin),
                "spin_tm": spin_tm,
                "envelope_rpm": round(envelope_rpm) if envelope_rpm else None,
                "envelope_snr": envelope_spin.snr if envelope_spin else None,
                "envelope_reportable": _envelope_reportable(envelope_spin),
                "envelope_nonrail": _envelope_nonrail(envelope_spin),
                "envelope_error_rpm": (
                    round(envelope_rpm - spin_tm, 1) if envelope_rpm else None
                ),
                "phase_rpm": round(phase.rpm) if phase.rpm else None,
                "phase_snr": round(phase.snr, 2),
                "phase_reportable": phase.reportable,
                "phase_error_rpm": (
                    round(phase.rpm - spin_tm, 1) if phase.reportable else None
                ),
                "phase_seam_cycles": (
                    round(phase.seam_cycles, 2)
                    if phase.seam_cycles is not None else None
                ),
                "phase_at_lower_rail": phase.at_lower_rail,
                "phase_at_upper_rail": phase.at_upper_rail,
                "phase_reason": phase.reason,
                "agreement_pct": round(agreement, 1) if agreement is not None else None,
                "combined_accept": combined_accept,
                "combined_rpm": round(accepted_rpm) if accepted_rpm else None,
                "combined_error_rpm": (
                    round(accepted_rpm - spin_tm, 1) if accepted_rpm else None
                ),
            })
    return rows


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(errors: list[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not errors:
        return None, None, None
    return (
        statistics.mean(errors),
        statistics.mean(abs(error) for error in errors),
        math.sqrt(statistics.mean(error * error for error in errors)),
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(f"{'strategy':<28} {'n':>3} {'MAE':>8} {'bias':>8} {'RMSE':>8}")
    strategies = []

    env_once = {}
    for row in rows:
        shot = row["shot_number"]
        if row["envelope_reportable"] and shot not in env_once:
            env_once[shot] = float(row["envelope_error_rpm"])
    strategies.append(("envelope_current", list(env_once.values())))

    for method in sorted({row["method"] for row in rows}):
        phase_errors = [
            float(row["phase_error_rpm"])
            for row in rows
            if row["method"] == method and row["phase_reportable"]
        ]
        combined_errors = [
            float(row["combined_error_rpm"])
            for row in rows
            if row["method"] == method and row["combined_accept"]
        ]
        recoveries = {
            row["shot_number"]: float(row["combined_error_rpm"])
            for row in rows
            if row["method"] == method
            and row["combined_accept"]
            and not row["envelope_reportable"]
        }
        union_errors = {
            **env_once,
            **recoveries,
        }
        strategies.append((f"{method}_only", phase_errors))
        strategies.append((f"combined_{method}", combined_errors))
        strategies.append((f"recoveries_{method}", list(recoveries.values())))
        strategies.append((f"envelope_plus_{method}", list(union_errors.values())))

    all_recoveries = {
        row["shot_number"]: float(row["combined_error_rpm"])
        for row in rows
        if row["combined_accept"] and not row["envelope_reportable"]
    }
    strategies.append(("recoveries_any_phase", list(all_recoveries.values())))
    strategies.append(("envelope_plus_any_phase", list({**env_once, **all_recoveries}.values())))

    for name, errors in strategies:
        bias, mae, rmse = _metric(errors)
        print(
            f"{name:<28} {len(errors):>3} "
            f"{mae:>8.1f} {bias:>8.1f} {rmse:>8.1f}"
            if errors else f"{name:<28} {0:>3} {'-':>8} {'-':>8} {'-':>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openflight", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=30000)
    args = parser.parse_args()

    shots, captures = _load_session_entries(args.openflight)
    trackman_by_shot = _load_trackman_by_shot(args.comparison)
    rows = _rows(shots, captures, trackman_by_shot, args.sample_rate)
    _write_csv(rows, args.output)
    _print_summary(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
