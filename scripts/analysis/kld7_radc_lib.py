"""Standalone helpers for K-LD7 raw ADC (RADC) signal processing.

Core processing functions live in src/openflight/kld7/radc.py.
This module re-exports them and adds offline-analysis-only functions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Re-export core processing functions
from openflight.kld7.radc import (  # noqa: F401, E402
    ANTENNA_SPACING_M,
    DC_MASK_BINS,
    RADC_PAYLOAD_BYTES,
    SAMPLES_PER_CHANNEL,
    WAVELENGTH_M,
    CFARDetection,
    RADCChannelStats,
    RADCFrameDiagnostics,
    _velocity_to_bin,
    aliased_velocity_from_ball_speed_mph,
    ball_bin_range_from_speed,
    bin_to_velocity_kmh,
    cfar_detect,
    circular_bin_distance,
    compute_fft_complex,
    compute_spectrum,
    default_ball_bin_ranges,
    expected_ball_bin_from_speed,
    extract_launch_angle,
    find_impact_frames,
    parse_radc_payload,
    per_bin_angle_deg,
    radc_capture_diagnostics,
    radc_frame_diagnostics,
    summarize_radc_diagnostics,
    to_complex_iq,
)

# Keep local: offline-only constants, types, and functions below

ADC_MIDPOINT = 32768  # uint16 midpoint for DC offset removal


@dataclass(frozen=True)
class RADCDetection:
    frame_index: int
    timestamp: float
    distance_m: float
    velocity_kmh: float
    angle_deg: float
    magnitude: float
    snr_db: float
    bin_index: int


def estimate_angle_from_phase(
    f1_complex: np.ndarray,
    f2_complex: np.ndarray,
) -> float:
    """Estimate angle from phase difference between two frequency channels.

    Uses cross-correlation phase to estimate the angle of arrival.
    The exact angle-to-phase mapping depends on K-LD7 antenna geometry
    (spacing, wavelength). This returns a proportional estimate that
    needs empirical calibration against known angles.

    Returns:
        Angle estimate in degrees (uncalibrated — proportional to phase diff)
    """
    # Cross-spectral phase
    cross = np.sum(f1_complex * np.conj(f2_complex))
    phase_rad = np.angle(cross)
    # Convert to degrees — scale factor TBD from calibration
    # For K-LD7 at 24 GHz with ~6mm antenna spacing, rough estimate:
    # angle ≈ arcsin(phase / pi) * (180/pi)
    # For now return raw phase in degrees as a proportional estimate
    return float(np.degrees(phase_rad))


# --- Beamforming and spatial filtering ---

# Aliased velocity ranges at RSPI=3 (100 km/h max, 200 km/h unambiguous).
# Golf ball speeds wrap into the negative velocity range:
#   Ball 100-120 mph (161-193 km/h) → aliases to -39 to -7 km/h
#   Club  70-85  mph (113-137 km/h) → aliases to -87 to -63 km/h
BALL_VELOCITY_ALIASED_MIN_KMH = -39.0
BALL_VELOCITY_ALIASED_MAX_KMH = -7.0
CLUB_VELOCITY_ALIASED_MIN_KMH = -87.0
CLUB_VELOCITY_ALIASED_MAX_KMH = -63.0


def ball_bin_range(fft_size: int = 2048, max_speed_kmh: float = 100.0) -> tuple[int, int]:
    """Return (lo, hi) FFT bin range for aliased ball velocities (broad default)."""
    return (
        _velocity_to_bin(BALL_VELOCITY_ALIASED_MIN_KMH, fft_size, max_speed_kmh),
        _velocity_to_bin(BALL_VELOCITY_ALIASED_MAX_KMH, fft_size, max_speed_kmh),
    )


def club_bin_range(fft_size: int = 2048, max_speed_kmh: float = 100.0) -> tuple[int, int]:
    """Return (lo, hi) FFT bin range for aliased club velocities."""
    return (
        _velocity_to_bin(CLUB_VELOCITY_ALIASED_MIN_KMH, fft_size, max_speed_kmh),
        _velocity_to_bin(CLUB_VELOCITY_ALIASED_MAX_KMH, fft_size, max_speed_kmh),
    )


def compute_angle_velocity_map(
    f1a_iq: np.ndarray,
    f2a_iq: np.ndarray,
    fft_size: int = 2048,
    steer_angles_deg: np.ndarray | None = None,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute angle-velocity power map via conventional beamforming.

    Steers a 2-element array across angles and computes beamformed power
    at each (angle, velocity_bin) pair.

    Returns:
        (power_map, steer_angles, velocity_bins)
        power_map shape: (n_angles, fft_size)
    """
    if steer_angles_deg is None:
        steer_angles_deg = np.arange(-90, 91, 1.0)

    f1a_fft = compute_fft_complex(f1a_iq, fft_size=fft_size)
    f2a_fft = compute_fft_complex(f2a_iq, fft_size=fft_size)

    power_map = np.zeros((len(steer_angles_deg), fft_size))

    for i, angle_deg in enumerate(steer_angles_deg):
        angle_rad = np.radians(angle_deg)
        # Phase shift to steer beam to this angle
        steering_phase = 2.0 * np.pi * antenna_spacing_m * np.sin(angle_rad) / wavelength_m
        # Apply steering vector to second channel and sum
        steered = f1a_fft + f2a_fft * np.exp(-1j * steering_phase)
        power_map[i, :] = np.abs(steered) ** 2

    return power_map, steer_angles_deg, np.arange(fft_size)


@dataclass(frozen=True)
class SpatialDetection:
    """Detection with per-bin angle from interferometry."""
    frame_index: int
    timestamp: float
    velocity_kmh: float
    angle_deg: float  # per-bin angle from phase difference
    magnitude: float
    snr_db: float
    bin_index: int


def process_radc_frame_spatial(
    frame: dict,
    frame_index: int,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    cfar_threshold: float = 4.0,
    cfar_guard: int = 4,
    cfar_training: int = 16,
    bin_range: tuple[int, int] | None = None,
) -> list[SpatialDetection]:
    """Process one RADC frame with per-bin angle estimation.

    Args:
        bin_range: Optional (lo, hi) to restrict CFAR to a specific FFT bin
                   range (e.g. ball or club velocity band). If None, runs on
                   the full spectrum.
    """
    radc_raw = frame.get("radc")
    if radc_raw is None:
        return []

    channels = parse_radc_payload(radc_raw) if isinstance(radc_raw, bytes) else radc_raw

    f1a_iq = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
    f2a_iq = to_complex_iq(channels["f2a_i"], channels["f2a_q"])

    spectrum = compute_spectrum(f1a_iq, fft_size=fft_size)

    # Band-limited CFAR: zero out everything outside the target bin range
    if bin_range is not None:
        masked = np.zeros_like(spectrum)
        lo, hi = bin_range
        masked[lo:hi] = spectrum[lo:hi]
        spectrum = masked

    cfar_hits = cfar_detect(
        spectrum,
        guard_cells=cfar_guard,
        training_cells=cfar_training,
        threshold_factor=cfar_threshold,
    )

    # Complex FFTs for per-bin angle
    f1a_fft = compute_fft_complex(f1a_iq, fft_size=fft_size)
    f2a_fft = compute_fft_complex(f2a_iq, fft_size=fft_size)
    angles = per_bin_angle_deg(f1a_fft, f2a_fft)

    timestamp = float(frame["timestamp"])
    detections = []
    for hit in cfar_hits:
        velocity = bin_to_velocity_kmh(hit.bin_index, fft_size, max_speed_kmh)
        detections.append(
            SpatialDetection(
                frame_index=frame_index,
                timestamp=timestamp,
                velocity_kmh=velocity,
                angle_deg=float(angles[hit.bin_index]),
                magnitude=hit.magnitude,
                snr_db=hit.snr_db,
                bin_index=hit.bin_index,
            )
        )
    return detections


def analyze_capture(
    data: dict,
    angle_offset_deg: float = 0.0,
    speed_tolerance_mph: float = 10.0,
) -> list[dict]:
    """Analyze a full RADC capture with OPS243 shot anchoring.

    Reads the pkl data dict (frames + ops243_shots/ops243_captures) and
    returns one result per shot. If OPS243 data is present, each shot's
    ball speed narrows the K-LD7 velocity search. Otherwise falls back
    to the broad default ball velocity range.

    Pairing: K-LD7 impact events are paired with OPS243 shots by
    sequence order (1st K-LD7 impact = 1st OPS shot), NOT by timestamp.
    The OPS243 timestamp has ~1s latency from I/Q read time, making
    direct timestamp matching unreliable.

    Args:
        data: Dict from pickle.load() with keys: frames, ops243_shots (optional)
        angle_offset_deg: Angle offset to apply
        speed_tolerance_mph: Velocity search window ± around OPS243 speed

    Returns:
        List of shot dicts from extract_launch_angle, with ops243_ball_speed_mph
        added to each.
    """
    frames = data["frames"]
    ops243_shots = data.get("ops243_shots", [])
    ops243_captures = data.get("ops243_captures", [])

    # Use shots if available, fall back to captures
    ops_data = ops243_shots or ops243_captures

    if not ops_data:
        # No OPS243 — use broad velocity range
        return extract_launch_angle(
            frames, angle_offset_deg=angle_offset_deg,
            speed_tolerance_mph=speed_tolerance_mph,
        )

    # Run extract_launch_angle on the full capture, using the first
    # OPS243 ball speed to anchor the velocity search. This lets
    # find_impact_frames detect energy spikes across the whole timeline
    # and group them into shots, then we pair by sequence.
    #
    # For multi-shot captures with varying ball speeds, we'd need to
    # run per-shot with different speeds. For now use the first shot's
    # speed as the anchor (all shots in a session are typically similar).
    first_ball_speed = None
    for s in ops_data:
        speed = s.get("ball_speed_mph")
        if speed and speed > 0:
            first_ball_speed = speed
            break

    all_shots = extract_launch_angle(
        frames,
        ops243_ball_speed_mph=first_ball_speed,
        angle_offset_deg=angle_offset_deg,
        speed_tolerance_mph=speed_tolerance_mph,
    )

    # Pair K-LD7 shots with OPS243 by sequence
    for i, shot in enumerate(all_shots):
        if i < len(ops_data):
            ops = ops_data[i]
            shot["ops243_ball_speed_mph"] = ops.get("ball_speed_mph")
            shot["ops243_club_speed_mph"] = ops.get("club_speed_mph")
            shot["ops243_impact_timestamp"] = ops.get(
                "impact_timestamp", ops.get("timestamp")
            )

    return all_shots


def process_radc_frame(
    frame: dict,
    frame_index: int,
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    cfar_threshold: float = 8.0,
    cfar_guard: int = 4,
    cfar_training: int = 16,
) -> list[RADCDetection]:
    """Process one RADC frame: parse → FFT → CFAR → physical units.

    Uses F1A channel as primary, F2A for angle estimation.
    """
    radc_raw = frame.get("radc")
    if radc_raw is None:
        return []

    if isinstance(radc_raw, bytes):
        channels = parse_radc_payload(radc_raw)
    else:
        channels = radc_raw

    f1a = to_complex_iq(channels["f1a_i"], channels["f1a_q"])
    f2a = to_complex_iq(channels["f2a_i"], channels["f2a_q"])

    spectrum = compute_spectrum(f1a, fft_size=fft_size)
    cfar_hits = cfar_detect(
        spectrum,
        guard_cells=cfar_guard,
        training_cells=cfar_training,
        threshold_factor=cfar_threshold,
    )

    angle_deg = estimate_angle_from_phase(f1a, f2a)
    timestamp = float(frame["timestamp"])

    detections = []
    for hit in cfar_hits:
        velocity = bin_to_velocity_kmh(hit.bin_index, fft_size, max_speed_kmh)
        detections.append(
            RADCDetection(
                frame_index=frame_index,
                timestamp=timestamp,
                distance_m=0.0,  # RADC gives velocity, not range — set from FMCW chirp later
                velocity_kmh=velocity,
                angle_deg=angle_deg,
                magnitude=hit.magnitude,
                snr_db=hit.snr_db,
                bin_index=hit.bin_index,
            )
        )

    return detections


def compare_radc_vs_pdat(
    radc_detections: list[RADCDetection],
    pdat: list[dict],
) -> dict:
    """Compare our RADC FFT detections against the module's PDAT output.

    Returns a summary dict for logging / CSV export.
    """
    pdat_speeds = [abs(p.get("speed", 0)) for p in pdat if p]
    pdat_mags = [p.get("magnitude", 0) for p in pdat if p]
    radc_velocities = [abs(d.velocity_kmh) for d in radc_detections]
    radc_mags = [d.magnitude for d in radc_detections]

    return {
        "radc_count": len(radc_detections),
        "pdat_count": len(pdat),
        "radc_max_velocity_kmh": max(radc_velocities) if radc_velocities else 0.0,
        "pdat_max_speed_kmh": max(pdat_speeds) if pdat_speeds else 0.0,
        "radc_max_magnitude": max(radc_mags) if radc_mags else 0.0,
        "pdat_max_magnitude": max(pdat_mags) if pdat_mags else 0.0,
    }
