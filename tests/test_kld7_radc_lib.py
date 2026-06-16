"""Tests for K-LD7 raw ADC processing library."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Functions available in the package — prefer importing from there
try:
    from openflight.kld7.radc import (
        ball_bin_range_from_speed,
        bin_to_velocity_kmh,
        cfar_detect,
        circular_bin_distance,
        compute_spectrum,
        expected_ball_bin_from_speed,
        extract_launch_angle,
        find_impact_frames,
        parse_radc_payload,
        radc_capture_diagnostics,
        radc_frame_diagnostics,
        select_best_shot_result,
        to_complex_iq,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))
    from kld7_radc_lib import (
        ball_bin_range_from_speed,
        bin_to_velocity_kmh,
        cfar_detect,
        circular_bin_distance,
        compute_spectrum,
        expected_ball_bin_from_speed,
        extract_launch_angle,
        find_impact_frames,
        parse_radc_payload,
        radc_capture_diagnostics,
        radc_frame_diagnostics,
        select_best_shot_result,
        to_complex_iq,
    )

# Functions that remain in scripts only
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))
from kld7_radc_lib import (
    ADC_MIDPOINT,
    RADCDetection,
    compare_radc_vs_pdat,
    estimate_angle_from_phase,
    process_radc_frame,
)


def test_select_best_shot_result_prefers_geometry_over_later_naive_group():
    results = [
        {
            "launch_angle_deg": 22.9,
            "impact_frames": [38, 40],
            "estimator": "geometry",
            "frame_count": 2,
            "confidence": 0.90,
            "selection_path": "geometry_primary",
        },
        {
            "launch_angle_deg": 10.6,
            "impact_frames": [44, 48],
            "estimator": "naive",
            "frame_count": 4,
            "confidence": 0.93,
            "selection_path": "legacy_naive_suspect",
        },
    ]

    assert select_best_shot_result(results)["launch_angle_deg"] == 22.9


def test_select_best_shot_result_prefers_single_frame_geometry_over_later_naive_group():
    results = [
        {
            "launch_angle_deg": 17.1,
            "impact_frames": [39, 40],
            "estimator": "geometry_single_frame",
            "frame_count": 1,
            "confidence": 0.72,
            "selection_path": "geometry_single_frame",
        },
        {
            "launch_angle_deg": 26.2,
            "impact_frames": [47, 50],
            "estimator": "naive",
            "frame_count": 3,
            "confidence": 0.54,
            "selection_path": "legacy_naive_suspect",
        },
    ]

    assert select_best_shot_result(results)["launch_angle_deg"] == 17.1


def test_select_best_shot_result_uses_latest_group_as_same_tier_tiebreaker():
    results = [
        {
            "launch_angle_deg": 14.0,
            "impact_frames": [35, 38],
            "estimator": "geometry_single_frame",
            "frame_count": 1,
            "confidence": 0.72,
            "selection_path": "geometry_single_frame",
        },
        {
            "launch_angle_deg": 18.5,
            "impact_frames": [38, 40],
            "estimator": "geometry_single_frame",
            "frame_count": 1,
            "confidence": 0.72,
            "selection_path": "geometry_single_frame",
        },
    ]

    assert select_best_shot_result(results)["launch_angle_deg"] == 18.5


def test_select_best_shot_result_rejects_empty_results():
    with pytest.raises(ValueError, match="at least one result"):
        select_best_shot_result([])


class TestParseRadcPayload:
    def test_parses_3072_bytes_into_six_channels(self):
        """RADC payload should split into 6 arrays of 256 uint16 samples."""
        # Create synthetic 3072-byte payload: 6 segments of 256 uint16 values
        payload = b""
        for seg in range(6):
            payload += np.arange(seg * 256, (seg + 1) * 256, dtype=np.uint16).tobytes()

        result = parse_radc_payload(payload)

        assert result["f1a_i"].shape == (256,)
        assert result["f1a_q"].shape == (256,)
        assert result["f2a_i"].shape == (256,)
        assert result["f2a_q"].shape == (256,)
        assert result["f1b_i"].shape == (256,)
        assert result["f1b_q"].shape == (256,)
        assert result["f1a_i"].dtype == np.uint16
        # Verify first segment starts at 0
        assert result["f1a_i"][0] == 0
        assert result["f1a_i"][255] == 255
        # Verify second segment starts at 256
        assert result["f1a_q"][0] == 256

    def test_rejects_wrong_payload_size(self):
        """Payloads that aren't 3072 bytes should raise ValueError."""
        with pytest.raises(ValueError, match="3072"):
            parse_radc_payload(b"\x00" * 1024)

    def test_to_complex_iq(self):
        """Should convert uint16 I/Q pairs to complex float with mean removal."""
        # Create I channel with a signal: ramp from 32768 to 32768+255
        i_vals = np.arange(32768, 32768 + 256, dtype=np.uint16)
        # Q channel constant
        q_vals = np.full(256, 33000, dtype=np.uint16)

        f1a = to_complex_iq(i_vals, q_vals)

        assert f1a.dtype == np.complex128
        assert f1a.shape == (256,)
        # Mean-removed: I should center around 0 with spread ~±128
        assert np.abs(np.mean(f1a.real)) < 1.0  # mean is ~0 after removal
        # Q is constant, so mean removal makes all values ~0
        assert np.abs(f1a[0].imag) < 1.0


class TestComputeSpectrum:
    def test_returns_magnitude_spectrum_of_correct_size(self):
        """FFT should produce magnitude spectrum with fft_size bins."""
        iq = np.random.randn(256) + 1j * np.random.randn(256)
        spectrum = compute_spectrum(iq, fft_size=2048)
        assert spectrum.shape == (2048,)
        assert spectrum.dtype == np.float64

    def test_detects_injected_tone(self):
        """A pure tone at a known bin should produce a clear peak."""
        n = 256
        fft_size = 2048
        bin_target = 100  # put energy at bin 100
        freq = bin_target / fft_size
        t = np.arange(n)
        iq = np.exp(2j * np.pi * freq * t)

        spectrum = compute_spectrum(iq, fft_size=fft_size)

        peak_bin = np.argmax(spectrum)
        # Peak should be at or very near the target bin
        assert abs(peak_bin - bin_target) <= 1

    def test_zero_padding_increases_resolution(self):
        """Larger FFT size should produce more bins."""
        iq = np.random.randn(256) + 1j * np.random.randn(256)
        spec_small = compute_spectrum(iq, fft_size=512)
        spec_large = compute_spectrum(iq, fft_size=4096)
        assert spec_small.shape == (512,)
        assert spec_large.shape == (4096,)


class TestCFARDetect:
    def test_detects_tone_above_noise(self):
        """A clear tone in noise should produce exactly one detection."""
        fft_size = 2048
        # Noise floor
        spectrum = np.random.exponential(1.0, size=fft_size)
        # Inject a strong tone at bin 500
        spectrum[500] = 200.0

        detections = cfar_detect(spectrum, guard_cells=4, training_cells=16, threshold_factor=8.0)

        assert len(detections) >= 1
        bins = [d.bin_index for d in detections]
        assert 500 in bins

    def test_no_detections_in_pure_noise(self):
        """Uniform noise should produce very few or zero false detections."""
        fft_size = 2048
        np.random.seed(42)
        spectrum = np.random.exponential(1.0, size=fft_size)

        detections = cfar_detect(spectrum, guard_cells=4, training_cells=16, threshold_factor=12.0)

        # With high threshold, noise should produce very few false alarms
        assert len(detections) <= 5

    def test_detection_has_required_fields(self):
        """Each detection should carry bin index, magnitude, and SNR."""
        spectrum = np.ones(2048)
        spectrum[300] = 100.0

        detections = cfar_detect(spectrum, guard_cells=4, training_cells=16, threshold_factor=8.0)

        assert len(detections) >= 1
        d = detections[0]
        assert hasattr(d, "bin_index")
        assert hasattr(d, "magnitude")
        assert hasattr(d, "snr_db")
        assert d.snr_db > 0


class TestBinToPhysical:
    def test_zero_bin_is_zero_velocity(self):
        """DC bin should map to zero velocity."""
        v = bin_to_velocity_kmh(0, fft_size=2048, max_speed_kmh=100.0)
        assert v == pytest.approx(0.0, abs=0.1)

    def test_velocity_scales_linearly(self):
        """Bins should map linearly to velocity up to max_speed."""
        fft_size = 2048
        max_speed = 100.0
        v_quarter = bin_to_velocity_kmh(fft_size // 4, fft_size=fft_size, max_speed_kmh=max_speed)
        v_half = bin_to_velocity_kmh(fft_size // 2, fft_size=fft_size, max_speed_kmh=max_speed)
        # fft_size//4 is half of fft_size//2 (the Nyquist bin), so maps to max_speed/2
        assert v_quarter == pytest.approx(max_speed / 2, abs=1.0)
        assert v_half == pytest.approx(max_speed, abs=1.0)

    def test_negative_velocity_for_upper_bins(self):
        """Upper half of FFT bins represent negative (inbound) velocity."""
        fft_size = 2048
        max_speed = 100.0
        v = bin_to_velocity_kmh(fft_size - 10, fft_size=fft_size, max_speed_kmh=max_speed)
        assert v < 0


class TestBallBinRangeFromSpeed:
    """Regression tests for the OPS-anchored ball-band helper.

    The K-LD7 has an unambiguous-velocity range of ±100 km/h. Real ball
    speeds (95-180 mph ≈ 153-290 km/h) alias into this range. When the
    aliased velocity is close to the wraparound boundary at ±0 km/h
    (e.g., 118.9 mph aliases to -8.7 km/h), a search window of
    ±tolerance around it straddles the boundary. The band must be
    represented as TWO ranges that wrap around DC, not one giant range
    spanning most of the spectrum.

    Failure mode (observed on session_20260501_183749_range.jsonl):
    shot 3 at 118.9 mph produced no horizontal angle because the helper
    returned [75, 1794] (86% of the spectrum, i.e. the COMPLEMENT of
    the actual ball band).
    """

    def test_normal_aliased_band_returns_single_range(self):
        """A 100 mph ball aliases to -39 km/h, well clear of the
        wraparound boundary. The band must be a single (lo, hi) range
        with hi-lo small (≈329 bins for ±10 mph tolerance).
        """
        ranges = ball_bin_range_from_speed(
            ball_speed_mph=100.0,
            tolerance_mph=10.0,
            fft_size=2048,
            max_speed_kmh=100.0,
        )
        assert isinstance(ranges, list), (
            "ball_bin_range_from_speed must return list[tuple[int,int]]"
        )
        assert len(ranges) == 1, f"Non-wrapping case must return one range, got {ranges}"
        lo, hi = ranges[0]
        assert lo < hi, "Single range must have lo < hi"
        width = hi - lo
        assert 200 < width < 500, (
            f"Single ±10 mph band ≈ ±32 km/h × 10.24 bins/km/h ≈ 330 bins, got {width}"
        )

    def test_wraparound_band_returns_two_ranges(self):
        """118.9 mph aliases to -8.7 km/h. With ±10 mph (≈±16 km/h)
        tolerance, the band straddles 0 km/h and must be represented
        as TWO ranges: [0, +tol_bin] and [N-|tol_bin|, N].
        """
        ranges = ball_bin_range_from_speed(
            ball_speed_mph=118.9,
            tolerance_mph=10.0,
            fft_size=2048,
            max_speed_kmh=100.0,
        )
        assert isinstance(ranges, list)
        assert len(ranges) == 2, f"Wrap-around case must return two ranges, got {ranges}"
        # Each sub-range must be a valid forward range.
        for lo, hi in ranges:
            assert 0 <= lo < hi <= 2048, (
                f"Sub-range ({lo}, {hi}) must be within FFT bounds and lo<hi"
            )
        # Total width should match a normal ±10 mph band (≈330 bins),
        # NOT the buggy ≈1719-bin spectrum-spanning band.
        total_width = sum(hi - lo for lo, hi in ranges)
        assert total_width < 500, (
            f"Wrap-around band must span ≈330 bins (the actual ball "
            f"location), not {total_width} (which is the COMPLEMENT)"
        )

    def test_wraparound_band_actually_contains_expected_bin(self):
        """The wrap-around bands must include the bin where the ball
        actually appears in the spectrum.
        """
        from openflight.kld7.radc import _velocity_to_bin

        ball_speed_mph = 118.9
        # Where the ball actually peaks
        ball_kmh = ball_speed_mph * 1.609
        aliased = ball_kmh % 200.0
        if aliased > 100.0:
            aliased -= 200.0
        expected_bin = _velocity_to_bin(aliased, 2048, 100.0)

        ranges = ball_bin_range_from_speed(
            ball_speed_mph=ball_speed_mph,
            tolerance_mph=10.0,
            fft_size=2048,
            max_speed_kmh=100.0,
        )
        in_some_range = any(lo <= expected_bin < hi for lo, hi in ranges)
        assert in_some_range, (
            f"Expected bin {expected_bin} (for {ball_speed_mph} mph) "
            f"must be inside one of the returned ranges {ranges}"
        )

    def test_extract_launch_angle_finds_ball_when_band_wraps(self):
        """End-to-end regression: extract_launch_angle must lock onto
        the OPS-anchored ball location (118.9 mph) rather than picking
        an arbitrary peak in the buggy ≈86%-of-spectrum band.

        The key assertion is that the recovered ball_speed_mph is
        close to the OPS-supplied 118.9 mph — proving the algorithm
        searched the correct (wrap-around) band.
        """
        from openflight.kld7.radc import _velocity_to_bin

        ball_speed_mph = 118.9
        ball_kmh = ball_speed_mph * 1.609
        aliased = ball_kmh % 200.0
        if aliased > 100.0:
            aliased -= 200.0
        target_bin = _velocity_to_bin(aliased, 2048, 100.0)

        # Build an IQ signal whose 2048-bin FFT peaks at target_bin.
        # A cleanly-windowed cisoid avoids leakage into other bins,
        # so the algorithm has to search the *correct* wrap-around
        # band to find this peak.
        n = 256
        digital_freq = target_bin / 2048.0
        if digital_freq > 0.5:
            digital_freq -= 1.0
        t = np.arange(n)
        amplitude = 200
        # Apply a Hann window to suppress sidelobes — without it the
        # rectangular-window leakage from a near-DC peak spans a huge
        # fraction of the spectrum, masking the band-search bug.
        win = np.hanning(n)
        f1a = amplitude * np.exp(2j * np.pi * digital_freq * t) * win
        # Add a competing weaker peak elsewhere in the spectrum to
        # ensure the algorithm picks the band-anchored one and not
        # whichever is loudest globally.
        decoy_freq = 200 / 2048.0
        f1a += 0.6 * amplitude * np.exp(2j * np.pi * decoy_freq * t) * win
        f2a = np.exp(1j * np.deg2rad(6.0) * 2.0) * f1a

        i1 = np.real(f1a) + 2048
        q1 = np.imag(f1a) + 2048
        i2 = np.real(f2a) + 2048
        q2 = np.imag(f2a) + 2048

        frame = {
            "timestamp": 0.0,
            "radc": {
                "f1a_i": i1.astype(np.float64),
                "f1a_q": q1.astype(np.float64),
                "f2a_i": i2.astype(np.float64),
                "f2a_q": q2.astype(np.float64),
            },
        }
        frames = [frame] * 5

        results = extract_launch_angle(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=10.0,
            impact_energy_threshold=0.0,
        )
        assert results, (
            "extract_launch_angle returned no results despite a strong "
            "ball signal at the aliased bin"
        )
        # The peak_bin for this shot must be within ±10 bins of the
        # true target bin (1959 for 118.9 mph). In the buggy regime
        # the algorithm picks something inside [75, 1794] (≈ bin 1300
        # giving recovered ≈ 136 mph), which is more than 10 bins from
        # the true target. After the fix the band wraps around DC and
        # the real peak at bin 1959 wins.
        # Re-extract the per-frame peak bins from a quick rescan of
        # the test frame. We reach into the algorithm by checking the
        # recovered ball_speed_mph vs the OPS-anchored expected value.
        recovered = results[0]["ball_speed_mph"]
        # Tight tolerance: ±5 mph from 118.9. The buggy implementation
        # currently produces ≈136 mph (peaks at the decoy/clutter side
        # of the spurious band), so this assertion fails today.
        assert abs(recovered - ball_speed_mph) < 5, (
            f"Expected ball_speed within ±5 mph of OPS-supplied "
            f"{ball_speed_mph} mph, got {recovered} — algorithm picked "
            f"a peak outside the wrap-around band, indicating the "
            f"band-wrap bug is still present"
        )


class TestAngleEstimation:
    def test_zero_phase_difference_gives_zero_angle(self):
        """Identical signals on both channels should give ~0 degrees."""
        n = 256
        signal = np.exp(2j * np.pi * 0.1 * np.arange(n))
        angle = estimate_angle_from_phase(signal, signal)
        assert abs(angle) < 2.0

    def test_known_phase_offset_gives_nonzero_angle(self):
        """A deliberate phase shift between channels should produce a measurable angle."""
        n = 256
        signal = np.exp(2j * np.pi * 0.1 * np.arange(n))
        shifted = signal * np.exp(1j * np.pi / 6)  # 30 degree phase shift
        angle = estimate_angle_from_phase(signal, shifted)
        assert abs(angle) > 5.0


class TestProcessRadcFrame:
    def _make_frame(self, tone_bin=100):
        """Create a synthetic RADC frame with a tone at a known bin."""
        n = 256
        fft_size = 2048
        freq = tone_bin / fft_size
        t = np.arange(n)
        signal = 5000.0 * np.exp(2j * np.pi * freq * t)
        noise = np.random.randn(n) + 1j * np.random.randn(n)
        iq = signal + noise

        i_vals = (iq.real + ADC_MIDPOINT).astype(np.uint16)
        q_vals = (iq.imag + ADC_MIDPOINT).astype(np.uint16)
        # Pack into 3072-byte payload (put signal in F1A, zeros elsewhere)
        payload = bytearray(3072)
        payload[0:512] = i_vals.tobytes()
        payload[512:1024] = q_vals.tobytes()

        return {
            "timestamp": 1000.0,
            "radc": bytes(payload),
        }

    def test_returns_detections_for_frame_with_tone(self):
        """A frame with an injected tone should produce at least one detection."""
        frame = self._make_frame(tone_bin=100)
        detections = process_radc_frame(
            frame,
            frame_index=0,
            fft_size=2048,
            max_speed_kmh=100.0,
            cfar_guard=32,
            cfar_training=32,
        )
        assert len(detections) >= 1
        assert all(isinstance(d, RADCDetection) for d in detections)

    def test_returns_empty_for_noise_only_frame(self):
        """A frame with only noise should produce few or no detections."""
        np.random.seed(42)
        noise_i = np.random.randint(30000, 35000, size=256, dtype=np.uint16)
        noise_q = np.random.randint(30000, 35000, size=256, dtype=np.uint16)
        payload = bytearray(3072)
        payload[0:512] = noise_i.tobytes()
        payload[512:1024] = noise_q.tobytes()
        frame = {"timestamp": 1000.0, "radc": bytes(payload)}
        detections = process_radc_frame(
            frame,
            frame_index=0,
            fft_size=2048,
            max_speed_kmh=100.0,
            cfar_threshold=12.0,
        )
        assert len(detections) <= 3


class TestCompareRadcVsPdat:
    def test_counts_radc_and_pdat_detections(self):
        """Comparison should report counts from both sources."""
        radc_detections = [
            RADCDetection(0, 1.0, 0.0, 50.0, 10.0, 100.0, 15.0, 500),
            RADCDetection(0, 1.0, 0.0, 30.0, 8.0, 80.0, 12.0, 300),
        ]
        pdat = [
            {"distance": 4.2, "speed": 25.0, "angle": 10.0, "magnitude": 2500},
        ]

        result = compare_radc_vs_pdat(radc_detections, pdat)

        assert result["radc_count"] == 2
        assert result["pdat_count"] == 1

    def test_handles_empty_inputs(self):
        """Should handle cases where one or both sources have no detections."""
        result = compare_radc_vs_pdat([], [])
        assert result["radc_count"] == 0
        assert result["pdat_count"] == 0


class TestAngleBoundsValidation:
    """Tests for hard angle bounds in extract_launch_angle."""

    def test_orientation_parameter_accepted(self):
        """extract_launch_angle should accept orientation parameter."""
        # Empty frames — just verify the parameter doesn't crash
        results = extract_launch_angle([], orientation="vertical")
        assert results == []

    def test_orientation_none_is_default(self):
        """Default orientation=None should not filter (backward compat)."""
        results = extract_launch_angle([])
        assert results == []

    def test_horizontal_orientation_accepted(self):
        """extract_launch_angle should accept horizontal orientation."""
        results = extract_launch_angle([], orientation="horizontal")
        assert results == []

    def test_horizontal_angle_limit_parameter_accepted(self):
        """extract_launch_angle should accept a tunable horizontal bound."""
        results = extract_launch_angle(
            [],
            orientation="horizontal",
            horizontal_angle_limit_deg=30.0,
        )
        assert results == []


class TestOpsBinSoftAnchor:
    """Tests for the OPS-expected-bin soft penalty in extract_launch_angle.

    The soft anchor reduces the SNR^2 weight of frames whose peak bin is
    far from the OPS-expected bin (clutter stripes), so the final
    weighted angle is dominated by the frame whose peak bin actually
    lines up with the OPS-measured ball speed.
    """

    # K-LD7 antenna geometry (must match radc.py)
    _D = 8.0e-3
    _LAMBDA = 3e8 / 24.125e9

    @classmethod
    def _phase_for_angle(cls, angle_deg: float) -> float:
        """Inverse of per_bin_angle_deg: returns Δφ in radians."""
        sin_theta = np.sin(np.radians(angle_deg))
        return float(2.0 * np.pi * cls._D * sin_theta / cls._LAMBDA)

    @staticmethod
    def _pack_payload(
        f1a_iq: np.ndarray,
        f2a_iq: np.ndarray,
    ) -> bytes:
        """Pack complex I/Q into a 3072-byte RADC payload."""
        payload = bytearray(3072)
        for slot, iq in ((0, f1a_iq), (2, f2a_iq)):
            i_vals = (iq.real + ADC_MIDPOINT).astype(np.uint16)
            q_vals = (iq.imag + ADC_MIDPOINT).astype(np.uint16)
            payload[slot * 512 : (slot + 1) * 512] = i_vals.tobytes()
            payload[(slot + 1) * 512 : (slot + 2) * 512] = q_vals.tobytes()
        return bytes(payload)

    @classmethod
    def _make_frame_at_bin(
        cls,
        peak_bin: int,
        angle_deg: float,
        amplitude: float = 6000.0,
        seed: int | None = 0,
        ts: float = 1000.0,
    ) -> dict:
        """Build a synthetic RADC frame whose F1A/F2A spectrum peaks at
        `peak_bin` and whose interferometric angle at that bin is
        approximately `angle_deg`."""
        n = 256
        fft_size = 2048
        # Tone is constructed for a 256-sample window; freq scales by FFT size
        freq = peak_bin / fft_size
        t = np.arange(n)
        carrier = np.exp(2j * np.pi * freq * t)
        signal = amplitude * carrier

        # Per-channel noise; F1A and F2A get independent noise but the
        # *signal* in F2A is phase-shifted relative to F1A.
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        noise1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 80.0
        noise2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 80.0

        delta_phi = cls._phase_for_angle(angle_deg)
        f1a_iq = signal + noise1
        f2a_iq = signal * np.exp(-1j * delta_phi) + noise2

        return {
            "timestamp": ts,
            "radc": cls._pack_payload(f1a_iq, f2a_iq),
        }

    @classmethod
    def _noise_frame(cls, ts: float, seed: int) -> dict:
        """A frame with very low-amplitude noise so the impact detector
        has a non-zero baseline to compare against, but the noise floor
        is whitened enough that no random bin in the ball band passes
        the SNR>=2 single-frame floor inside extract_launch_angle."""
        n = 256
        rng = np.random.default_rng(seed)
        # Hann-window of pure white noise has a flat magnitude spectrum
        # in expectation, so peak/median ratio is close to 1. Amplitude
        # itself doesn't change the peak/median ratio — only the energy
        # vs the impact threshold. Keep low and the impact detector
        # will not pick up these frames.
        noise1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 5.0
        noise2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 5.0
        return {
            "timestamp": ts,
            "radc": cls._pack_payload(noise1, noise2),
        }

    @classmethod
    def _impact_window(
        cls,
        center_frame: int,
        total: int,
        ball_frames: list[dict],
    ) -> list[dict]:
        """Build a list of `total` synthesized RADC frames. Most are
        low-energy noise (so the impact detector's median energy
        baseline is non-zero); positions starting at `center_frame`
        get the supplied ball/clutter frames."""
        out = [cls._noise_frame(ts=1000.0 + i * 0.056, seed=900 + i) for i in range(total)]
        for offset, frame in enumerate(ball_frames):
            idx = center_frame + offset
            if 0 <= idx < total:
                frame["timestamp"] = 1000.0 + idx * 0.056
                out[idx] = frame
        return out

    @classmethod
    def _two_strong_frames(
        cls,
        ops_bin: int,
        anchor_offset: int,
        anchor_angle: float,
        outlier_offset: int,
        outlier_angle: float,
    ) -> list[dict]:
        """Build a small frame list where every frame either lands at
        bin (ops_bin + anchor_offset) with `anchor_angle`, or at bin
        (ops_bin + outlier_offset) with `outlier_angle`. There are no
        noise filler frames, so the per-frame loop sees exactly the
        intended 2 detections after impact-window expansion. With
        len(angs) == 2 the outlier-rejection step is skipped (it only
        activates when len(angs) >= 3), so the average reflects the
        soft-anchor weights cleanly."""
        a = cls._make_frame_at_bin(
            peak_bin=ops_bin + anchor_offset,
            angle_deg=anchor_angle,
            amplitude=8000.0,
            seed=101,
        )
        b = cls._make_frame_at_bin(
            peak_bin=ops_bin + outlier_offset,
            angle_deg=outlier_angle,
            amplitude=8000.0,
            seed=202,
        )
        # Place the two strong frames adjacent so the impact detector
        # groups them, and pad the rest with clones of the same frames
        # so impact-window expansion can't pull in random-noise frames.
        # 4 frames keeps things minimal but lets the (-1, +3) expansion
        # only see clones of these two, not random noise.
        frames = [
            dict(a, timestamp=1.0),
            dict(b, timestamp=2.0),
            dict(a, timestamp=3.0),
            dict(b, timestamp=4.0),
        ]
        return frames

    def test_far_outlier_frame_is_downweighted(self):
        """Two strong frames, equal SNR, both inside the velocity band —
        the one near the OPS bin should dominate when the penalty is on.

        The frame furthest from the median angle is dropped by the
        existing outlier-rejection step. With the soft anchor on, the
        surviving outlier frame still gets its weight reduced, so the
        result is pulled clearly toward the OPS-anchored angle.
        """
        ops_speed_mph = 70.0
        ops_bin = 1163
        speed_tol = 40.0

        # Use _two_strong_frames so the per-frame loop sees only the
        # ball + clutter detections (and their clones) — no noise.
        # Ball at OPS bin, +5°. Clutter +60 bins away, -25°.
        frames = self._two_strong_frames(
            ops_bin=ops_bin,
            anchor_offset=0,
            anchor_angle=5.0,
            outlier_offset=60,
            outlier_angle=-25.0,
        )

        # impact_energy_threshold relative to median energy: with a
        # window full of strong frames the median ~ peak, so we lower
        # the gate to ensure detection. (Real captures have a clear
        # impact spike against a low-energy baseline.)
        impact_threshold = 0.5

        # PENALTY ON
        results_on = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=ops_speed_mph,
            speed_tolerance_mph=speed_tol,
            impact_energy_threshold=impact_threshold,
            orientation=None,
            ops_bin_outlier_tol=25,
            ops_bin_outlier_penalty=10.0,
        )
        # PENALTY OFF
        results_off = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=ops_speed_mph,
            speed_tolerance_mph=speed_tol,
            impact_energy_threshold=impact_threshold,
            orientation=None,
            ops_bin_outlier_tol=25,
            ops_bin_outlier_penalty=1.0,
        )
        assert results_on and results_off
        on = results_on[0]["launch_angle_deg"]
        off = results_off[0]["launch_angle_deg"]
        # The penalty must move the answer toward +5 (anchor angle)
        # relative to the un-penalized baseline.
        assert on > off, f"penalty should pull angle toward anchor; on={on:+.1f} off={off:+.1f}"
        assert abs(on - 5.0) < abs(off - 5.0), (
            f"penalty result should be closer to +5 than baseline; on={on:+.1f} off={off:+.1f}"
        )

    def test_no_penalty_when_ball_speed_unknown(self):
        """ops243_ball_speed_mph=None must keep the legacy code path —
        the penalty is a no-op (no anchor available). Both frames at
        the same angle must produce that angle as the result."""
        # Both peaks inside the broad default range (-39 to -7 km/h
        # => bins ~1250 to ~1857) so they survive the band filter
        # without an OPS-anchored band.
        frames = self._two_strong_frames(
            ops_bin=1300,
            anchor_offset=0,
            anchor_angle=10.0,
            outlier_offset=500,
            outlier_angle=10.0,
        )

        results = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=None,
            impact_energy_threshold=0.5,
            orientation=None,
        )
        assert results, "expected at least one detected shot"
        # Both frames at +10° -> result must be near +10°
        assert abs(results[0]["launch_angle_deg"] - 10.0) < 5.0

    def test_soft_anchor_layered_on_top_of_band_filter(self):
        """Strong peaks well outside the OPS-anchored band do NOT make
        it through to the SNR^2 average. The anchor penalty only acts on
        frames that already survived the velocity band filter; this
        layering is what keeps the change safe (we never pull a clutter
        frame in that the existing band filter would have rejected)."""
        ops_speed_mph = 70.0
        ops_bin = 1163

        # Strong tones placed 600 bins from OPS bin — far outside the
        # ±10 mph band [988, 1318]. The peak inside the band is just
        # noise, so the resulting frames have low SNR (~2-3) and any
        # angle the algorithm reports has no physical meaning.
        far_a = self._make_frame_at_bin(
            peak_bin=ops_bin + 600,
            angle_deg=-30.0,
            amplitude=8000.0,
            seed=55,
        )
        far_b = self._make_frame_at_bin(
            peak_bin=ops_bin + 610,
            angle_deg=-30.0,
            amplitude=8000.0,
            seed=66,
        )
        frames = [
            dict(far_a, timestamp=1.0),
            dict(far_b, timestamp=2.0),
            dict(far_a, timestamp=3.0),
            dict(far_b, timestamp=4.0),
        ]

        results = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=ops_speed_mph,
            speed_tolerance_mph=10.0,
            impact_energy_threshold=0.5,
            orientation=None,
        )
        # Either nothing detected, or whatever the algorithm latches
        # onto inside the narrow band has near-noise SNR — i.e. the
        # strong outside-band tone was already filtered. The injected
        # -30° angle must NOT propagate into the result.
        if results:
            assert abs(results[0]["launch_angle_deg"] - (-30.0)) > 5.0, (
                "strong out-of-band tone leaked into the angle result"
            )
            assert results[0]["avg_snr_db"] < 5.0, (
                "out-of-band peak should not produce a high-SNR shot"
            )

    def test_majority_outlier_emits_warning(self, caplog):
        """When ≥50% of frames trigger the OPS-bin penalty, the log
        must be emitted at WARNING level rather than INFO so it surfaces
        in production logs without a replay. Pattern observed in users'
        logs: 5/5, 8/8, 3/3 frames > 25 bins from expected = setup
        problem.
        """
        import logging

        ops_speed_mph = 70.0
        ops_bin = 1163

        # Both "strong" frames sit far outside the OPS-bin tolerance,
        # so >50% of surviving frames will be penalized.
        frames = self._two_strong_frames(
            ops_bin=ops_bin,
            anchor_offset=60,
            anchor_angle=-10.0,
            outlier_offset=80,
            outlier_angle=-12.0,
        )

        with caplog.at_level(logging.WARNING, logger="openflight.kld7.radc"):
            extract_launch_angle(
                frames=frames,
                ops243_ball_speed_mph=ops_speed_mph,
                speed_tolerance_mph=40.0,
                impact_energy_threshold=0.5,
                ops_bin_outlier_tol=25,
                ops_bin_outlier_penalty=10.0,
            )

        warns = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "OPS-bin penalty" in r.message
        ]
        assert warns, (
            "Expected a WARNING-level OPS-bin penalty log when ≥50% of "
            f"frames are outliers; got records: {caplog.records}"
        )
        # Message should include the actual peak bins for diagnosis
        assert "peak bins:" in warns[0].message
        # And reference the troubleshooting doc
        assert "troubleshooting" in warns[0].message

    def test_vertical_rule_stack_prefers_rising_previous_pair(self, caplog):
        """Vertical rule stack should select previous+anchor when the pair rises.

        Scenario:
          - frame at 23ms: weaker, in-bin, lower angle
          - frame at 57ms: strongest anchor, in-bin
          - frame at 91ms: also strong, but selected pair should still be
            previous->anchor according to the rule stack.
        """
        import logging

        ops_speed_mph = 108.0
        ops_bin = expected_ball_bin_from_speed(ops_speed_mph)
        impact_t = 1000.0

        frames = [
            self._noise_frame(ts=impact_t - 0.22, seed=301),
            self._noise_frame(ts=impact_t - 0.10, seed=302),
            self._make_frame_at_bin(
                peak_bin=ops_bin + 18,
                angle_deg=-12.0,
                amplitude=6800.0,
                seed=303,
                ts=impact_t + 0.023,
            ),
            self._make_frame_at_bin(
                peak_bin=ops_bin,
                angle_deg=-8.0,
                amplitude=9800.0,
                seed=304,
                ts=impact_t + 0.057,
            ),
            self._make_frame_at_bin(
                peak_bin=ops_bin,
                angle_deg=-6.0,
                amplitude=8600.0,
                seed=305,
                ts=impact_t + 0.091,
            ),
            self._noise_frame(ts=impact_t + 0.16, seed=306),
        ]

        with caplog.at_level(logging.INFO, logger="openflight.kld7.radc"):
            results = extract_launch_angle(
                frames=frames,
                ops243_ball_speed_mph=ops_speed_mph,
                speed_tolerance_mph=10.0,
                impact_energy_threshold=0.5,
                orientation="vertical",
                angle_offset_deg=18.0,
                impact_timestamp=impact_t,
            )

        assert results, "expected a vertical launch result from synthetic frames"
        assert results[0]["frame_count"] == 2
        assert any("Pair winner" in rec.message for rec in caplog.records), (
            "expected rule-stack logs to include the winning pair"
        )

    def test_high_rmse_geometry_pair_falls_back_to_earliest_single_frame(self):
        """A noisy second bearing should not keep a high-RMSE two-frame fit.

        This mirrors the indoor-screen failure mode where an early OPS-bin
        frame is a clean ball return, but the adjacent later frame has a
        plausible bin/SNR and an inconsistent bearing. The pair should be
        categorized as a capped-confidence single-frame geometry result.
        """
        from openflight.kld7.geometry import predicted_bearing_deg

        ops_speed_mph = 108.0
        ops_bin = expected_ball_bin_from_speed(ops_speed_mph)
        impact_t = 1000.0
        distance_ft = 5.0
        mount_deg = 10.0
        clean_launch_deg = 20.0
        clean_t = 0.028
        contaminated_t = 0.064

        clean_bearing = predicted_bearing_deg(
            clean_launch_deg,
            clean_t,
            ops_speed_mph,
            distance_ft,
            mount_deg,
        )

        frames = [
            self._noise_frame(ts=impact_t - 0.20, seed=401),
            self._noise_frame(ts=impact_t - 0.10, seed=402),
            self._make_frame_at_bin(
                peak_bin=ops_bin,
                angle_deg=clean_bearing,
                amplitude=8000.0,
                seed=403,
                ts=impact_t + clean_t,
            ),
            self._make_frame_at_bin(
                peak_bin=ops_bin,
                angle_deg=16.0,
                amplitude=14000.0,
                seed=404,
                ts=impact_t + contaminated_t,
            ),
            self._noise_frame(ts=impact_t + 0.16, seed=405),
        ]

        results = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=ops_speed_mph,
            speed_tolerance_mph=10.0,
            impact_energy_threshold=0.5,
            orientation="vertical",
            vertical_estimator="geometry",
            impact_timestamp=impact_t,
            mount_deg=mount_deg,
            distance_ft=distance_ft,
        )

        assert results
        best = results[0]
        assert best["estimator"] == "geometry_single_frame"
        assert best["selection_path"] == "geometry_single_frame"
        assert best["frame_count"] == 1
        assert best["selected_frame_indices"] == [2]
        assert best["geom_fit_rmse_deg"] is None
        assert best["geom_single_frame_resid_deg"] is not None
        assert best["confidence"] <= 0.72
        assert best["launch_angle_deg"] == pytest.approx(clean_launch_deg, abs=1.0)


class TestMultiBinCentroidAngle:
    """Tests for the magnitude²-weighted centroid angle aggregation
    across the per-frame spectral peak.

    The change replaces single-peak-bin angle extraction with a
    centroid across all bins above `centroid_floor_frac * peak`.
    For a clean single-bin tone, both should produce the same answer.
    For a range-spread target (or a noisy peak), the centroid should
    pull toward the bulk of the energy and be less sensitive to a
    single wild bin.
    """

    @staticmethod
    def _pack_two_channel_payload(f1a_iq: np.ndarray, f2a_iq: np.ndarray) -> bytes:
        payload = bytearray(3072)
        for slot, iq in ((0, f1a_iq), (2, f2a_iq)):
            i_vals = (iq.real + ADC_MIDPOINT).astype(np.uint16)
            q_vals = (iq.imag + ADC_MIDPOINT).astype(np.uint16)
            payload[slot * 512 : (slot + 1) * 512] = i_vals.tobytes()
            payload[(slot + 1) * 512 : (slot + 2) * 512] = q_vals.tobytes()
        return bytes(payload)

    @staticmethod
    def _phase_for_angle(angle_deg: float) -> float:
        d = 8.0e-3
        wavelength = 3e8 / 24.125e9
        return float(2.0 * np.pi * d * np.sin(np.radians(angle_deg)) / wavelength)

    @classmethod
    def _make_clean_tone_frame(
        cls,
        peak_bin: int,
        angle_deg: float,
        seed: int,
    ) -> dict:
        """Single-tone, very low noise. The peak bin will dominate
        such that the half-power window contains effectively only the
        peak (and its immediate spectral neighbors)."""
        n = 256
        fft_size = 2048
        freq = peak_bin / fft_size
        t = np.arange(n)
        carrier = np.exp(2j * np.pi * freq * t)
        signal = 8000.0 * carrier
        rng = np.random.default_rng(seed)
        noise1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 5.0
        noise2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 5.0
        delta = cls._phase_for_angle(angle_deg)
        return {
            "timestamp": 1000.0,
            "radc": cls._pack_two_channel_payload(
                signal + noise1,
                signal * np.exp(-1j * delta) + noise2,
            ),
        }

    @classmethod
    def _make_spread_target_frame(
        cls,
        peak_bin: int,
        angle_deg: float,
        spread_bins: int,
        seed: int,
    ) -> dict:
        """Wide-shoulder target: the same angle present at multiple
        adjacent bins (simulating a range-spread / Doppler-spread ball).
        With magnitude²-weighted centroid the answer should still be
        close to angle_deg even if the *peak* bin's individual phase
        is jittered by noise."""
        n = 256
        fft_size = 2048
        t = np.arange(n)
        delta = cls._phase_for_angle(angle_deg)
        # Sum tones at peak_bin-1, peak_bin, peak_bin+1, peak_bin+2
        # with descending amplitudes — produces a broad spectral lobe
        # all carrying the same relative phase.
        f1 = np.zeros(n, dtype=np.complex128)
        f2 = np.zeros(n, dtype=np.complex128)
        for offset, amp in zip(range(-1, 3), (5000.0, 8000.0, 6500.0, 4000.0)):
            freq = (peak_bin + offset) / fft_size
            carrier = amp * np.exp(2j * np.pi * freq * t)
            f1 += carrier
            f2 += carrier * np.exp(-1j * delta)
        rng = np.random.default_rng(seed)
        f1 += (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 80.0
        f2 += (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 80.0
        return {
            "timestamp": 1000.0,
            "radc": cls._pack_two_channel_payload(f1, f2),
        }

    def test_clean_tone_centroid_matches_peak_bin(self):
        """For a clean single-tone target the centroid should reproduce
        the peak-bin angle to within ~1 degree."""
        ops_bin = 1163
        a = self._make_clean_tone_frame(ops_bin, angle_deg=8.0, seed=11)
        b = self._make_clean_tone_frame(ops_bin, angle_deg=8.0, seed=22)
        frames = [
            dict(a, timestamp=1.0),
            dict(b, timestamp=2.0),
            dict(a, timestamp=3.0),
            dict(b, timestamp=4.0),
        ]
        results_centroid = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=70.0,
            speed_tolerance_mph=40.0,
            impact_energy_threshold=0.5,
            orientation=None,
            centroid_floor_frac=0.5,
        )
        results_peak = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=70.0,
            speed_tolerance_mph=40.0,
            impact_energy_threshold=0.5,
            orientation=None,
            centroid_floor_frac=1.0,  # legacy single-bin
        )
        assert results_centroid and results_peak
        ang_centroid = results_centroid[0]["launch_angle_deg"]
        ang_peak = results_peak[0]["launch_angle_deg"]
        assert abs(ang_centroid - ang_peak) < 1.5, (
            f"clean tone: centroid={ang_centroid:+.2f} "
            f"peak={ang_peak:+.2f} (should be near-identical)"
        )
        # And both should be near the injected +8°
        assert abs(ang_centroid - 8.0) < 2.0, f"clean tone: expected ~+8°, got {ang_centroid:+.2f}"

    def test_spread_target_centroid_more_robust_than_peak_bin(self):
        """For a range-spread target the centroid should be at least as
        accurate as peak-bin extraction. The spread is symmetric around
        the central bin so both should converge to the injected angle.
        This documents that the multi-bin centroid is no worse than the
        legacy peak-bin path on the spread case."""
        ops_bin = 1163
        a = self._make_spread_target_frame(
            ops_bin,
            angle_deg=10.0,
            spread_bins=4,
            seed=101,
        )
        b = self._make_spread_target_frame(
            ops_bin,
            angle_deg=10.0,
            spread_bins=4,
            seed=202,
        )
        frames = [
            dict(a, timestamp=1.0),
            dict(b, timestamp=2.0),
            dict(a, timestamp=3.0),
            dict(b, timestamp=4.0),
        ]
        results = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=70.0,
            speed_tolerance_mph=40.0,
            impact_energy_threshold=0.5,
            orientation=None,
            centroid_floor_frac=0.5,
        )
        assert results
        ang = results[0]["launch_angle_deg"]
        assert abs(ang - 10.0) < 3.0, f"spread target: expected ~+10°, centroid got {ang:+.2f}"

    def test_centroid_floor_one_reverts_to_legacy(self):
        """Setting centroid_floor_frac=1.0 must reproduce the legacy
        single-peak-bin behavior exactly."""
        ops_bin = 1163
        # Use a clean tone where peak-bin angle has a deterministic value.
        f = self._make_clean_tone_frame(ops_bin, angle_deg=4.0, seed=33)
        frames = [
            dict(f, timestamp=1.0),
            dict(f, timestamp=2.0),
            dict(f, timestamp=3.0),
            dict(f, timestamp=4.0),
        ]
        legacy = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=70.0,
            speed_tolerance_mph=40.0,
            impact_energy_threshold=0.5,
            orientation=None,
            centroid_floor_frac=1.0,
        )
        # Same call again — must be deterministic
        legacy2 = extract_launch_angle(
            frames=frames,
            ops243_ball_speed_mph=70.0,
            speed_tolerance_mph=40.0,
            impact_energy_threshold=0.5,
            orientation=None,
            centroid_floor_frac=1.0,
        )
        assert legacy and legacy2
        assert legacy[0]["launch_angle_deg"] == legacy2[0]["launch_angle_deg"]
        # And it should be very near +4°
        assert abs(legacy[0]["launch_angle_deg"] - 4.0) < 2.0


class TestRawADCDiagnostics:
    """Tests for frame-level raw ADC diagnostics."""

    def test_expected_bin_uses_circular_distance_at_fft_wrap(self):
        """OPS bin comparisons should treat bin 0 and bin N-1 as adjacent."""
        assert circular_bin_distance(2047, 2, fft_size=2048) == 3

    def test_frame_diagnostics_reports_peak_angle_and_coherence(self):
        """A clean synthetic ball tone should yield strong frame diagnostics."""
        ball_speed_mph = 70.0
        expected_bin = expected_ball_bin_from_speed(ball_speed_mph)
        frame = TestOpsBinSoftAnchor._make_frame_at_bin(
            peak_bin=expected_bin,
            angle_deg=7.0,
            amplitude=8000.0,
            seed=123,
        )

        diag = radc_frame_diagnostics(
            frame,
            frame_index=4,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=40.0,
            orientation="vertical",
        )

        assert diag.valid_payload
        assert diag.expected_bin == expected_bin
        assert diag.peak_bin is not None
        assert abs(diag.peak_bin - expected_bin) <= 2
        assert diag.bin_error is not None and diag.bin_error <= 2
        assert diag.speed_error_mph is not None and abs(diag.speed_error_mph) < 1.0
        assert diag.snr_linear > 10.0
        assert diag.phase_coherence is not None and diag.phase_coherence > 0.8
        assert diag.angle_centroid_deg is not None
        assert abs(diag.angle_centroid_deg - 7.0) < 2.0

        as_dict = diag.to_dict()
        assert as_dict["channel_stats"]["f1a_i"]["std"] > 0
        assert as_dict["iq_stats"]["f1a"]["q_to_i_std_ratio"] > 0

    def test_frame_diagnostics_handles_missing_and_invalid_payloads(self):
        missing = radc_frame_diagnostics({"timestamp": 1.0}, frame_index=0)
        assert not missing.valid_payload
        assert missing.reason == "missing_radc"
        assert "missing_radc" in missing.warnings

        invalid = radc_frame_diagnostics({"timestamp": 1.0, "radc": b"\x00" * 10})
        assert not invalid.valid_payload
        assert invalid.reason == "invalid_payload_size"
        assert "invalid_payload" in invalid.warnings

    def test_capture_diagnostics_summarizes_peak_bins_and_warnings(self):
        ball_speed_mph = 70.0
        expected_bin = expected_ball_bin_from_speed(ball_speed_mph)
        frames = [
            TestOpsBinSoftAnchor._make_frame_at_bin(
                peak_bin=expected_bin,
                angle_deg=5.0,
                amplitude=8000.0,
                seed=11,
            ),
            {"timestamp": 2.0},
        ]

        diagnostics, summary = radc_capture_diagnostics(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=40.0,
        )

        assert len(diagnostics) == 2
        assert summary["frame_count"] == 2
        assert summary["valid_payload_count"] == 1
        assert summary["peak_frame_count"] == 1
        assert summary["expected_bin"] == expected_bin
        assert summary["median_abs_bin_error"] is not None
        assert summary["median_abs_bin_error"] <= 2
        assert summary["median_abs_speed_error_mph"] is not None
        assert summary["median_abs_speed_error_mph"] < 1.0
        assert summary["warnings_by_type"]["missing_radc"] == 1
        assert summary["peak_bin_histogram_top"][0]["bin"] == diagnostics[0].peak_bin

    def test_extraction_skips_invalid_payloads(self):
        ball_speed_mph = 70.0
        expected_bin = expected_ball_bin_from_speed(ball_speed_mph)
        frames = TestOpsBinSoftAnchor._impact_window(
            center_frame=4,
            total=12,
            ball_frames=[
                TestOpsBinSoftAnchor._make_frame_at_bin(
                    peak_bin=expected_bin,
                    angle_deg=6.0,
                    amplitude=9000.0,
                    seed=31,
                ),
                {"timestamp": 1000.28, "radc": b"\x00" * 3045},
                TestOpsBinSoftAnchor._make_frame_at_bin(
                    peak_bin=expected_bin,
                    angle_deg=6.0,
                    amplitude=9000.0,
                    seed=32,
                ),
            ],
        )

        impact_indices = find_impact_frames(
            frames,
            ball_bands=[(expected_bin - 20, expected_bin + 21)],
            energy_threshold=1.5,
        )
        results = extract_launch_angle(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=5.0,
            impact_energy_threshold=1.5,
            orientation="horizontal",
        )

        assert impact_indices
        assert results
        assert results[0]["launch_angle_deg"] == pytest.approx(6.0, abs=3.0)

    def test_horizontal_angle_limit_can_accept_wider_trackman_replay_target(self):
        """The default ±15° horizontal bound is tunable for TrackMan replay."""
        ball_speed_mph = 95.0
        expected_bin = expected_ball_bin_from_speed(ball_speed_mph)
        frames = TestOpsBinSoftAnchor._impact_window(
            center_frame=4,
            total=12,
            ball_frames=[
                TestOpsBinSoftAnchor._make_frame_at_bin(
                    peak_bin=expected_bin,
                    angle_deg=20.0,
                    amplitude=9000.0,
                    seed=41,
                ),
                TestOpsBinSoftAnchor._make_frame_at_bin(
                    peak_bin=expected_bin,
                    angle_deg=20.0,
                    amplitude=9000.0,
                    seed=42,
                ),
            ],
        )

        default_results = extract_launch_angle(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=5.0,
            impact_energy_threshold=1.5,
            orientation="horizontal",
        )
        wide_results = extract_launch_angle(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=5.0,
            impact_energy_threshold=1.5,
            orientation="horizontal",
            horizontal_angle_limit_deg=30.0,
        )

        assert default_results == []
        assert wide_results
        assert wide_results[0]["launch_angle_deg"] == pytest.approx(20.0, abs=3.0)

    def test_horizontal_ops_anchored_min_snr_can_accept_weak_near_ops_peak(self):
        """Replay can lower the OPS-local SNR gate without changing defaults."""
        ball_speed_mph = 95.0
        expected_bin = expected_ball_bin_from_speed(ball_speed_mph)
        frames = TestOpsBinSoftAnchor._impact_window(
            center_frame=4,
            total=12,
            ball_frames=[
                TestOpsBinSoftAnchor._make_frame_at_bin(
                    peak_bin=expected_bin,
                    angle_deg=6.0,
                    amplitude=30.0,
                    seed=100,
                ),
                TestOpsBinSoftAnchor._make_frame_at_bin(
                    peak_bin=expected_bin,
                    angle_deg=6.0,
                    amplitude=30.0,
                    seed=101,
                ),
            ],
        )

        default_results = extract_launch_angle(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=5.0,
            impact_energy_threshold=1.5,
            orientation="horizontal",
        )
        relaxed_results = extract_launch_angle(
            frames,
            ops243_ball_speed_mph=ball_speed_mph,
            speed_tolerance_mph=5.0,
            impact_energy_threshold=1.5,
            orientation="horizontal",
            ops_anchored_peak_min_snr=2.0,
        )

        assert default_results == []
        assert relaxed_results
        assert relaxed_results[0]["launch_angle_deg"] == pytest.approx(6.0, abs=3.0)
