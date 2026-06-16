"""Tests for the offline K-LD7 geometry selection report."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))

import kld7_geometry_selection_report as report

from openflight.kld7.geometry import predicted_bearing_deg


def _frame(
    index: int,
    t_ms: float,
    *,
    bin_error: int,
    snr: float,
    bearing_deg: float,
    speed_mph: float = 110.0,
) -> report.FrameReport:
    return report.FrameReport(
        shot_number=1,
        frame_index=index,
        timestamp=100.0 + t_ms / 1000.0,
        t_ms=t_ms,
        expected_bin=1800,
        peak_bin=1800 + bin_error,
        bin_error=bin_error,
        speed_mph=speed_mph,
        speed_error_mph=0.0,
        snr=snr,
        angle_centroid_deg=bearing_deg - 2.5,
        bearing_deg=bearing_deg,
        phase_coherence=0.95,
        status="candidate",
    )


def test_select_candidate_frames_prefers_primary_adjacent_pair():
    config = report.ReportConfig()
    frames = [
        _frame(38, 12.0, bin_error=12, snr=4.0, bearing_deg=-6.0),
        _frame(39, 48.0, bin_error=2, snr=18.0, bearing_deg=2.0),
        _frame(40, 82.0, bin_error=8, snr=10.0, bearing_deg=8.0),
    ]

    selected, notes = report.select_candidate_frames(frames, config)

    assert [frame.frame_index for frame in selected] == [39, 40]
    assert [frame.selection_role for frame in selected] == ["anchor", "neighbor"]
    assert notes == ["selected_2_frames"]


def test_populated_low_snr_peak_is_not_selectable():
    config = report.ReportConfig()
    low_snr = _frame(38, 35.0, bin_error=1, snr=2.5, bearing_deg=4.0)
    low_snr.peak_source = "ops_anchored_low_snr"
    low_snr.peak_selectable = False
    low_snr.status = "invalid"
    low_snr.reasons.append("ops_anchored_snr_too_low")
    anchor = _frame(39, 70.0, bin_error=2, snr=10.0, bearing_deg=8.0)

    selected, notes = report.select_candidate_frames([low_snr, anchor], config)

    assert low_snr.peak_bin is not None
    assert low_snr.selectable is False
    assert low_snr.status == "invalid"
    assert selected == [anchor]
    assert notes == ["anchor_only_no_rising_neighbor"]


def test_select_candidate_frames_uses_early_context_when_no_primary_pair():
    config = report.ReportConfig()
    frames = [
        _frame(38, 12.0, bin_error=12, snr=4.0, bearing_deg=-6.0),
        _frame(39, 48.0, bin_error=2, snr=18.0, bearing_deg=2.0),
    ]
    frames[0].peak_source = "ops_anchored_weak"

    selected, notes = report.select_candidate_frames(frames, config)

    assert [frame.frame_index for frame in selected] == [38, 39]
    assert [frame.selection_role for frame in selected] == ["neighbor", "anchor"]
    assert notes == ["selected_2_frames"]


def test_select_candidate_frames_falls_back_to_anchor_when_neighbor_is_not_rising():
    config = report.ReportConfig()
    frames = [
        _frame(38, 14.0, bin_error=5, snr=10.0, bearing_deg=7.0),
        _frame(39, 48.0, bin_error=1, snr=14.0, bearing_deg=5.0),
    ]

    selected, notes = report.select_candidate_frames(frames, config)

    assert [frame.frame_index for frame in selected] == [39]
    assert notes == ["anchor_only_no_rising_neighbor"]
    assert "not_rising" in frames[0].reasons


def test_range_shift_fit_recovers_synthetic_timing_offset():
    config = report.ReportConfig(clock_error_ms=20.0, shift_step_ms=1.0)
    ball_speed_mph = 110.0
    launch_angle_deg = 18.0
    true_times_ms = [22.0, 55.0]
    measured_time_error_ms = -10.0
    frames = []
    for index, true_t_ms in enumerate(true_times_ms, start=39):
        true_t_s = true_t_ms / 1000.0
        bearing = predicted_bearing_deg(
            launch_angle_deg,
            true_t_s,
            ball_speed_mph,
            config.ball_distance_ft,
            config.mount_deg,
            config.ball_above_radar_ft,
        )
        rng = report._predicted_range_ft(launch_angle_deg, true_t_s, ball_speed_mph, config)
        frame = _frame(
            index,
            true_t_ms + measured_time_error_ms,
            bin_error=2,
            snr=12.0,
            bearing_deg=bearing,
        )
        frame.f1b_range_ft = rng
        frames.append(frame)

    fit = report.best_range_shift_fit(frames, ball_speed_mph, config)

    assert fit is not None
    assert fit.shift_ms == pytest.approx(10.0, abs=1.0)
    assert fit.launch_angle_deg == pytest.approx(launch_angle_deg, abs=0.2)
    assert fit.range_rmse_ft == pytest.approx(0.0, abs=0.05)


def test_high_rmse_pair_falls_back_to_first_strong_single_frame():
    config = report.ReportConfig()
    frames = [
        _frame(38, 27.5, bin_error=10, snr=18.4, bearing_deg=-1.6, speed_mph=112.1),
        _frame(39, 62.5, bin_error=1, snr=16.6, bearing_deg=13.5, speed_mph=112.8),
    ]
    for frame in frames:
        frame.peak_source = "ops_anchored"
        frame.status = "selected"
    notes: list[str] = []

    selected = report.apply_high_rmse_single_frame_fallback(
        frames,
        112.7,
        config,
        notes,
    )

    assert [frame.frame_index for frame in selected] == [38]
    assert "high_rmse_pair_fallback_to_single_frame" in notes[0]
    assert frames[1].status == "rejected"
    assert "high_rmse_pair_fallback" in frames[1].reasons


def test_logged_ball_angle_selection_reads_radc_selection_when_present():
    details = report._logged_ball_angle_selection(
        {
            "ball_angle": {
                "vertical_deg": 19.9,
                "confidence": 0.93,
                "accepted": True,
                "selection_reason": "strict_accept",
                "num_frames": 2,
                "radc_selection": {
                    "estimator": "geometry",
                    "selection_path": "geometry_primary",
                    "selected_frame_indices": [39, 40],
                    "selected_t_ms": [51.4, 84.5],
                    "selected_bin_errors": [2, 7],
                    "geom_fit_rmse_deg": 1.37,
                },
            }
        }
    )

    assert details["logged_ball_angle_deg"] == pytest.approx(19.9)
    assert details["logged_ball_angle_accepted"] is True
    assert details["logged_radc_selection_available"] is True
    assert details["logged_radc_selected_frame_indices"] == [39, 40]
    assert details["logged_radc_selected_t_ms"] == [51.4, 84.5]
    assert details["logged_radc_selected_bin_errors"] == [2, 7]
    assert details["logged_radc_fit_rmse_deg"] == pytest.approx(1.37)


def test_high_snr_selector_prefers_broad_peak_snr_over_bin_error():
    frames = [
        _frame(38, 16.0, bin_error=3, snr=4.0, bearing_deg=-4.0),
        _frame(39, 52.0, bin_error=2, snr=7.0, bearing_deg=2.0),
        _frame(40, 86.0, bin_error=1, snr=8.0, bearing_deg=9.0),
    ]
    # The broad/high-SNR view can disagree with the OPS-anchored peak.
    frames[0].broad_peak_bin = 1900
    frames[0].broad_bin_error = 150
    frames[0].broad_speed_mph = 119.0
    frames[0].broad_snr = 11.0
    frames[0].broad_bearing_deg = -3.0
    frames[1].broad_peak_bin = 1910
    frames[1].broad_bin_error = 160
    frames[1].broad_speed_mph = 120.0
    frames[1].broad_snr = 30.0
    frames[1].broad_bearing_deg = 4.0
    frames[2].broad_peak_bin = 1801
    frames[2].broad_bin_error = 1
    frames[2].broad_speed_mph = 110.2
    frames[2].broad_snr = 15.0
    frames[2].broad_bearing_deg = 10.0

    high_snr_frames = report.broad_high_snr_frames(frames, 110.0, report.ReportConfig())
    selected, notes = report.select_high_snr_candidate_frames(
        high_snr_frames,
        report.ReportConfig(),
    )

    assert high_snr_frames[1].peak_source == "broad_high_snr"
    assert high_snr_frames[1].bin_error == 160
    assert [frame.frame_index for frame in selected] == [39, 40]
    assert [frame.selection_role for frame in selected] == ["anchor", "neighbor"]
    assert notes == ["selected_2_frames_high_snr"]


def test_logged_ball_angle_selection_marks_older_logs_without_radc_selection():
    details = report._logged_ball_angle_selection(
        {
            "ball_angle": {
                "vertical_deg": 21.7,
                "confidence": 0.94,
                "accepted": True,
                "selection_reason": "strict_accept",
                "num_frames": 2,
            }
        }
    )

    assert details["logged_ball_angle_deg"] == pytest.approx(21.7)
    assert details["logged_ball_angle_num_frames"] == 2
    assert details["logged_radc_selection_available"] is False
    assert details["logged_radc_selected_frame_indices"] == []


def test_shot_csv_marks_logged_radc_fields_unavailable_for_older_sessions():
    shot = report.ShotReport(
        shot_number=3,
        ball_speed_mph=112.0,
        impact_timestamp=100.0,
        impact_timestamp_source="kld7_buffer.shot_timestamp",
        logged_launch_angle_deg=12.5,
        logged_angle_source="estimated",
        logged_ball_angle_deg=29.2,
        logged_ball_angle_confidence=0.77,
        logged_ball_angle_accepted=False,
        logged_ball_angle_selection_reason="outside_soft_lane",
        logged_ball_angle_num_frames=2,
        logged_radc_selection_available=False,
        logged_radc_estimator=None,
        logged_radc_selection_path=None,
        logged_radc_selected_frame_indices=[],
        logged_radc_selected_t_ms=[],
        logged_radc_selected_bin_errors=[],
        logged_radc_fit_rmse_deg=None,
        frame_count=204,
        considered_frame_count=63,
        selected_frame_indices=[38],
        selection_method="single_anchor",
        selection_notes=[],
        nominal_fit=None,
        best_range_fit=None,
        minus_sensitivity_fit=None,
        plus_sensitivity_fit=None,
    )

    row = report._shot_csv_row(shot)

    assert row["logged_radc_estimator"] == "not_logged_in_session"
    assert row["logged_radc_selection_path"] == "not_logged_in_session"
    assert row["logged_radc_selected_frame_indices"] == "not_logged_in_session"
    assert row["logged_radc_selected_t_ms"] == "not_logged_in_session"
    assert row["logged_radc_selected_bin_errors"] == "not_logged_in_session"
    assert row["logged_radc_fit_rmse_deg"] == "not_logged_in_session"


def test_unwrap_f1b_ranges_adds_period_for_good_wrapped_ball_frame():
    config = report.ReportConfig(kld7_range_m=5.0)
    frames = [
        _frame(39, 35.0, bin_error=3, snr=18.0, bearing_deg=4.6),
        _frame(40, 70.0, bin_error=2, snr=10.5, bearing_deg=24.8),
    ]
    frames[0].f1b_range_ft = 13.25
    frames[1].f1b_range_ft = 1.67

    report.unwrap_f1b_ranges(frames, config)

    assert frames[0].f1b_range_unwrapped_ft == pytest.approx(13.25)
    assert frames[0].f1b_range_unwrap_count == 0
    assert frames[1].f1b_range_unwrapped_ft == pytest.approx(1.67 + config.unambiguous_range_ft)
    assert frames[1].f1b_range_unwrap_count == 1


def test_unwrap_f1b_ranges_does_not_unwrap_bad_bin_or_low_snr_frame():
    config = report.ReportConfig(range_unwrap_bin_error_max=80, range_unwrap_snr_min=3.0)
    bad_bin = _frame(39, 35.0, bin_error=120, snr=18.0, bearing_deg=4.6)
    low_snr = _frame(40, 70.0, bin_error=2, snr=2.5, bearing_deg=24.8)
    bad_bin.f1b_range_ft = 1.67
    low_snr.f1b_range_ft = 1.91

    report.unwrap_f1b_ranges([bad_bin, low_snr], config)

    assert bad_bin.f1b_range_unwrapped_ft == pytest.approx(1.67)
    assert bad_bin.f1b_range_unwrap_count == 0
    assert low_snr.f1b_range_unwrapped_ft == pytest.approx(1.91)
    assert low_snr.f1b_range_unwrap_count == 0
