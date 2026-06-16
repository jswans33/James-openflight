"""Tests for replaying experimental K-LD7 RADC logs against TrackMan."""

import base64
import csv
import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))

import replay_kld7_trackman as replay

VALID_RADC_B64 = base64.b64encode(b"\x00" * replay.RADC_PAYLOAD_BYTES).decode("ascii")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "shot_number_of",
        "timestamp_of",
        "club",
        "ball_speed_of",
        "club_speed_of",
        "launch_v_tm",
        "launch_h_tm",
        "match_quality",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_targets_reads_good_vertical_and_horizontal_pairs(tmp_path):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 12.3,
                "launch_h_tm": -4.5,
                "match_quality": "good",
            },
            {
                "shot_number_of": 11,
                "club": "driver",
                "ball_speed_of": 151.0,
                "launch_v_tm": 13.0,
                "launch_h_tm": "",
                "match_quality": "ball_speed_mismatch",
            },
        ],
    )

    targets = replay.load_targets(comparison)

    assert [(t.shot_number, t.orientation, t.trackman_angle_deg) for t in targets] == [
        (10, "vertical", 12.3),
        (10, "horizontal", -4.5),
    ]


def test_load_targets_can_filter_axis(tmp_path):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 12.3,
                "launch_h_tm": -4.5,
                "match_quality": "good",
            },
        ],
    )

    targets = replay.load_targets(comparison, axis="vertical")

    assert [(t.shot_number, t.orientation) for t in targets] == [(10, "vertical")]


def test_load_targets_preserves_openflight_timestamp_and_club_speed(tmp_path):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "timestamp_of": "2026-05-11T12:17:04.593816",
                "club": "7-iron",
                "ball_speed_of": 115.0,
                "club_speed_of": 78.5,
                "launch_v_tm": "",
                "launch_h_tm": -2.5,
                "match_quality": "good",
            },
        ],
    )

    target = replay.load_targets(comparison, axis="horizontal")[0]

    assert target.openflight_timestamp is not None
    assert target.openflight_timestamp.isoformat() == "2026-05-11T12:17:04.593816"
    assert target.club_speed_mph == 78.5


def test_load_buffers_decodes_experimental_radc_payload(tmp_path):
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "pdat": [], "radc_b64": VALID_RADC_B64}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    buffers = replay.load_buffers(log, jsonl_window_before_s=2.0)

    assert buffers[(10, "vertical")][0]["radc"] == b"\x00" * replay.RADC_PAYLOAD_BYTES


def test_load_buffers_filters_jsonl_radc_frames_to_live_shot_window(tmp_path):
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "shot_timestamp": 100.0,
                "orientation": "vertical",
                "frames": [
                    {"timestamp": 97.9, "pdat": [], "radc_b64": VALID_RADC_B64},
                    {"timestamp": 98.0, "pdat": [], "radc_b64": VALID_RADC_B64},
                    {"timestamp": 100.5, "pdat": [], "radc_b64": VALID_RADC_B64},
                    {"timestamp": 100.8, "pdat": [], "radc_b64": VALID_RADC_B64},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    buffers = replay.load_buffers(log, jsonl_window_before_s=2.0)

    assert [frame["timestamp"] for frame in buffers[(10, "vertical")]] == [
        98.0,
        100.5,
    ]


def test_load_buffers_rejects_wrong_size_experimental_radc_payload(tmp_path):
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "pdat": [], "radc_b64": "AQID"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid radc_b64 payload size"):
        replay.load_buffers(log, jsonl_window_before_s=2.0)


def test_jsonl_capture_info_reads_session_wall_clock_bounds(tmp_path):
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_start",
                        "ts": "2026-05-11T12:00:01",
                        "start_time": "2026-05-11T12:00:00",
                        "config": {
                            "kld7_experiments": {
                                "trackman_calibration_enabled": False,
                                "raw_radc_payload_logging_enabled": True,
                                "raw_radc_payload_logging_requested": True,
                                "radc_tuning_enabled": False,
                                "radc_tuning_params": {},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "kld7_buffer",
                        "ts": "2026-05-11T12:05:30",
                        "radc_frame_count": 2,
                        "radc_payload_count": 2,
                        "radc_payload_valid_count": 2,
                        "radc_payload_invalid_count": 0,
                        "radc_payload_expected": True,
                        "radc_payload_complete": True,
                    }
                ),
                json.dumps(
                    {
                        "type": "kld7_buffer",
                        "ts": "2026-05-11T12:05:31",
                        "radc_frame_count": 2,
                        "radc_payload_count": 1,
                        "radc_payload_valid_count": 0,
                        "radc_payload_invalid_count": 1,
                        "radc_payload_expected": True,
                        "radc_payload_complete": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    info = replay.jsonl_capture_info(log)

    assert info["capture_start"].isoformat() == "2026-05-11T12:00:00"
    assert info["capture_end"].isoformat() == "2026-05-11T12:05:31"
    assert info["kld7_buffer_count"] == 2
    assert info["kld7_radc_frames_total"] == 4
    assert info["kld7_radc_payloads_total"] == 3
    assert info["kld7_radc_payloads_valid_total"] == 2
    assert info["kld7_radc_payloads_invalid_total"] == 1
    assert info["kld7_payload_expected_count"] == 2
    assert info["kld7_payload_complete_count"] == 1
    assert info["kld7_payload_incomplete_count"] == 1
    assert info["kld7_experiments"] == {
        "trackman_calibration_enabled": False,
        "raw_radc_payload_logging_enabled": True,
        "raw_radc_payload_logging_requested": True,
        "radc_tuning_enabled": False,
        "radc_tuning_params": {},
    }


def test_load_buffers_reads_pickle_capture_with_shot_number_mapping(tmp_path):
    capture = tmp_path / "radc.pkl"
    with capture.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {"orientation": "horizontal"},
                "frames": [
                    {"timestamp": 94.0, "radc": b"too-old"},
                    {"timestamp": 95.0, "radc": b"shot-10-pre"},
                    {"timestamp": 100.2, "radc": b"shot-10"},
                    {"timestamp": 100.9, "radc": b"between"},
                    {"timestamp": 104.0, "radc": b"shot-11-pre"},
                    {"timestamp": 109.2, "radc": b"shot-11"},
                    {"timestamp": 110.0, "radc": b"too-late"},
                ],
                "ops243_shots": [
                    {"timestamp": 100.0},
                    {"timestamp": 109.0},
                ],
            },
            handle,
        )

    buffers = replay.load_buffers(
        capture,
        pickle_first_shot_number=10,
        pickle_buffer_seconds=5.0,
        pickle_shot_window_after_s=0.5,
    )

    assert [frame["radc"] for frame in buffers[(10, "horizontal")]] == [
        b"shot-10-pre",
        b"shot-10",
    ]
    assert [frame["radc"] for frame in buffers[(11, "horizontal")]] == [
        b"shot-11-pre",
        b"shot-11",
    ]


def test_pickle_first_shot_candidates_use_orientation_and_club_metadata():
    targets = [
        replay.TrackmanTarget(10, "horizontal", -2.0, 120.0, "driver"),
        replay.TrackmanTarget(28, "horizontal", -2.0, 120.0, "7-iron"),
        replay.TrackmanTarget(29, "horizontal", -2.0, 120.0, "7-iron"),
        replay.TrackmanTarget(30, "vertical", 14.0, 120.0, "7-iron"),
    ]

    candidates = replay.pickle_first_shot_candidates(
        targets,
        {"orientation": "horizontal", "club": "7i", "shot_count": 3},
    )

    assert candidates == [26, 27, 28, 29]


def test_replay_one_reports_trackman_error(monkeypatch):
    target = replay.TrackmanTarget(
        shot_number=10,
        orientation="vertical",
        trackman_angle_deg=12.0,
        ball_speed_mph=150.0,
        club="driver",
    )
    params = replay.ReplayParams(
        speed_tolerance_mph=8.0,
        impact_energy_threshold=2.0,
        centroid_floor_frac=0.75,
        ops_bin_outlier_tol=12,
        ops_bin_outlier_penalty=4.0,
    )
    calls = []

    def fake_extract_launch_angle(frames, **kwargs):
        calls.append((frames, kwargs))
        return [
            {
                "launch_angle_deg": 4.0,
                "frame_count": 2,
                "avg_snr_db": 7.2,
                "impact_frames": [1],
            },
            {
                "launch_angle_deg": 11.6,
                "frame_count": 2,
                "avg_snr_db": 7.2,
                "impact_frames": [8],
            },
        ]

    monkeypatch.setattr(replay, "extract_launch_angle", fake_extract_launch_angle)

    row = replay.replay_one(target, [{"timestamp": 1.0, "radc": b"abc"}], params)

    assert row.error_deg == -0.40000000000000036
    assert row.reason == "ok"
    assert row.target_ball_speed_mph == 150.0
    assert row.buffer_frame_count == 1
    assert row.detection_frame_count == 2
    assert calls[0][1]["orientation"] == "vertical"
    assert calls[0][1]["speed_tolerance_mph"] == 8.0
    assert calls[0][1]["impact_energy_threshold"] == 2.0
    assert calls[0][1]["centroid_floor_frac"] == 0.75
    assert calls[0][1]["ops_bin_outlier_tol"] == 12
    assert calls[0][1]["ops_bin_outlier_penalty"] == 4.0
    assert calls[0][1]["ops_anchored_peak_min_snr"] == 5.0
    assert calls[0][1]["require_ops_anchored_peak"] is False
    assert calls[0][1]["horizontal_angle_limit_deg"] == 15.0
    assert calls[0][1]["angle_offset_deg"] == 0.0


def test_replay_one_uses_axis_specific_angle_offsets(monkeypatch):
    target = replay.TrackmanTarget(
        shot_number=10,
        orientation="horizontal",
        trackman_angle_deg=1.0,
        ball_speed_mph=150.0,
        club="driver",
    )
    params = replay.ReplayParams(
        speed_tolerance_mph=8.0,
        impact_energy_threshold=2.0,
        centroid_floor_frac=0.75,
        ops_bin_outlier_tol=12,
        ops_bin_outlier_penalty=4.0,
        vertical_angle_offset_deg=8.0,
        horizontal_angle_offset_deg=-1.5,
    )
    calls = []

    def fake_extract_launch_angle(frames, **kwargs):
        calls.append(kwargs)
        return [{"launch_angle_deg": 0.5, "frame_count": 1, "avg_snr_db": 8.0}]

    monkeypatch.setattr(replay, "extract_launch_angle", fake_extract_launch_angle)

    replay.replay_one(target, [{"timestamp": 1.0, "radc": b"abc"}], params)

    assert calls[0]["angle_offset_deg"] == -1.5


def test_summarize_reports_mae_and_max_error():
    params = replay.ReplayParams(10.0, 3.0, 0.5, 25, 10.0)
    rows = [
        replay.ReplayRow(1, "vertical", "driver", 10.0, 11.0, 1.0, 2, 5.0, "ok"),
        replay.ReplayRow(2, "vertical", "driver", 10.0, 8.0, -2.0, 2, 5.0, "ok"),
        replay.ReplayRow(3, "vertical", "driver", 10.0, None, None, 0, None, "missing"),
    ]

    summary = replay.summarize(params, rows)

    assert summary.attempted == 3
    assert summary.detected == 2
    assert summary.detection_rate == 2 / 3
    assert summary.mae == 1.5
    assert summary.p90_abs_error == 2.0
    assert summary.max_abs_error == 2.0
    assert summary.within_half_degree == 0
    assert summary.reason_counts == {"missing": 1, "ok": 2}


def test_filter_targets_to_buffers_keeps_only_mapped_targets():
    targets = [
        replay.TrackmanTarget(10, "vertical", 12.0, 150.0, "driver"),
        replay.TrackmanTarget(10, "horizontal", -2.0, 150.0, "driver"),
        replay.TrackmanTarget(11, "horizontal", -3.0, 151.0, "driver"),
    ]
    buffers = {
        (10, "horizontal"): [{"timestamp": 1.0, "radc": b"abc"}],
        (12, "horizontal"): [{"timestamp": 2.0, "radc": b"def"}],
    }

    filtered = replay.filter_targets_to_buffers(targets, buffers)

    assert filtered == [targets[1]]


def test_raw_radc_readiness_counts_missing_buffers_and_payloads():
    targets = [
        replay.TrackmanTarget(10, "vertical", 12.0, 150.0, "driver"),
        replay.TrackmanTarget(10, "horizontal", -2.0, 150.0, "driver"),
        replay.TrackmanTarget(11, "vertical", 13.0, 151.0, "driver"),
        replay.TrackmanTarget(12, "vertical", 14.0, 152.0, "driver"),
    ]
    buffers = {
        (10, "vertical"): [{"timestamp": 1.0, "radc": b"\x00" * replay.RADC_PAYLOAD_BYTES}],
        (10, "horizontal"): [{"timestamp": 1.0, "pdat": []}],
        (12, "vertical"): [{"timestamp": 1.0, "radc": b"abc"}],
    }

    assert replay.raw_radc_readiness(targets, buffers) == {
        "targets": 4,
        "buffered": 3,
        "with_radc": 1,
        "missing_buffer": 1,
        "missing_radc_payload": 1,
        "invalid_radc_payload": 1,
    }
    assert not replay.raw_radc_readiness_passes(replay.raw_radc_readiness(targets, buffers))
    assert replay.raw_radc_readiness_passes(
        {
            "targets": 1,
            "buffered": 1,
            "with_radc": 1,
            "missing_buffer": 0,
            "missing_radc_payload": 0,
            "invalid_radc_payload": 0,
        }
    )


def test_trackman_test_provenance_issues_require_clean_collection_flags():
    clean = {
        "kld7_experiments": {
            "trackman_calibration_enabled": False,
            "raw_radc_payload_logging_enabled": True,
            "raw_radc_payload_logging_requested": True,
            "radc_tuning_enabled": False,
        }
    }
    contaminated = {
        "kld7_experiments": {
            "trackman_calibration_enabled": True,
            "raw_radc_payload_logging_enabled": True,
            "raw_radc_payload_logging_requested": True,
            "radc_tuning_enabled": True,
        }
    }

    assert replay.trackman_test_provenance_issues(clean) == []
    assert replay.trackman_test_provenance_issues({}) == [
        "missing session_start config.kld7_experiments"
    ]
    assert replay.trackman_test_provenance_issues(contaminated) == [
        "trackman_calibration_enabled is not false",
        "radc_tuning_enabled is not false",
    ]


def test_summary_sort_key_prefers_coverage_before_small_subset_error():
    params = replay.ReplayParams(10.0, 3.0, 0.5, 25, 10.0)
    full_coverage = replay.ReplaySummary(
        params=params,
        attempted=10,
        detected=10,
        detection_rate=1.0,
        mae=0.9,
        p90_abs_error=1.4,
        max_abs_error=1.5,
        within_half_degree=4,
        reason_counts={"ok": 10},
    )
    tiny_subset = replay.ReplaySummary(
        params=params,
        attempted=10,
        detected=1,
        detection_rate=0.1,
        mae=0.1,
        p90_abs_error=0.1,
        max_abs_error=0.1,
        within_half_degree=1,
        reason_counts={"missing_kld7_buffer": 9, "ok": 1},
    )

    ranked = sorted(
        [tiny_subset, full_coverage],
        key=lambda summary: replay._summary_sort_key(summary, min_detection_rate=1.0),
    )

    assert ranked == [full_coverage, tiny_subset]


def test_summary_sort_key_prefers_more_detection_within_eligible_set():
    params = replay.ReplayParams(10.0, 3.0, 0.5, 25, 10.0)
    more_coverage = replay.ReplaySummary(
        params=params,
        attempted=10,
        detected=4,
        detection_rate=0.4,
        mae=2.0,
        p90_abs_error=3.0,
        max_abs_error=3.0,
        within_half_degree=1,
        reason_counts={"missing_kld7_buffer": 6, "ok": 4},
    )
    tiny_low_error_subset = replay.ReplaySummary(
        params=params,
        attempted=10,
        detected=1,
        detection_rate=0.1,
        mae=0.1,
        p90_abs_error=0.1,
        max_abs_error=0.1,
        within_half_degree=1,
        reason_counts={"missing_kld7_buffer": 9, "ok": 1},
    )

    ranked = sorted(
        [tiny_low_error_subset, more_coverage],
        key=lambda summary: replay._summary_sort_key(summary, min_detection_rate=0.1),
    )

    assert ranked == [more_coverage, tiny_low_error_subset]


def test_summary_csv_row_includes_ranked_metrics():
    params = replay.ReplayParams(8.0, 0.25, 0.75, 25, 1.0)
    summary = replay.ReplaySummary(
        params=params,
        attempted=6,
        detected=4,
        detection_rate=2 / 3,
        mae=1.234,
        p90_abs_error=2.345,
        max_abs_error=3.456,
        within_half_degree=1,
        reason_counts={"no_radc_detection": 2, "ok": 4},
    )

    row = replay._summary_csv_row(summary, 29, min_detection_rate=1.0)

    assert row == (
        "29,8,0.25,0.75,25,1,5,False,15,0,0,6,4,0.667,1,1.234,2.345,3.456,False,no_radc_detection:2|ok:4"
    )


def test_write_rows_includes_replay_parameters(tmp_path):
    params = replay.ReplayParams(8.0, 2.0, 0.75, 12, 4.0)
    rows = [
        replay.ReplayRow(
            shot_number=1,
            orientation="vertical",
            club="driver",
            trackman_angle_deg=10.0,
            replay_angle_deg=10.4,
            error_deg=0.4,
            frame_count=2,
            avg_snr_db=7.0,
            reason="ok",
        )
    ]
    output = tmp_path / "rows.csv"

    replay.write_rows(output, rows, params)

    written = list(csv.DictReader(output.open(encoding="utf-8")))
    assert written[0]["speed_tolerance_mph"] == "8.0"
    assert written[0]["target_ball_speed_mph"] == ""
    assert written[0]["buffer_frame_count"] == ""
    assert written[0]["detection_frame_count"] == ""
    assert written[0]["impact_energy_threshold"] == "2.0"
    assert written[0]["centroid_floor_frac"] == "0.75"
    assert written[0]["ops_bin_outlier_tol"] == "12"
    assert written[0]["ops_bin_outlier_penalty"] == "4.0"
    assert written[0]["ops_anchored_peak_min_snr"] == "5.0"
    assert written[0]["require_ops_anchored_peak"] == "False"
    assert written[0]["horizontal_angle_limit_deg"] == "15.0"
    assert written[0]["vertical_angle_offset_deg"] == "0.0"
    assert written[0]["horizontal_angle_offset_deg"] == "0.0"


def test_write_summary_includes_best_params_and_gate_result(tmp_path):
    params = replay.ReplayParams(8.0, 2.0, 0.75, 12, 4.0)
    summary = replay.ReplaySummary(
        params=params,
        attempted=2,
        detected=2,
        detection_rate=1.0,
        mae=0.3,
        p90_abs_error=0.4,
        max_abs_error=0.4,
        within_half_degree=2,
        reason_counts={"ok": 2},
    )
    output = tmp_path / "summary.json"

    replay.write_summary(
        output,
        summary,
        min_detection_rate=1.0,
        axis="vertical",
        pickle_first_shot_number=28,
        only_buffered_targets=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["axis"] == "vertical"
    assert payload["params"] == {
        "speed_tolerance_mph": 8.0,
        "impact_energy_threshold": 2.0,
        "centroid_floor_frac": 0.75,
        "ops_bin_outlier_tol": 12,
        "ops_bin_outlier_penalty": 4.0,
        "ops_anchored_peak_min_snr": 5.0,
        "require_ops_anchored_peak": False,
        "horizontal_angle_limit_deg": 15.0,
        "vertical_angle_offset_deg": 0.0,
        "horizontal_angle_offset_deg": 0.0,
    }
    assert payload["eligible"] is True
    assert payload["passes_within_half_degree_gate"] is True
    assert payload["raw_radc_readiness_passes"] is False
    assert payload["trackman_replay_gate_passes"] is False
    assert payload["trackman_replay_gate_issues"] == [
        "raw RADC readiness not evaluated",
        "TrackMan-test provenance not evaluated",
    ]
    assert payload["max_abs_error"] == 0.4
    assert payload["pickle_first_shot_number"] == 28
    assert payload["only_buffered_targets"] is True
    assert payload["reason_counts"] == {"ok": 2}


def test_write_diagnostics_includes_target_replay_and_frame_diagnostics(tmp_path, monkeypatch):
    params = replay.ReplayParams(8.0, 2.0, 0.75, 12, 4.0)
    target = replay.TrackmanTarget(
        10,
        "vertical",
        12.0,
        150.0,
        "driver",
        openflight_timestamp=replay._parse_datetime("2026-05-11T12:00:30"),
        club_speed_mph=100.0,
    )
    row = replay.ReplayRow(
        10,
        "vertical",
        "driver",
        12.0,
        12.4,
        0.4,
        2,
        8.0,
        "ok",
        target_ball_speed_mph=150.0,
        buffer_frame_count=3,
        detection_frame_count=2,
    )
    output = tmp_path / "diagnostics.jsonl"

    class DummyDiagnostic:
        def to_dict(self):
            return {"frame_index": 0, "peak_bin": 123, "snr_db": 8.0}

    monkeypatch.setattr(
        replay,
        "radc_capture_diagnostics",
        lambda frames, **kwargs: (
            [DummyDiagnostic()],
            {"frame_count": 1, "peak_frame_count": 1, "expected_bin": 321},
        ),
    )

    replay.write_diagnostics(
        output,
        [target],
        {(10, "vertical"): [{"timestamp": 1.0, "radc": b"abc"}]},
        [row],
        params,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["target"]["shot_number"] == 10
    assert payload["target"]["openflight_timestamp"] == "2026-05-11T12:00:30"
    assert payload["replay"]["abs_error_deg"] == 0.4
    assert payload["params"]["speed_tolerance_mph"] == 8.0
    assert payload["diagnostics_summary"]["expected_bin"] == 321
    assert payload["frame_diagnostics"] == [{"frame_index": 0, "peak_bin": 123, "snr_db": 8.0}]


def test_recommended_start_kiosk_flags_are_axis_aware():
    params = replay.ReplayParams(8.0, 0.25, 0.5, 12, 4.0)

    vertical = replay.recommended_start_kiosk_flags(params, axis="vertical")
    horizontal = replay.recommended_start_kiosk_flags(params, axis="horizontal")

    assert "--experimental-kld7-radc-tuning" in vertical
    assert "--experimental-kld7-speed-tolerance 8" in vertical
    assert "--experimental-kld7-centroid-floor 0.5" in vertical
    assert "--experimental-kld7-ops-bin-tol 12" in vertical
    assert "--experimental-kld7-ops-bin-penalty 4" in vertical
    assert "--experimental-kld7-ops-anchored-min-snr 5" in vertical
    assert "--experimental-kld7-vertical-impact-energy 0.25" in vertical
    assert "--experimental-kld7-horizontal-impact-energy" not in vertical

    assert "--experimental-kld7-horizontal-impact-energy 0.25" in horizontal
    assert "--experimental-kld7-horizontal-retry-impact-energy 0.25" in horizontal
    assert "--experimental-kld7-horizontal-angle-limit 15" in horizontal
    assert "--experimental-kld7-vertical-impact-energy" not in horizontal


def test_within_half_degree_gate_requires_full_detection_and_error_bound():
    params = replay.ReplayParams(10.0, 3.0, 0.5, 25, 10.0)
    passing = replay.ReplaySummary(
        params=params,
        attempted=2,
        detected=2,
        detection_rate=1.0,
        mae=0.35,
        p90_abs_error=0.5,
        max_abs_error=0.5,
        within_half_degree=2,
        reason_counts={"ok": 2},
    )
    missing_detection = replay.ReplaySummary(
        params=params,
        attempted=2,
        detected=1,
        detection_rate=0.5,
        mae=0.1,
        p90_abs_error=0.1,
        max_abs_error=0.1,
        within_half_degree=1,
        reason_counts={"missing_kld7_buffer": 1, "ok": 1},
    )
    too_far = replay.ReplaySummary(
        params=params,
        attempted=2,
        detected=2,
        detection_rate=1.0,
        mae=0.55,
        p90_abs_error=0.6,
        max_abs_error=0.6,
        within_half_degree=1,
        reason_counts={"ok": 2},
    )

    assert replay.passes_within_half_degree_gate(passing, min_detection_rate=1.0)
    assert not replay.passes_within_half_degree_gate(
        missing_detection,
        min_detection_rate=1.0,
    )
    assert not replay.passes_within_half_degree_gate(too_far, min_detection_rate=1.0)


def test_cli_within_half_degree_gate_returns_pass_and_fail(
    tmp_path,
    monkeypatch,
    capsys,
):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def pass_extract(frames, **kwargs):
        return [{"launch_angle_deg": 10.4, "frame_count": 1, "avg_snr_db": 8.0}]

    monkeypatch.setattr(replay, "extract_launch_angle", pass_extract)
    args = [
        "--comparison",
        str(comparison),
        "--openflight",
        str(log),
        "--axis",
        "vertical",
        "--require-within-half-degree",
    ]

    assert replay.main(args) == 0
    assert "PASS: best replay satisfies" in capsys.readouterr().out

    def fail_extract(frames, **kwargs):
        return [{"launch_angle_deg": 10.6, "frame_count": 1, "avg_snr_db": 8.0}]

    monkeypatch.setattr(replay, "extract_launch_angle", fail_extract)

    assert replay.main(args) == 2
    captured = capsys.readouterr()
    assert "FAIL: best replay does not satisfy" in captured.err


def test_cli_require_raw_radc_rejects_saved_angle_only_logs(tmp_path):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "has_radc": True}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--require-raw-radc",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_reports_malformed_radc_payload_as_preflight_error(tmp_path, capsys):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "radc_b64": "AQID"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--check-raw-radc-only",
            ]
        )

    assert excinfo.value.code == 2
    assert "failed to load OpenFlight K-LD7 buffers" in capsys.readouterr().err


def test_cli_require_raw_radc_accepts_raw_enabled_logs(tmp_path, monkeypatch):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        replay,
        "extract_launch_angle",
        lambda frames, **kwargs: [{"launch_angle_deg": 10.4, "frame_count": 1}],
    )

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--require-raw-radc",
            ]
        )
        == 0
    )


def test_cli_check_raw_radc_only_reports_readiness_without_replay(tmp_path, monkeypatch, capsys):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_if_replayed(*args, **kwargs):
        raise AssertionError("raw readiness preflight should not run extraction")

    monkeypatch.setattr(replay, "extract_launch_angle", fail_if_replayed)

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--check-raw-radc-only",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "capture_raw_payloads,kld7_buffers,radc_frames,radc_payloads" in output
    assert "payload_valid,payload_invalid" in output
    assert "raw_radc_readiness,pickle_first_shot,targets" in output
    assert "raw_radc_readiness,1,1,1,1,0,0,0,True" in output
    assert "trackman_test_provenance,False,missing session_start config.kld7_experiments" in output


def test_cli_check_raw_radc_only_can_write_summary_output(tmp_path, monkeypatch):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_start",
                        "ts": "2026-05-11T12:00:00",
                        "config": {
                            "kld7_experiments": {
                                "trackman_calibration_enabled": False,
                                "raw_radc_payload_logging_enabled": True,
                                "raw_radc_payload_logging_requested": True,
                                "radc_tuning_enabled": False,
                                "radc_tuning_params": {},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "kld7_buffer",
                        "shot_number": 10,
                        "orientation": "vertical",
                        "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_output = tmp_path / "preflight.json"

    def fail_if_replayed(*args, **kwargs):
        raise AssertionError("raw readiness preflight should not run extraction")

    monkeypatch.setattr(replay, "extract_launch_angle", fail_if_replayed)

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--check-raw-radc-only",
                "--summary-output",
                str(summary_output),
            ]
        )
        == 0
    )

    payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert payload["mode"] == "raw_radc_preflight"
    assert payload["axis"] == "vertical"
    assert payload["raw_radc_readiness_passes"] is True
    assert payload["raw_radc_readiness_by_first_shot"]["1"]["passes"] is True
    assert payload["trackman_test_provenance_passes"] is True
    assert payload["trackman_test_provenance_issues"] == []
    assert payload["capture_info"]["kld7_experiments"]["raw_radc_payload_logging_requested"]


def test_cli_can_require_clean_trackman_test_provenance(tmp_path, capsys):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_start",
                        "ts": "2026-05-11T12:00:00",
                        "config": {
                            "kld7_experiments": {
                                "trackman_calibration_enabled": False,
                                "raw_radc_payload_logging_enabled": True,
                                "raw_radc_payload_logging_requested": True,
                                "radc_tuning_enabled": False,
                                "radc_tuning_params": {},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "kld7_buffer",
                        "shot_number": 10,
                        "orientation": "vertical",
                        "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--require-trackman-test-provenance",
                "--check-raw-radc-only",
            ]
        )
        == 0
    )
    assert "raw_radc_readiness,1,1,1,1,0,0,0,True" in capsys.readouterr().out


def test_cli_rejects_contaminated_trackman_test_provenance(tmp_path, capsys):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "session_start",
                "ts": "2026-05-11T12:00:00",
                "config": {
                    "kld7_experiments": {
                        "trackman_calibration_enabled": True,
                        "raw_radc_payload_logging_enabled": True,
                        "raw_radc_payload_logging_requested": True,
                        "radc_tuning_enabled": False,
                        "radc_tuning_params": {},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--require-trackman-test-provenance",
                "--check-raw-radc-only",
            ]
        )

    assert excinfo.value.code == 2
    assert "trackman_calibration_enabled is not false" in capsys.readouterr().err


def test_cli_check_raw_radc_only_fails_for_saved_angle_only_logs(tmp_path, capsys):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "has_radc": True}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--check-raw-radc-only",
            ]
        )
        == 2
    )

    output = capsys.readouterr().out
    assert "capture_raw_payloads,1,1,0,0,0,0,0,0" in output
    assert "raw_radc_readiness,1,1,1,0,0,1,0,False" in output
    assert "trackman_test_provenance,False,missing session_start config.kld7_experiments" in output


def test_cli_rejects_time_mismatched_jsonl_session(tmp_path):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "timestamp_of": "2026-05-11T12:30:00",
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_start",
                        "ts": "2026-05-11T12:00:00",
                        "config": {
                            "kld7_experiments": {
                                "trackman_calibration_enabled": False,
                                "raw_radc_payload_logging_enabled": True,
                                "raw_radc_payload_logging_requested": True,
                                "radc_tuning_enabled": False,
                                "radc_tuning_params": {},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "kld7_buffer",
                        "ts": "2026-05-11T12:01:00",
                        "shot_number": 10,
                        "orientation": "vertical",
                        "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_accepts_time_matched_jsonl_session(tmp_path, monkeypatch):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "timestamp_of": "2026-05-11T12:30:00",
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_start", "ts": "2026-05-11T12:29:00"}),
                json.dumps(
                    {
                        "type": "kld7_buffer",
                        "ts": "2026-05-11T12:30:30",
                        "shot_number": 10,
                        "orientation": "vertical",
                        "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        replay,
        "extract_launch_angle",
        lambda frames, **kwargs: [{"launch_angle_deg": 10.4, "frame_count": 1}],
    )

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--require-raw-radc",
            ]
        )
        == 0
    )


def test_cli_writes_summary_output(tmp_path, monkeypatch):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "timestamp_of": "2026-05-11T12:00:30",
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_start",
                        "ts": "2026-05-11T12:00:00",
                        "config": {
                            "kld7_experiments": {
                                "trackman_calibration_enabled": False,
                                "raw_radc_payload_logging_enabled": True,
                                "raw_radc_payload_logging_requested": True,
                                "radc_tuning_enabled": False,
                                "radc_tuning_params": {},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "kld7_buffer",
                        "ts": "2026-05-11T12:00:30",
                        "shot_number": 10,
                        "orientation": "vertical",
                        "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_output = tmp_path / "best-summary.json"
    diagnostics_output = tmp_path / "diagnostics.jsonl"

    def fake_extract(frames, **kwargs):
        return [{"launch_angle_deg": 10.4, "frame_count": 1, "avg_snr_db": 8.0}]

    monkeypatch.setattr(replay, "extract_launch_angle", fake_extract)

    class DummyDiagnostic:
        def to_dict(self):
            return {"frame_index": 0, "peak_bin": 123}

    monkeypatch.setattr(
        replay,
        "radc_capture_diagnostics",
        lambda frames, **kwargs: (
            [DummyDiagnostic()],
            {"frame_count": 1, "peak_frame_count": 1},
        ),
    )

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--summary-output",
                str(summary_output),
                "--diagnostics-output",
                str(diagnostics_output),
            ]
        )
        == 0
    )

    payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert payload["detected"] == 1
    assert payload["passes_within_half_degree_gate"] is True
    assert payload["raw_radc_readiness_passes"] is True
    assert payload["trackman_replay_gate_passes"] is True
    assert payload["trackman_replay_gate_issues"] == []
    assert payload["raw_radc_readiness"] == {
        "targets": 1,
        "buffered": 1,
        "with_radc": 1,
        "missing_buffer": 0,
        "missing_radc_payload": 0,
        "invalid_radc_payload": 0,
    }
    assert payload["capture_info"]["capture_start"] == "2026-05-11T12:00:00"
    assert payload["capture_info"]["capture_end"] == "2026-05-11T12:00:30"
    assert payload["capture_info"]["kld7_buffer_count"] == 1
    assert payload["capture_info"]["kld7_radc_payloads_total"] == 1
    assert payload["capture_info"]["kld7_radc_payloads_valid_total"] == 0
    assert payload["capture_info"]["kld7_radc_payloads_invalid_total"] == 0
    assert payload["capture_info"]["kld7_experiments"] == {
        "trackman_calibration_enabled": False,
        "raw_radc_payload_logging_enabled": True,
        "raw_radc_payload_logging_requested": True,
        "radc_tuning_enabled": False,
        "radc_tuning_params": {},
    }
    assert payload["trackman_test_provenance_passes"] is True
    assert payload["trackman_test_provenance_issues"] == []
    diagnostics_payload = json.loads(diagnostics_output.read_text(encoding="utf-8"))
    assert diagnostics_payload["target"]["shot_number"] == 10
    assert diagnostics_payload["replay"]["reason"] == "ok"
    assert diagnostics_payload["diagnostics_summary"]["peak_frame_count"] == 1


def test_cli_auto_aligns_pickle_first_shot_number(tmp_path, monkeypatch):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 28,
                "club": "7-iron",
                "ball_speed_of": 120.0,
                "launch_v_tm": "",
                "launch_h_tm": -2.0,
                "match_quality": "good",
            },
            {
                "shot_number_of": 29,
                "club": "7-iron",
                "ball_speed_of": 121.0,
                "launch_v_tm": "",
                "launch_h_tm": -5.0,
                "match_quality": "good",
            },
        ],
    )
    capture = tmp_path / "radc.pkl"
    with capture.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {"orientation": "horizontal", "club": "7i"},
                "frames": [
                    {"timestamp": 99.5, "radc": b"first"},
                    {"timestamp": 109.5, "radc": b"second"},
                ],
                "ops243_shots": [{"timestamp": 100.0}, {"timestamp": 110.0}],
            },
            handle,
        )
    summary_output = tmp_path / "summary.json"

    def fake_extract(frames, **kwargs):
        if frames[0]["radc"] == b"first":
            return [{"launch_angle_deg": -2.0, "frame_count": 1, "avg_snr_db": 8.0}]
        return [{"launch_angle_deg": -5.0, "frame_count": 1, "avg_snr_db": 8.0}]

    monkeypatch.setattr(replay, "extract_launch_angle", fake_extract)

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(capture),
                "--axis",
                "h",
                "--pickle-first-shot-number",
                "auto",
                "--pickle-buffer-seconds",
                "1",
                "--summary-output",
                str(summary_output),
            ]
        )
        == 0
    )

    payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert payload["axis"] == "horizontal"
    assert payload["pickle_first_shot_number"] == 28
    assert payload["detected"] == 2
    assert payload["passes_within_half_degree_gate"] is True


def test_cli_rejects_time_mismatched_pickle_capture(tmp_path):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 29,
                "timestamp_of": "2026-05-11T12:17:04",
                "club": "7-iron",
                "ball_speed_of": 115.0,
                "launch_v_tm": "",
                "launch_h_tm": -2.0,
                "match_quality": "good",
            },
        ],
    )
    capture = tmp_path / "radc.pkl"
    with capture.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {
                    "orientation": "horizontal",
                    "club": "7i",
                    "capture_start": "2026-05-11T11:44:17",
                    "capture_end": "2026-05-11T11:45:50",
                },
                "frames": [{"timestamp": 100.0, "radc": b"frame"}],
                "ops243_shots": [{"timestamp": 100.0}],
            },
            handle,
        )

    with pytest.raises(SystemExit) as excinfo:
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(capture),
                "--axis",
                "horizontal",
                "--pickle-first-shot-number",
                "29",
                "--only-buffered-targets",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_rejects_pickle_capture_with_no_timestamp_overlap_before_auto_alignment(
    tmp_path, capsys
):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 29,
                "timestamp_of": "2026-05-11T12:17:04",
                "club": "7-iron",
                "ball_speed_of": 115.0,
                "launch_v_tm": "",
                "launch_h_tm": -2.0,
                "match_quality": "good",
            },
        ],
    )
    capture = tmp_path / "radc.pkl"
    with capture.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {
                    "orientation": "horizontal",
                    "club": "7i",
                    "capture_start": "2026-05-11T11:44:17",
                    "capture_end": "2026-05-11T11:45:50",
                },
                "frames": [{"timestamp": 100.0, "radc": b"frame"}],
                "ops243_shots": [{"timestamp": 100.0}],
            },
            handle,
        )

    with pytest.raises(SystemExit) as excinfo:
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(capture),
                "--axis",
                "horizontal",
                "--pickle-first-shot-number",
                "auto",
                "--only-buffered-targets",
            ]
        )

    assert excinfo.value.code == 2
    assert "does not overlap any timestamped comparison targets" in capsys.readouterr().err


def test_cli_can_limit_replay_to_buffered_targets(tmp_path, monkeypatch):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
            {
                "shot_number_of": 11,
                "club": "driver",
                "ball_speed_of": 151.0,
                "launch_v_tm": 20.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_output = tmp_path / "summary.json"

    monkeypatch.setattr(
        replay,
        "extract_launch_angle",
        lambda frames, **kwargs: [{"launch_angle_deg": 10.4, "frame_count": 1}],
    )

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--only-buffered-targets",
                "--summary-output",
                str(summary_output),
            ]
        )
        == 0
    )

    payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert payload["attempted"] == 1
    assert payload["detected"] == 1
    assert payload["only_buffered_targets"] is True


def test_cli_top_limits_printed_summary_rows(tmp_path, monkeypatch, capsys):
    comparison = tmp_path / "comparison.csv"
    _write_csv(
        comparison,
        [
            {
                "shot_number_of": 10,
                "club": "driver",
                "ball_speed_of": 150.0,
                "launch_v_tm": 10.0,
                "launch_h_tm": "",
                "match_quality": "good",
            },
        ],
    )
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "type": "kld7_buffer",
                "shot_number": 10,
                "orientation": "vertical",
                "frames": [{"timestamp": 1.0, "radc_b64": VALID_RADC_B64}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        replay,
        "extract_launch_angle",
        lambda frames, **kwargs: [{"launch_angle_deg": 10.4, "frame_count": 1}],
    )

    assert (
        replay.main(
            [
                "--comparison",
                str(comparison),
                "--openflight",
                str(log),
                "--axis",
                "vertical",
                "--speed-tolerance",
                "8,10",
                "--top",
                "1",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.count("\n") >= 4
    assert "# omitted 1 lower-ranked rows" in output
    assert output.splitlines()[0].startswith("pickle_first_shot,speed_tol")
