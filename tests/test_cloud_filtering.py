"""Tests for the openflight-cloud client-side filtering (raw-ADC strip)."""

import gzip
import json
import uuid

import pytest

from openflight.cloud import filtering as flt


def _line(entry_type, **fields):
    return json.dumps({"ts": "2026-06-14T00:00:00", "type": entry_type, **fields})


class TestFilterSessionLines:
    def test_keeps_only_allowlisted_types(self):
        lines = [
            _line("session_start", session_uuid="u"),
            _line("rolling_buffer_capture", i_samples=[1, 2, 3]),
            _line("shot_detected", ball_speed_mph=100),
            _line("iq_blocks", blocks=[]),
            _line("trigger_event", accepted=True),
            _line("session_end"),
        ]
        result = flt.filter_session_lines(lines, device_id="dev-1")
        kept_types = [json.loads(line)["type"] for line in result.kept_lines]
        assert kept_types == [
            "session_start",
            "shot_detected",
            "trigger_event",
            "session_end",
        ]

    def test_keeps_both_error_and_session_error(self):
        lines = [
            _line("error", error="boom"),
            _line("session_error", error="boom2"),
        ]
        result = flt.filter_session_lines(lines, device_id="dev-1")
        kept_types = [json.loads(line)["type"] for line in result.kept_lines]
        assert kept_types == ["error", "session_error"]

    def test_drops_unknown_future_type_by_default(self):
        lines = [_line("some_future_heavy_type", data="x" * 10)]
        result = flt.filter_session_lines(lines, device_id="dev-1")
        assert result.kept_lines == []

    def test_drops_kept_line_over_32kb_and_counts_it(self):
        big = _line("shot_detected", note="x" * (33 * 1024))
        small = _line("shot_detected", ball_speed_mph=90)
        result = flt.filter_session_lines([big, small], device_id="dev-1")
        assert result.dropped_oversize == 1
        assert len(result.kept_lines) == 1

    def test_ignores_blank_and_unparseable_lines(self):
        lines = ["", "   ", "not json", _line("shot_detected", ball_speed_mph=90)]
        result = flt.filter_session_lines(lines, device_id="dev-1")
        assert len(result.kept_lines) == 1

    def test_manifest_has_expected_shape(self):
        lines = [
            _line("session_start", session_uuid="u"),
            _line("shot_detected", ball_speed_mph=90),
        ]
        result = flt.filter_session_lines(lines, device_id="dev-7", client_version="9.9.9")
        m = result.manifest
        assert m["type"] == "upload_manifest"
        assert m["format_version"] == 1
        assert m["client_version"] == "9.9.9"
        assert m["device_id"] == "dev-7"
        assert m["filtered"] is True
        assert set(m["kept_entry_types"]) == {"session_start", "shot_detected"}

    def test_kept_entry_types_are_sorted_and_unique(self):
        lines = [
            _line("shot_detected"),
            _line("shot_detected"),
            _line("session_start"),
        ]
        result = flt.filter_session_lines(lines, device_id="d")
        assert result.manifest["kept_entry_types"] == ["session_start", "shot_detected"]


class TestFilterSessionFile:
    """Streaming filter that reads a file path without materializing it."""

    def _write(self, path, *entries):
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    def test_keeps_shots_and_strips_raw_adc(self, tmp_path):
        path = tmp_path / "session_x.jsonl"
        self._write(
            path,
            {"type": "session_start", "session_uuid": "1F0E9C2A-7B3D-4E5F-8A9B-0C1D2E3F4A5B"},
            {"type": "rolling_buffer_capture", "i_samples": [1, 2, 3], "q_samples": [4, 5]},
            {"type": "shot_detected", "ball_speed_mph": 142},
            {"type": "kld7_buffer", "frames": [{"radc_b64": "AAAA"}]},
            {"type": "shot_detected", "ball_speed_mph": 99},
            {"type": "session_end"},
        )
        result = flt.filter_session_file(path, device_id="dev-1")
        kept_types = [json.loads(line)["type"] for line in result.kept_lines]
        assert kept_types == ["session_start", "shot_detected", "shot_detected", "session_end"]
        assert result.kept_type_counts["shot_detected"] == 2

    def test_resolves_session_id_from_embedded_uuid(self, tmp_path):
        path = tmp_path / "session_x.jsonl"
        self._write(
            path,
            {"type": "session_start", "session_uuid": "1F0E9C2A-7B3D-4E5F-8A9B-0C1D2E3F4A5B"},
            {"type": "shot_detected", "ball_speed_mph": 90},
        )
        result = flt.filter_session_file(path, device_id="dev-1")
        assert result.session_id == "1f0e9c2a-7b3d-4e5f-8a9b-0c1d2e3f4a5b"

    def test_session_id_uuid5_fallback_uses_filename(self, tmp_path):
        path = tmp_path / "session_x.jsonl"
        self._write(path, {"type": "session_start"}, {"type": "shot_detected"})
        result = flt.filter_session_file(path, device_id="dev-1")
        expected = str(uuid.uuid5(flt.SESSION_NAMESPACE, "dev-1:session_x.jsonl"))
        assert result.session_id == expected

    def test_does_not_load_whole_file_into_memory(self, tmp_path):
        """Regression: filtering a large raw-ADC file must use bounded memory.

        The old path read the entire file via read_text().splitlines(), peaking
        at ~2x the file size — enough to OOM-kill the push on a Pi, dropping the
        whole upload ("0 shots uploaded"). Streaming keeps peak ~one line.
        """
        import tracemalloc

        path = tmp_path / "session_big.jsonl"
        # ~20 MB of raw ADC (100 lines x ~200 KB), plus a handful of shots.
        big = {"type": "rolling_buffer_capture", "i_samples": list(range(25000))}
        with path.open("w") as f:
            f.write(json.dumps({"type": "session_start", "session_uuid": "u"}) + "\n")
            for i in range(100):
                f.write(json.dumps(big) + "\n")
                if i % 25 == 0:
                    f.write(json.dumps({"type": "shot_detected", "ball_speed_mph": 90}) + "\n")
            f.write(json.dumps({"type": "session_end"}) + "\n")

        file_size = path.stat().st_size
        assert file_size > 15 * 1024 * 1024  # sanity: the file really is large

        tracemalloc.start()
        result = flt.filter_session_file(path, device_id="d")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert result.kept_type_counts["shot_detected"] == 4
        # Streaming peaks at roughly one line; the file is >15 MB. A generous
        # 8 MB ceiling cleanly fails the old whole-file-in-memory approach.
        assert peak < 8 * 1024 * 1024, f"peak {peak / 1024 / 1024:.1f} MB — file not streamed"


class TestBuildUploadBody:
    def test_body_is_gzip_with_manifest_first(self):
        lines = [
            _line("session_start", session_uuid="u"),
            _line("shot_detected", ball_speed_mph=90),
        ]
        result = flt.filter_session_lines(lines, device_id="dev-1")
        body = flt.build_upload_body(result)

        ndjson = gzip.decompress(body).decode("utf-8")
        out_lines = ndjson.strip().split("\n")
        first = json.loads(out_lines[0])
        assert first["type"] == "upload_manifest"
        assert json.loads(out_lines[1])["type"] == "session_start"
        assert json.loads(out_lines[2])["type"] == "shot_detected"

    def test_raises_when_gzip_exceeds_cap(self):
        result = flt.FilterResult(
            manifest={"type": "upload_manifest"},
            kept_lines=[_line("shot_detected", ball_speed_mph=90)],
            dropped_oversize=0,
        )
        # 1-byte caps make any real body too large.
        with pytest.raises(flt.BodyTooLargeError):
            flt.build_upload_body(result, max_gzip_bytes=1, max_inflated_bytes=1_000_000)

    def test_raises_when_inflated_exceeds_cap(self):
        result = flt.FilterResult(
            manifest={"type": "upload_manifest"},
            kept_lines=[_line("shot_detected", ball_speed_mph=90)],
            dropped_oversize=0,
        )
        with pytest.raises(flt.BodyTooLargeError):
            flt.build_upload_body(result, max_gzip_bytes=1_000_000, max_inflated_bytes=1)
