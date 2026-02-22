"""Tests for DTL (down-the-line) swing camera module."""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openflight.dtl_camera import (
    DTLCameraStatus,
    DTLConfig,
    MockDTLCameraRecorder,
    SavedClip,
)


# ---------------------------------------------------------------------------
# DTLConfig tests
# ---------------------------------------------------------------------------
class TestDTLConfig:
    """Tests for DTLConfig dataclass."""

    def test_default_config(self):
        """Default config should have sane values for 1080p swing recording."""
        config = DTLConfig()
        assert config.width == 1920
        assert config.height == 1080
        assert config.framerate == 30
        assert config.pre_trigger_seconds == 2.0
        assert config.post_trigger_seconds == 3.0
        assert config.quality == "MEDIUM"
        assert config.camera_num == 0

    def test_custom_config(self):
        """Custom values should override defaults."""
        config = DTLConfig(
            width=1280,
            height=720,
            framerate=60,
            pre_trigger_seconds=1.5,
            post_trigger_seconds=4.0,
            quality="HIGH",
            camera_num=1,
        )
        assert config.width == 1280
        assert config.height == 720
        assert config.framerate == 60
        assert config.pre_trigger_seconds == 1.5
        assert config.post_trigger_seconds == 4.0
        assert config.quality == "HIGH"
        assert config.camera_num == 1


# ---------------------------------------------------------------------------
# SavedClip tests
# ---------------------------------------------------------------------------
class TestSavedClip:
    """Tests for SavedClip dataclass."""

    def test_clip_creation(self):
        """Create a basic clip metadata object."""
        clip = SavedClip(
            path=Path("/tmp/dtl_shot001_143000.h264"),
            shot_number=1,
            trigger_time="2026-02-22T14:30:00",
            pre_seconds=2.0,
            post_seconds=3.0,
            resolution="1920x1080",
            framerate=30,
            file_size_bytes=1024000,
        )
        assert clip.shot_number == 1
        assert clip.resolution == "1920x1080"
        assert clip.file_size_bytes == 1024000

    def test_default_file_size(self):
        """File size should default to 0."""
        clip = SavedClip(
            path=Path("/tmp/test.h264"),
            shot_number=1,
            trigger_time="2026-02-22T14:30:00",
            pre_seconds=2.0,
            post_seconds=3.0,
            resolution="1920x1080",
            framerate=30,
        )
        assert clip.file_size_bytes == 0


# ---------------------------------------------------------------------------
# DTLCameraStatus tests
# ---------------------------------------------------------------------------
class TestDTLCameraStatus:
    """Tests for DTLCameraStatus enum."""

    def test_status_values(self):
        """Status enum should have expected string values."""
        assert DTLCameraStatus.IDLE.value == "idle"
        assert DTLCameraStatus.BUFFERING.value == "buffering"
        assert DTLCameraStatus.SAVING.value == "saving"
        assert DTLCameraStatus.ERROR.value == "error"

    def test_status_is_string_enum(self):
        """Status should work as a string for JSON serialization."""
        assert str(DTLCameraStatus.BUFFERING) == "DTLCameraStatus.BUFFERING"
        assert DTLCameraStatus.BUFFERING.value == "buffering"


# ---------------------------------------------------------------------------
# MockDTLCameraRecorder tests
# ---------------------------------------------------------------------------
class TestMockDTLCameraRecorder:
    """Tests for the mock recorder (no hardware required)."""

    def test_start_stop(self, tmp_path):
        """Mock recorder should start and stop cleanly."""
        recorder = MockDTLCameraRecorder(clip_dir=tmp_path)
        assert not recorder.is_running
        assert recorder.status == DTLCameraStatus.IDLE

        recorder.start()
        assert recorder.is_running
        assert recorder.status == DTLCameraStatus.BUFFERING

        recorder.stop()
        assert not recorder.is_running
        assert recorder.status == DTLCameraStatus.IDLE

    def test_context_manager(self, tmp_path):
        """Mock recorder should work as a context manager."""
        with MockDTLCameraRecorder(clip_dir=tmp_path) as recorder:
            assert recorder.is_running
            assert recorder.status == DTLCameraStatus.BUFFERING
        assert not recorder.is_running

    def test_on_shot_saves_clip(self, tmp_path):
        """on_shot should save a clip file and return metadata."""
        recorder = MockDTLCameraRecorder(clip_dir=tmp_path)
        recorder.start()

        mock_shot = MagicMock()
        mock_shot.ball_speed_mph = 150.0

        clip = recorder.on_shot(mock_shot)

        assert clip is not None
        assert clip.shot_number == 1
        assert clip.path.exists()
        assert clip.file_size_bytes == 128
        assert "dtl_shot001" in clip.path.name
        assert clip.path.suffix == ".h264"

        recorder.stop()

    def test_multiple_shots_increment_number(self, tmp_path):
        """Each shot should increment the clip number."""
        recorder = MockDTLCameraRecorder(clip_dir=tmp_path)
        recorder.start()

        mock_shot = MagicMock()

        clip1 = recorder.on_shot(mock_shot)
        clip2 = recorder.on_shot(mock_shot)
        clip3 = recorder.on_shot(mock_shot)

        assert clip1.shot_number == 1
        assert clip2.shot_number == 2
        assert clip3.shot_number == 3
        assert len(recorder.clips) == 3

        recorder.stop()

    def test_on_shot_when_not_running_returns_none(self, tmp_path):
        """on_shot should return None if recorder isn't running."""
        recorder = MockDTLCameraRecorder(clip_dir=tmp_path)
        mock_shot = MagicMock()

        result = recorder.on_shot(mock_shot)
        assert result is None

    def test_clips_list_is_copy(self, tmp_path):
        """clips property should return a copy, not the internal list."""
        recorder = MockDTLCameraRecorder(clip_dir=tmp_path)
        recorder.start()
        recorder.on_shot(MagicMock())

        clips = recorder.clips
        clips.clear()

        assert len(recorder.clips) == 1  # Internal list unaffected

        recorder.stop()

    def test_status_callback_called(self, tmp_path):
        """Status callback should fire on state transitions."""
        statuses = []

        def on_status(status):
            statuses.append(status)

        recorder = MockDTLCameraRecorder(
            clip_dir=tmp_path, status_callback=on_status,
        )
        recorder.start()
        recorder.on_shot(MagicMock())
        recorder.stop()

        assert DTLCameraStatus.BUFFERING in statuses
        assert DTLCameraStatus.SAVING in statuses
        assert DTLCameraStatus.IDLE in statuses

    def test_custom_config_used(self, tmp_path):
        """Custom config should propagate to clip metadata."""
        config = DTLConfig(
            width=1280, height=720, framerate=60,
            pre_trigger_seconds=1.0, post_trigger_seconds=2.0,
        )
        recorder = MockDTLCameraRecorder(config=config, clip_dir=tmp_path)
        recorder.start()

        clip = recorder.on_shot(MagicMock())
        assert clip.resolution == "1280x720"
        assert clip.framerate == 60
        assert clip.pre_seconds == 1.0
        assert clip.post_seconds == 2.0

        recorder.stop()

    def test_clip_dir_created(self, tmp_path):
        """start() should create clip directory if it doesn't exist."""
        clip_dir = tmp_path / "nested" / "clips"
        assert not clip_dir.exists()

        recorder = MockDTLCameraRecorder(clip_dir=clip_dir)
        recorder.start()

        assert clip_dir.exists()
        recorder.stop()

    def test_status_callback_exception_swallowed(self, tmp_path):
        """Broken status callback should not crash the recorder."""
        def bad_callback(status):
            raise ValueError("boom")

        recorder = MockDTLCameraRecorder(
            clip_dir=tmp_path, status_callback=bad_callback,
        )
        # Should not raise
        recorder.start()
        recorder.on_shot(MagicMock())
        recorder.stop()


# ---------------------------------------------------------------------------
# Session logger integration tests
# ---------------------------------------------------------------------------
class TestSessionLoggerDTLClip:
    """Tests for the DTL clip logging in SessionLogger."""

    def test_log_dtl_clip_writes_entry(self, tmp_path):
        """log_dtl_clip should write a valid JSONL entry."""
        from openflight.session_logger import SessionLogger

        logger = SessionLogger(log_dir=tmp_path, enabled=True)
        logger.start_session(mode="streaming")

        logger.log_dtl_clip(
            shot_number=1,
            clip_path="/home/pi/openflight_sessions/dtl_clips/dtl_shot001_143000.h264",
            trigger_time="2026-02-22T14:30:00",
            pre_seconds=2.0,
            post_seconds=3.0,
            resolution="1920x1080",
            framerate=30,
            file_size_bytes=512000,
        )

        lines = logger.session_path.read_text().strip().split("\n")
        entry = json.loads(lines[-1])

        assert entry["type"] == "dtl_clip"
        assert entry["shot_number"] == 1
        assert entry["resolution"] == "1920x1080"
        assert entry["file_size_bytes"] == 512000
        assert entry["pre_seconds"] == 2.0
        assert entry["post_seconds"] == 3.0

    def test_log_dtl_clip_disabled_noop(self, tmp_path):
        """Disabled logger should silently skip logging."""
        from openflight.session_logger import SessionLogger

        logger = SessionLogger(log_dir=tmp_path, enabled=False)
        # Should not raise
        logger.log_dtl_clip(
            shot_number=1,
            clip_path="/tmp/test.h264",
            trigger_time="2026-02-22T14:30:00",
            pre_seconds=2.0,
            post_seconds=3.0,
            resolution="1920x1080",
            framerate=30,
        )
        assert logger.session_path is None
