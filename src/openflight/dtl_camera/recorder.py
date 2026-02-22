"""
DTL swing camera recorder with ring buffer and shot-triggered clip saving.

Uses picamera2's built-in CircularOutput for an efficient H.264 ring buffer,
avoiding manual frame management. On shot detection, the buffer is flushed
to disk and recording continues for a configurable post-impact duration.

Hardware: Raspberry Pi Camera Module v3 Wide (IMX708) on CSI-0.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder, Quality
    from picamera2.outputs import CircularOutput

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logger = logging.getLogger("openflight.dtl_camera")


class DTLCameraStatus(str, Enum):
    """Camera recorder state machine."""

    IDLE = "idle"
    BUFFERING = "buffering"
    SAVING = "saving"
    ERROR = "error"


@dataclass
class DTLConfig:
    """Configuration for DTL camera recording."""

    # Resolution — 1080p is widely compatible; ISP scales from IMX708's native 2304x1296
    width: int = 1920
    height: int = 1080
    framerate: int = 30

    # Ring buffer: seconds of video to keep before impact
    pre_trigger_seconds: float = 2.0

    # Post-impact recording duration
    post_trigger_seconds: float = 3.0

    # H.264 encoder quality
    quality: str = "MEDIUM"  # LOW, MEDIUM, HIGH, VERY_HIGH

    # Camera sensor tuning
    exposure_time_us: int = 0  # 0 = auto
    analogue_gain: float = 0.0  # 0 = auto

    # Which CSI camera index to use (0 for CSI-0)
    camera_num: int = 0


@dataclass
class SavedClip:
    """Metadata about a saved swing clip."""

    path: Path
    shot_number: int
    trigger_time: str
    pre_seconds: float
    post_seconds: float
    resolution: str
    framerate: int
    file_size_bytes: int = 0


class DTLCameraRecorder:
    """
    DTL swing camera with ring buffer and shot-triggered clip saving.

    Lifecycle:
        recorder = DTLCameraRecorder(config, clip_dir)
        recorder.start()          # Begin buffering
        recorder.on_shot(shot)    # Called by shot_callback — saves clip
        recorder.stop()           # Shutdown

    The recorder runs a background thread that handles the post-trigger
    recording and file finalization so the shot callback returns immediately.
    """

    def __init__(
        self,
        config: Optional[DTLConfig] = None,
        clip_dir: Optional[Path] = None,
        status_callback: Optional[Callable[[DTLCameraStatus], None]] = None,
        clip_callback: Optional[Callable[["SavedClip"], None]] = None,
    ):
        self.config = config or DTLConfig()
        self.clip_dir = clip_dir or Path.home() / "openflight_sessions" / "dtl_clips"
        self._status_callback = status_callback
        self._clip_callback = clip_callback

        self._camera: Optional["Picamera2"] = None
        self._encoder: Optional["H264Encoder"] = None
        self._circular: Optional["CircularOutput"] = None
        self._status = DTLCameraStatus.IDLE
        self._lock = threading.Lock()
        self._save_thread: Optional[threading.Thread] = None
        self._running = False
        self._shot_count = 0
        self._clips: list[SavedClip] = []

    @property
    def status(self) -> DTLCameraStatus:
        """Current recorder state."""
        return self._status

    @property
    def clips(self) -> list[SavedClip]:
        """All clips saved this session."""
        return self._clips.copy()

    def _set_status(self, new_status: DTLCameraStatus):
        """Update status and notify listener."""
        self._status = new_status
        if self._status_callback:
            try:
                self._status_callback(new_status)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning("Status callback failed", exc_info=True)

    def start(self):
        """Initialize camera and begin buffering into the ring buffer."""
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "picamera2 is not available. Install with: sudo apt install -y python3-picamera2"
            )

        self.clip_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._camera = Picamera2(self.config.camera_num)

            video_config = self._camera.create_video_configuration(
                main={"size": (self.config.width, self.config.height)},
                controls={"FrameRate": self.config.framerate},
            )
            # Apply manual exposure if configured
            if self.config.exposure_time_us > 0:
                video_config["controls"]["ExposureTime"] = self.config.exposure_time_us
                video_config["controls"]["AeEnable"] = False
            if self.config.analogue_gain > 0:
                video_config["controls"]["AnalogueGain"] = self.config.analogue_gain

            self._camera.configure(video_config)

            quality_map = {
                "LOW": Quality.LOW,
                "MEDIUM": Quality.MEDIUM,
                "HIGH": Quality.HIGH,
                "VERY_HIGH": Quality.VERY_HIGH,
            }
            quality = quality_map.get(self.config.quality.upper(), Quality.MEDIUM)

            self._encoder = H264Encoder()
            # CircularOutput buffersize is frame count, not bytes
            buffer_frames = max(1, math.ceil(
                self.config.pre_trigger_seconds * self.config.framerate * 1.1
            ))
            self._circular = CircularOutput(buffersize=buffer_frames)

            self._camera.start()
            self._camera.start_encoder(self._encoder, self._circular, quality=quality)
            self._running = True
            self._set_status(DTLCameraStatus.BUFFERING)
            logger.info(
                "DTL camera started: %dx%d@%dfps, %.1fs pre-trigger buffer",
                self.config.width,
                self.config.height,
                self.config.framerate,
                self.config.pre_trigger_seconds,
            )

        except Exception as exc:
            self._set_status(DTLCameraStatus.ERROR)
            logger.error("Failed to start DTL camera: %s", exc)
            self._cleanup_camera()
            raise RuntimeError(f"Failed to start DTL camera: {exc}") from exc

    def stop(self):
        """Stop recording and release camera resources."""
        self._running = False

        # Wait for any in-progress save to finish
        if self._save_thread and self._save_thread.is_alive():
            logger.info("Waiting for clip save to finish...")
            self._save_thread.join(timeout=self.config.post_trigger_seconds + 2)

        self._cleanup_camera()
        self._set_status(DTLCameraStatus.IDLE)
        logger.info("DTL camera stopped. Clips saved: %d", len(self._clips))

    def _cleanup_camera(self):
        """Release picamera2 resources."""
        try:
            if self._camera:
                try:
                    self._camera.stop_encoder()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                self._camera.stop()
                self._camera.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        finally:
            self._camera = None
            self._encoder = None
            self._circular = None

    def on_shot(self, shot) -> Optional[SavedClip]:  # pylint: disable=unused-argument
        """
        Shot detection callback — triggers clip save.

        This is designed to be passed directly as (or called from) the
        LaunchMonitor shot_callback. It kicks off clip saving in a
        background thread so the radar pipeline isn't blocked.

        Args:
            shot: Shot object from launch_monitor (accepts but does not
                  use the shot data — kept for callback signature compatibility).

        Returns:
            SavedClip metadata if save was started, None if recorder busy/stopped.
        """
        if not self._running:
            logger.warning("on_shot called but recorder is not running")
            return None

        with self._lock:
            if self._status == DTLCameraStatus.SAVING:
                logger.warning("Clip save already in progress, skipping shot")
                return None

            self._shot_count += 1
            shot_number = self._shot_count

        trigger_ts = datetime.now()
        clip_filename = f"dtl_shot{shot_number:03d}_{trigger_ts.strftime('%H%M%S')}.h264"
        clip_path = self.clip_dir / clip_filename

        self._set_status(DTLCameraStatus.SAVING)

        # Save in background so shot_callback returns fast
        self._save_thread = threading.Thread(
            target=self._save_clip,
            args=(clip_path, shot_number, trigger_ts),
            daemon=True,
        )
        self._save_thread.start()

        clip = SavedClip(
            path=clip_path,
            shot_number=shot_number,
            trigger_time=trigger_ts.isoformat(),
            pre_seconds=self.config.pre_trigger_seconds,
            post_seconds=self.config.post_trigger_seconds,
            resolution=f"{self.config.width}x{self.config.height}",
            framerate=self.config.framerate,
        )
        return clip

    def _save_clip(self, clip_path: Path, shot_number: int, trigger_ts: datetime):
        """Background worker: flush ring buffer + record post-trigger, then finalize."""
        try:
            self._circular.fileoutput = str(clip_path)
            self._circular.start()

            # Post-trigger recording with cooperative shutdown check
            end_time = time.monotonic() + self.config.post_trigger_seconds
            while time.monotonic() < end_time and self._running:
                time.sleep(0.1)

            self._circular.stop()

            file_size = clip_path.stat().st_size
            clip = SavedClip(
                path=clip_path,
                shot_number=shot_number,
                trigger_time=trigger_ts.isoformat(),
                pre_seconds=self.config.pre_trigger_seconds,
                post_seconds=self.config.post_trigger_seconds,
                resolution=f"{self.config.width}x{self.config.height}",
                framerate=self.config.framerate,
                file_size_bytes=file_size,
            )
            self._clips.append(clip)
            logger.info(
                "Clip saved: %s (%.1f KB, shot #%d)",
                clip_path.name,
                file_size / 1024,
                shot_number,
            )

            if self._clip_callback:
                try:
                    self._clip_callback(clip)
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.warning("Clip callback failed", exc_info=True)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to save clip for shot #%d: %s", shot_number, exc)

        finally:
            if self._running:
                self._set_status(DTLCameraStatus.BUFFERING)
            else:
                self._set_status(DTLCameraStatus.IDLE)

    @property
    def is_running(self) -> bool:
        """Whether the recorder is actively buffering."""
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class MockDTLCameraRecorder:
    """Mock DTL camera for testing without hardware."""

    def __init__(
        self,
        config: Optional[DTLConfig] = None,
        clip_dir: Optional[Path] = None,
        status_callback: Optional[Callable[[DTLCameraStatus], None]] = None,
        clip_callback: Optional[Callable[["SavedClip"], None]] = None,
    ):
        self.config = config or DTLConfig()
        self.clip_dir = clip_dir or Path.home() / "openflight_sessions" / "dtl_clips"
        self._status_callback = status_callback
        self._clip_callback = clip_callback
        self._status = DTLCameraStatus.IDLE
        self._running = False
        self._shot_count = 0
        self._clips: list[SavedClip] = []

    @property
    def status(self) -> DTLCameraStatus:
        """Current recorder state."""
        return self._status

    @property
    def clips(self) -> list[SavedClip]:
        """All clips saved this session."""
        return self._clips.copy()

    @property
    def is_running(self) -> bool:
        """Whether the mock recorder is active."""
        return self._running

    def _set_status(self, new_status: DTLCameraStatus):
        """Update status and notify listener."""
        self._status = new_status
        if self._status_callback:
            try:
                self._status_callback(new_status)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def start(self):
        """Start mock recording."""
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._set_status(DTLCameraStatus.BUFFERING)

    def stop(self):
        """Stop mock recording."""
        self._running = False
        self._set_status(DTLCameraStatus.IDLE)

    def on_shot(self, shot) -> Optional[SavedClip]:  # pylint: disable=unused-argument
        """Save a mock clip placeholder file."""
        if not self._running:
            return None

        self._shot_count += 1
        trigger_ts = datetime.now()
        clip_filename = f"dtl_shot{self._shot_count:03d}_{trigger_ts.strftime('%H%M%S')}.h264"
        clip_path = self.clip_dir / clip_filename

        # Write a small placeholder file
        clip_path.write_bytes(b"\x00" * 128)

        clip = SavedClip(
            path=clip_path,
            shot_number=self._shot_count,
            trigger_time=trigger_ts.isoformat(),
            pre_seconds=self.config.pre_trigger_seconds,
            post_seconds=self.config.post_trigger_seconds,
            resolution=f"{self.config.width}x{self.config.height}",
            framerate=self.config.framerate,
            file_size_bytes=128,
        )
        self._clips.append(clip)

        self._set_status(DTLCameraStatus.SAVING)
        # Immediately transition back (mock has no real delay)
        self._set_status(DTLCameraStatus.BUFFERING)

        if self._clip_callback:
            try:
                self._clip_callback(clip)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        return clip

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
