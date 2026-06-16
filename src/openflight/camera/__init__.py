"""
Camera module for launch angle and spin detection.

Uses Raspberry Pi HQ Camera with IR illumination to track
golf ball trajectory from behind the tee.

Tracking options:
- BallTracker: Hough circles + ByteTrack (recommended for Pi)
- HybridBallTracker: Optional YOLO detection for better hardware
"""

from .capture import (
    CameraCapture,
    CaptureConfig,
    CapturedFrame,
    CaptureResult,
    MockCameraCapture,
)
from .detector import (
    BallDetector,
    DetectedBall,
    DetectorConfig,
)
from .launch_angle import (
    CameraCalibration,
    LaunchAngleCalculator,
    LaunchAngles,
)

try:
    from .tracker import (
        BallTracker,
        BallTrajectory,
        HybridBallTracker,
        TrackedBall,
        TrackerConfig,
    )

    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False

__all__ = [
    # Capture
    "CameraCapture",
    "MockCameraCapture",
    "CaptureConfig",
    "CapturedFrame",
    "CaptureResult",
    # Detection
    "BallDetector",
    "DetectedBall",
    "DetectorConfig",
    # Launch angle
    "LaunchAngleCalculator",
    "LaunchAngles",
    "CameraCalibration",
]

if _TRACKER_AVAILABLE:
    __all__ += [
        # Tracking
        "BallTracker",
        "HybridBallTracker",
        "TrackedBall",
        "BallTrajectory",
        "TrackerConfig",
    ]
