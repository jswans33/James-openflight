"""
DTL (down-the-line) swing camera module.

Records swing video clips triggered by shot detection, using a continuous
ring buffer to capture pre-impact footage. Designed for the Raspberry Pi
Camera Module v3 Wide (IMX708) on CSI-0.

This is separate from the existing camera/ module which handles launch
angle detection from behind the tee.
"""

from .recorder import (
    DTLCameraRecorder,
    DTLCameraStatus,
    DTLConfig,
    MockDTLCameraRecorder,
    SavedClip,
)

__all__ = [
    "DTLCameraRecorder",
    "DTLCameraStatus",
    "DTLConfig",
    "MockDTLCameraRecorder",
    "SavedClip",
]
