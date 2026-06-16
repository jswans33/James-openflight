"""Helpers for reporting Linux USB serial latency timer settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

USB_SERIAL_SYSFS_ROOT = Path("/sys/bus/usb-serial/devices")


@dataclass(frozen=True)
class UsbSerialLatencyTimer:
    """Kernel USB serial latency timer information for one serial port."""

    port: str
    resolved_port: str
    device_name: str
    latency_ms: Optional[int]
    sysfs_path: Path
    unavailable_reason: Optional[str] = None


def read_usb_serial_latency_timer(
    port: str,
    *,
    sysfs_root: Path = USB_SERIAL_SYSFS_ROOT,
) -> UsbSerialLatencyTimer:
    """Read Linux FTDI/USB-serial ``latency_timer`` for a port if available.

    ``port`` may be a udev alias such as ``/dev/kld7_vertical``. The helper
    resolves symlinks first, then maps the basename to:

        /sys/bus/usb-serial/devices/<ttyUSBx>/latency_timer

    The file exists for many USB-serial adapters, especially FTDI. It is not
    present on every serial backend, so missing data is returned as a normal
    unavailable result rather than an exception.
    """
    port_path = Path(port)
    resolved = port_path.resolve(strict=False)
    device_name = resolved.name or port_path.name
    sysfs_path = sysfs_root / device_name / "latency_timer"

    try:
        raw = sysfs_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return UsbSerialLatencyTimer(
            port=port,
            resolved_port=str(resolved),
            device_name=device_name,
            latency_ms=None,
            sysfs_path=sysfs_path,
            unavailable_reason="latency_timer not exposed",
        )
    except OSError as exc:
        return UsbSerialLatencyTimer(
            port=port,
            resolved_port=str(resolved),
            device_name=device_name,
            latency_ms=None,
            sysfs_path=sysfs_path,
            unavailable_reason=f"read failed: {exc}",
        )

    try:
        latency_ms = int(raw)
    except ValueError:
        return UsbSerialLatencyTimer(
            port=port,
            resolved_port=str(resolved),
            device_name=device_name,
            latency_ms=None,
            sysfs_path=sysfs_path,
            unavailable_reason=f"invalid latency_timer value: {raw!r}",
        )

    return UsbSerialLatencyTimer(
        port=port,
        resolved_port=str(resolved),
        device_name=device_name,
        latency_ms=latency_ms,
        sysfs_path=sysfs_path,
    )


def log_usb_serial_latency_timer(logger, label: str, port: str) -> UsbSerialLatencyTimer:
    """Log the kernel USB serial latency timer for a startup-connected device."""
    info = read_usb_serial_latency_timer(port)
    if info.latency_ms is None:
        logger.info(
            "[%s] USB serial latency_timer unavailable: port=%s resolved=%s (%s)",
            label,
            info.port,
            info.resolved_port,
            info.unavailable_reason,
        )
        return info

    log_fn = logger.warning if info.latency_ms > 1 else logger.info
    log_fn(
        "[%s] USB serial latency_timer=%dms port=%s resolved=%s sysfs=%s",
        label,
        info.latency_ms,
        info.port,
        info.resolved_port,
        info.sysfs_path,
    )
    return info
