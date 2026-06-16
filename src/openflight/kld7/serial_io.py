"""Serial I/O helpers for the K-LD7.

Two cross-cutting concerns shared by the live tracker and the offline
capture scripts:

1. ``install_robust_read_packet`` — Replace the kld7 library's
   ``_read_packet`` with a version that loops over partial reads.
   The default reads the 8-byte header and the variable-length body
   each with a single ``serial.read(N)`` call, but at 12M USB Full
   Speed the FTDI driver splits large packets across USB microframes.
   Without the loop you get "Failed to read all of reply" on the body
   and "Wrong length reply" on the header.

2. ``connect_with_recovery`` — Open a kld7.KLD7 with retry. If a
   prior session crashed mid-stream the radar is left at 3 Mbaud and
   the next ``INIT`` at 115200 is garbled, leaving the radar in an
   undefined state that returns "Timeout waiting for reply" on the
   first command. We recover by sending a binary GBYE packet at
   3 Mbaud to cleanly close the prior session before retrying INIT.

Both helpers are idempotent and safe to call from any orientation.
"""

from __future__ import annotations

import struct
import time
from typing import Any, Optional

_MAX_PACKET_PAYLOAD_BYTES = 8192
_PARTIAL_READ_GRACE_SECONDS = 0.08
_PARTIAL_READ_RETRY_TIMEOUT_SECONDS = 0.01
_PACKET_CODES = (
    b"RESP",
    b"RADC",
    b"DONE",
    b"RPST",
    b"TDAT",
    b"PDAT",
    b"RFFT",
    b"TFFT",
    b"PFFT",
)
_MAX_STALE_RESPONSE_PACKETS = 3


def _serial_port_errors() -> tuple[type[BaseException], ...]:
    """Exception types raised by pyserial and the OS for port I/O."""
    errors: list[type[BaseException]] = [OSError]
    try:
        import serial as pyserial  # type: ignore[import-not-found]

        errors.append(pyserial.SerialException)
    except ImportError:
        pass
    return tuple(errors)


_SERIAL_PORT_ERRORS = _serial_port_errors()


def install_robust_read_packet(radar: Any) -> None:
    """Replace ``radar._read_packet`` with a short-read-tolerant version.

    Args:
        radar: A connected ``kld7.KLD7`` instance.
    """
    # Lazy import so this module is safe to import on machines where
    # the kld7 library is not installed (CI, dev laptops).
    from kld7 import KLD7Exception  # type: ignore[import-not-found]

    def _read_exact(device: Any, n: int, on_first_chunk: Optional[Any] = None) -> bytes:
        """Read exactly n bytes from the device port, looping over
        partial reads. Returns whatever was actually read if the
        underlying serial.read returns 0 bytes (timeout / EOF).
        """
        buf = b""
        remaining = n
        partial_deadline: Optional[float] = None
        first_chunk_seen = False
        port = device._port
        original_timeout = getattr(port, "timeout", None)
        using_retry_timeout = False
        try:
            while remaining > 0:
                try:
                    chunk = port.read(remaining)
                except _SERIAL_PORT_ERRORS as e:
                    raise KLD7Exception(f"Serial read failed: {e}") from e
                if not chunk:
                    if not buf:
                        break
                    now = time.monotonic()
                    if partial_deadline is None:
                        partial_deadline = now + _PARTIAL_READ_GRACE_SECONDS
                    if now >= partial_deadline:
                        break
                    if original_timeout is not None and not using_retry_timeout:
                        port.timeout = min(original_timeout, _PARTIAL_READ_RETRY_TIMEOUT_SECONDS)
                        using_retry_timeout = True
                    time.sleep(0.002)
                    continue
                if not first_chunk_seen:
                    first_chunk_seen = True
                    if on_first_chunk is not None:
                        on_first_chunk(time.time())
                buf += chunk
                remaining -= len(chunk)
                partial_deadline = None
            return buf
        finally:
            if using_retry_timeout:
                port.timeout = original_timeout

    def _resync_header(device: Any, header: bytes) -> bytes:
        """Slide past stale bytes when a valid packet code starts inside the header."""
        if header[:4] in _PACKET_CODES:
            return header
        for offset in range(1, len(header)):
            if header[offset : offset + 4] in _PACKET_CODES:
                tail = _read_exact(device, offset)
                if len(tail) != offset:
                    raise KLD7Exception(
                        f"Short header resync read: got {len(tail)} of {offset} bytes"
                    )
                return header[offset:] + tail
        return header

    def _robust_read_packet(device: Any):
        if device._port is None:
            raise KLD7Exception("serial port has been closed")
        read_started_timestamp = time.time()
        arrival_timestamp: Optional[float] = None

        def _note_arrival(timestamp: float) -> None:
            nonlocal arrival_timestamp
            if arrival_timestamp is None:
                arrival_timestamp = timestamp

        # The 8-byte header itself can be split across USB microframes
        # at 12M USB Full Speed (FTDI), so read it with the same
        # exact-length loop we use for the payload.
        header = _read_exact(device, 8, on_first_chunk=_note_arrival)
        header_complete_timestamp = time.time()
        if len(header) == 0:
            raise KLD7Exception("Timeout waiting for reply")
        if len(header) != 8:
            raise KLD7Exception(f"Short header read: got {len(header)} of 8 bytes")
        header = _resync_header(device, header)
        raw_reply, length = struct.unpack("<4sI", header)
        try:
            reply = raw_reply.decode("ASCII")
        except UnicodeDecodeError as e:
            raise KLD7Exception(f"Invalid packet header: {raw_reply!r}") from e
        if length > _MAX_PACKET_PAYLOAD_BYTES:
            raise KLD7Exception(f"Invalid packet length: {length} bytes for header {raw_reply!r}")
        if length != 0:
            payload = _read_exact(device, length)
            if len(payload) != length:
                raise KLD7Exception(f"Short payload read: got {len(payload)} of {length} bytes")
        else:
            payload = None
        complete_timestamp = time.time()
        packet_arrival_timestamp = arrival_timestamp or header_complete_timestamp
        device._openflight_last_packet_timing = {
            "reply": reply,
            "payload_bytes": length,
            "read_started_timestamp": read_started_timestamp,
            "arrival_timestamp": packet_arrival_timestamp,
            "header_complete_timestamp": header_complete_timestamp,
            "complete_timestamp": complete_timestamp,
            "read_duration_ms": (complete_timestamp - packet_arrival_timestamp) * 1000.0,
            "total_wait_ms": (complete_timestamp - read_started_timestamp) * 1000.0,
        }
        return reply, payload

    def _robust_get_response(device: Any):
        from kld7.device import Response  # type: ignore[import-not-found]

        stale_packets = 0
        while stale_packets <= _MAX_STALE_RESPONSE_PACKETS:
            reply, payload = _robust_read_packet(device)
            if reply == "RESP":
                if len(payload) != 1:
                    raise KLD7Exception("Response packet with incorrect payload length")
                code = payload[0]
                return Response(code if code < Response.MAX_RESPONSE else -1)
            stale_packets += 1
        raise KLD7Exception("Packet was not a response")

    radar._read_packet = lambda: _robust_read_packet(radar)
    radar._get_response = lambda: _robust_get_response(radar)


def _install_safe_kld7_destructor(kld7_cls: Any) -> None:
    """Patch kld7.KLD7.__del__ so serial close failures do not print tracebacks."""
    if getattr(kld7_cls, "_openflight_safe_del_installed", False):
        return

    original_del = getattr(kld7_cls, "__del__", None)

    def _safe_del(self: Any) -> None:
        try:
            if original_del is not None:
                original_del(self)
            elif hasattr(self, "close"):
                self.close()
        except Exception:
            try:
                self._port = None
            except Exception:
                pass

    kld7_cls.__del__ = _safe_del
    kld7_cls._openflight_safe_del_installed = True


# Binary GBYE packet: 4-byte command + 4-byte length (0).
_GBYE_PACKET = struct.pack("<4sI", b"GBYE", 0)


def _send_gbye_at_3mbaud(port: str, log: Optional[Any] = None) -> None:
    """Send a binary GBYE packet at 3 Mbaud to cleanly close a stuck
    prior session, then drain any response. Best-effort — failures
    are silenced (a stuck radar may also fail to respond to GBYE).
    """
    try:
        import serial as pyserial  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        with pyserial.Serial(
            port,
            3000000,
            parity=pyserial.PARITY_EVEN,
            timeout=0.1,
        ) as ser:
            ser.reset_input_buffer()
            ser.write(_GBYE_PACKET)
            ser.flush()
            time.sleep(0.3)
            while ser.in_waiting:
                ser.read(ser.in_waiting)
                time.sleep(0.1)
        if log is not None:
            log("[KLD7] Sent GBYE at 3Mbaud to reset prior session")
    except _SERIAL_PORT_ERRORS as e:
        if log is not None:
            log(f"[KLD7] GBYE flush failed: {e}")


def connect_with_recovery(
    port: str,
    baudrate: int = 3_000_000,
    max_attempts: int = 5,
    log: Optional[Any] = None,
) -> Any:
    """Open a kld7.KLD7 instance, recovering from a stuck prior
    session.

    The kld7 library always opens at 115200 and negotiates up via
    INIT. If a previous session left the radar streaming at 3 Mbaud,
    the INIT is garbled and the radar enters an undefined state in
    which the first command times out. We recover by sending a
    binary GBYE packet at 3 Mbaud between attempts.

    Args:
        port: Serial port path (e.g. ``/dev/ttyUSB0``).
        baudrate: Target baud after INIT negotiation. Default 3 Mbaud.
        max_attempts: Number of connect attempts before giving up.
        log: Optional callable accepting a single string for progress
            messages (use ``logger.info`` or ``print``).

    Returns:
        A connected ``kld7.KLD7`` instance with the robust _read_packet
        patch already installed.

    Raises:
        Exception: re-raises the last underlying connection error if
            all attempts fail.
    """
    from kld7 import KLD7  # type: ignore[import-not-found]

    _install_safe_kld7_destructor(KLD7)

    last_err: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            radar = KLD7(port, baudrate=baudrate)
            install_robust_read_packet(radar)
            if log is not None:
                log(
                    f"[KLD7] Connected on {port} at {baudrate} baud "
                    f"(attempt {attempt}/{max_attempts})"
                )
            return radar
        except _SERIAL_PORT_ERRORS as e:
            last_err = e
            failure_kind = "serial"
        except Exception as e:  # pylint: disable=broad-except
            last_err = e
            failure_kind = "library"

        if log is not None:
            log(
                f"[KLD7] Connect attempt {attempt}/{max_attempts} "
                f"failed ({failure_kind}): {last_err}"
            )
        if attempt >= max_attempts:
            break
        _send_gbye_at_3mbaud(port, log=log)
        time.sleep(0.3)

    # All attempts failed.
    if last_err is not None:
        raise last_err
    raise RuntimeError("connect_with_recovery: no attempts ran")
