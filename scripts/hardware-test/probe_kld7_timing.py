#!/usr/bin/env python3
"""Guarded K-LD7 timing and protocol probe."""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

try:
    import serial
    from serial.tools.list_ports import comports
except ImportError:  # pragma: no cover - operator environment issue
    serial = None
    comports = None


DEFAULT_BAUD = 3_000_000
DEFAULT_START_BAUD = 115_200
SUPPORTED_BAUD_RATES = [115_200, 460_800, 921_600, 2_000_000, 3_000_000]
MAX_PACKET_PAYLOAD_BYTES = 8192
FRAME_CODES = {
    "RADC": 0x01,
    "RFFT": 0x02,
    "PDAT": 0x04,
    "TDAT": 0x08,
    "DDAT": 0x10,
    "DONE": 0x20,
}
DOCUMENTED_WRITE_COMMANDS = {
    "RBFR",
    "RSPI",
    "RRAI",
    "THOF",
    "TRFT",
    "VISU",
    "MIRA",
    "MARA",
    "MIAN",
    "MAAN",
    "MISP",
    "MASP",
    "DEDI",
    "RATH",
    "ANTH",
    "SPTH",
    "DIG1",
    "DIG2",
    "DIG3",
    "HOLD",
    "MIDE",
    "MIDS",
}
DESTRUCTIVE_COMMANDS = {"RFSE"}
PARAM_STRUCT_FORMAT = "<19s8B2b4Bb4BH2B"
PARAM_FIELDS = [
    "software_version",
    "RBFR",
    "RSPI",
    "RRAI",
    "THOF",
    "TRFT",
    "VISU",
    "MIRA",
    "MARA",
    "MIAN",
    "MAAN",
    "MISP",
    "MASP",
    "DEDI",
    "RATH",
    "ANTH",
    "SPTH",
    "DIG1",
    "DIG2",
    "DIG3",
    "HOLD",
    "MIDE",
    "MIDS",
]


@dataclass
class PacketRecord:
    """One packet or packet-level error observed while probing."""

    code: str
    payload_bytes: int
    command: Optional[str] = None
    response_code: Optional[int] = None
    send_monotonic: Optional[float] = None
    first_byte_monotonic: Optional[float] = None
    header_complete_monotonic: Optional[float] = None
    complete_monotonic: Optional[float] = None
    read_duration_ms: Optional[float] = None
    done_frame: Optional[int] = None
    error: Optional[str] = None
    payload: bytes = field(default=b"", repr=False)


def build_packet(command: str, payload: bytes = b"") -> bytes:
    """Build a K-LD7 binary command packet."""
    cmd = command.upper().encode("ascii")
    if len(cmd) != 4:
        raise ValueError("command must be exactly 4 ASCII characters")
    return struct.pack("<4sI", cmd, len(payload)) + payload


def validate_probe_command(command: str, hex_payload: str) -> Optional[str]:
    """Validate one explicit unsafe probe command specification."""
    try:
        raw = command.encode("ascii")
    except UnicodeEncodeError:
        return "command must be ASCII"
    if len(raw) != 4:
        return "command must be exactly 4 ASCII characters"
    if command != command.upper():
        return "command must be uppercase ASCII"
    if len(hex_payload) % 2:
        return "hex payload must have an even number of characters"
    try:
        bytes.fromhex(hex_payload)
    except ValueError:
        return "hex payload must be valid hexadecimal"
    return None


def parse_frame_mask(value: str) -> int:
    """Parse a comma-separated frame mask such as RADC,DONE."""
    mask = 0
    for raw_name in value.split(","):
        name = raw_name.strip().upper()
        if not name:
            continue
        if name not in FRAME_CODES:
            raise argparse.ArgumentTypeError(f"unknown frame type {name!r}")
        mask |= FRAME_CODES[name]
    if mask == 0:
        raise argparse.ArgumentTypeError("at least one frame type is required")
    return mask


def _response_name(code: Optional[int]) -> Optional[str]:
    names = {
        0: "OK",
        1: "UnknownCommand",
        2: "InvalidParameter",
        3: "InvalidRPSTVersion",
        4: "UARTError",
        5: "SensorBusy",
    }
    return names.get(code)


class KLD7Protocol:
    """Small direct serial protocol wrapper for K-LD7 packets."""

    def __init__(self, port_path: str, baud: int = DEFAULT_BAUD, timeout: float = 0.2):
        if serial is None:
            raise RuntimeError("pyserial is required for hardware probing")
        self.port_path = port_path
        self.baud = baud
        self.port = serial.Serial(
            port=port_path,
            baudrate=DEFAULT_START_BAUD,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        self._init_device()

    def _init_device(self) -> None:
        """Negotiate the requested baud rate, recovering once from a stuck 3M session."""
        try:
            self._send_init()
        except Exception:
            self._send_gbye_at_3mbaud()
            self._send_init()

    def _send_init(self) -> None:
        if self.baud not in SUPPORTED_BAUD_RATES:
            raise ValueError(f"unsupported baud rate: {self.baud}")
        response = self.send_command(
            "INIT",
            SUPPORTED_BAUD_RATES.index(self.baud).to_bytes(4, "little", signed=True),
        )
        if response.response_code != 0:
            name = _response_name(response.response_code) or response.response_code
            raise RuntimeError(f"INIT failed: {name}")
        if self.baud != DEFAULT_START_BAUD:
            self.port.baudrate = self.baud

    def _send_gbye_at_3mbaud(self) -> None:
        try:
            self.port.baudrate = DEFAULT_BAUD
            self.port.reset_input_buffer()
            self.port.write(build_packet("GBYE"))
            self.port.flush()
            time.sleep(0.3)
        finally:
            self.port.baudrate = DEFAULT_START_BAUD

    def _read_exact(self, n: int) -> tuple[bytes, Optional[float]]:
        buf = b""
        first_byte_at = None
        timeout = max(float(getattr(self.port, "timeout", 0.2) or 0.2), 0.2)
        deadline = time.monotonic() + timeout
        while len(buf) < n:
            chunk = self.port.read(n - len(buf))
            if chunk:
                if first_byte_at is None:
                    first_byte_at = time.monotonic()
                buf += chunk
                deadline = time.monotonic() + timeout
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.001)
        return buf, first_byte_at

    def read_packet(self) -> PacketRecord:
        started = time.monotonic()
        header, first_byte_at = self._read_exact(8)
        header_complete = time.monotonic()
        if len(header) != 8:
            return PacketRecord(
                code="",
                payload_bytes=0,
                first_byte_monotonic=first_byte_at,
                header_complete_monotonic=header_complete,
                complete_monotonic=header_complete,
                read_duration_ms=(header_complete - started) * 1000.0,
                error=f"short header read: got {len(header)} of 8 bytes",
            )
        raw_code, length = struct.unpack("<4sI", header)
        code = raw_code.decode("ascii", errors="replace")
        if length > MAX_PACKET_PAYLOAD_BYTES:
            complete = time.monotonic()
            return PacketRecord(
                code=code,
                payload_bytes=length,
                first_byte_monotonic=first_byte_at,
                header_complete_monotonic=header_complete,
                complete_monotonic=complete,
                read_duration_ms=(complete - (first_byte_at or started)) * 1000.0,
                error=f"invalid payload length: {length} bytes",
            )
        payload = b""
        payload_first = None
        if length:
            payload, payload_first = self._read_exact(length)
        complete = time.monotonic()
        error = None
        if len(payload) != length:
            error = f"short payload read: got {len(payload)} of {length} bytes"
        done_frame = None
        if code == "DONE" and len(payload) == 4:
            done_frame = int.from_bytes(payload, "little", signed=False)
        return PacketRecord(
            code=code,
            payload_bytes=length,
            first_byte_monotonic=first_byte_at or payload_first,
            header_complete_monotonic=header_complete,
            complete_monotonic=complete,
            read_duration_ms=(complete - (first_byte_at or started)) * 1000.0,
            done_frame=done_frame,
            error=error,
            payload=payload,
        )

    def send_command(self, command: str, payload: bytes = b"") -> PacketRecord:
        sent = time.monotonic()
        self.port.reset_input_buffer()
        self.port.write(build_packet(command, payload))
        self.port.flush()
        packet = self.read_packet()
        packet.command = command.upper()
        packet.send_monotonic = sent
        if packet.code == "RESP" and packet.payload:
            packet.response_code = packet.payload[0]
        return packet

    def request_frame(self, frame_mask: int) -> list[PacketRecord]:
        records = [self.send_command("GNFD", int(frame_mask).to_bytes(4, "little", signed=True))]
        expected = bin(frame_mask).count("1")
        for _ in range(expected):
            record = self.read_packet()
            records.append(record)
            if record.code == "DONE":
                break
        return records

    def read_params(self) -> tuple[dict, list[PacketRecord]]:
        response = self.send_command("GRPS")
        records = [response]
        if response.response_code != 0:
            return {}, records
        packet = self.read_packet()
        records.append(packet)
        if packet.code != "RPST" or packet.error:
            return {}, records
        values = struct.unpack(PARAM_STRUCT_FORMAT, packet.payload)
        params = {}
        for name, value in zip(PARAM_FIELDS, values):
            if name == "software_version":
                params[name] = value.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            else:
                params[name] = int(value)
        return params, records

    def set_param(self, command: str, value: int) -> PacketRecord:
        if command not in DOCUMENTED_WRITE_COMMANDS:
            raise ValueError(f"not a documented parameter command: {command}")
        return self.send_command(command, int(value).to_bytes(4, "little", signed=True))

    def restore_params(self, params: dict) -> list[PacketRecord]:
        records = []
        for command in PARAM_FIELDS[1:]:
            if command in params:
                records.append(self.set_param(command, int(params[command])))
        return records

    def close(self) -> None:
        try:
            self.port.write(build_packet("GBYE"))
            self.port.flush()
        finally:
            self.port.close()


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((percentile / 100.0) * (len(values) - 1))))
    return values[index]


def summarize_packets(packets: list[PacketRecord], duration_s: float) -> dict:
    """Summarize packet-level timing and frame continuity."""
    radc_packets = [p for p in packets if p.code == "RADC" and not p.error]
    done_packets = [p for p in packets if p.code == "DONE" and not p.error]
    done_frames = [p.done_frame for p in done_packets if p.done_frame is not None]
    gaps = 0
    for previous, current in zip(done_frames, done_frames[1:]):
        if current > previous + 1:
            gaps += current - previous - 1
    read_durations = [p.read_duration_ms for p in radc_packets if p.read_duration_ms is not None]
    errors: dict[str, int] = {}
    for packet in packets:
        if packet.error:
            errors[packet.error] = errors.get(packet.error, 0) + 1
    return {
        "duration_s": duration_s,
        "radc_frames": len(radc_packets),
        "done_frames": len(done_packets),
        "effective_radc_hz": round(len(radc_packets) / duration_s, 3) if duration_s > 0 else 0.0,
        "effective_done_hz": round(len(done_packets) / duration_s, 3) if duration_s > 0 else 0.0,
        "done_frame_gaps": gaps,
        "read_duration_ms_mean": statistics.mean(read_durations) if read_durations else None,
        "read_duration_ms_p50": statistics.median(read_durations) if read_durations else None,
        "read_duration_ms_p95": _percentile(read_durations, 95),
        "errors": errors,
    }


def parse_probe_command(command_spec: str) -> tuple[str, bytes]:
    command, _, hex_payload = command_spec.partition(":")
    return command, bytes.fromhex(hex_payload)


def find_kld7_port() -> Optional[str]:
    """Return a K-LD7-like serial port only when exactly one candidate is present."""
    if comports is None:
        return None
    matches = []
    for port in comports():
        desc = (port.description or "").lower()
        mfg = (port.manufacturer or "").lower()
        if any(kw in desc for kw in ["ftdi", "cp210", "usb-serial", "uart"]) or any(
            kw in mfg for kw in ["ftdi", "silicon labs"]
        ):
            matches.append(port.device)
    return matches[0] if len(matches) == 1 else None


def measure(protocol: KLD7Protocol, frame_mask: int, duration_s: float) -> list[PacketRecord]:
    """Request frames until the duration expires."""
    records = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        records.extend(protocol.request_frame(frame_mask))
    return records


def apply_documented_settings(
    protocol: KLD7Protocol, args: argparse.Namespace
) -> list[PacketRecord]:
    records = []
    if args.rrai is not None:
        records.append(protocol.set_param("RRAI", args.rrai))
    if args.rbfr is not None:
        records.append(protocol.set_param("RBFR", args.rbfr))
    return records


def run_unsafe_probes(protocol: KLD7Protocol, args: argparse.Namespace) -> list[PacketRecord]:
    records = []
    for command_spec in args.probe_command:
        command, payload = parse_probe_command(command_spec)
        records.append(protocol.send_command(command, payload))
        _params, param_records = protocol.read_params()
        records.extend(param_records)
    return records


def write_jsonl(path: Path, records: Iterable[PacketRecord], summary: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            data = asdict(record)
            data.pop("payload", None)
            handle.write(json.dumps({"type": "packet", **data}, sort_keys=True) + "\n")
        handle.write(json.dumps({"type": "summary", **summary}, sort_keys=True) + "\n")


def _run_one_configuration(
    args: argparse.Namespace, rspi: Optional[int] = None
) -> tuple[list, dict]:
    port = args.port or find_kld7_port()
    if not port:
        raise RuntimeError(
            "--port is required unless exactly one K-LD7-like serial port is present"
        )
    frame_mask = parse_frame_mask(args.frame_mask)
    protocol = KLD7Protocol(port, baud=args.baud)
    records: list[PacketRecord] = []
    params_before: dict = {}
    try:
        params_before, param_records = protocol.read_params()
        records.extend(param_records)
        records.extend(apply_documented_settings(protocol, args))
        if rspi is not None:
            records.append(protocol.set_param("RSPI", rspi))
        params_active, param_records = protocol.read_params()
        records.extend(param_records)
        if args.unsafe_probe:
            records.extend(run_unsafe_probes(protocol, args))
        measurement_records = measure(protocol, frame_mask, args.duration)
        records.extend(measurement_records)
        summary = summarize_packets(measurement_records, args.duration)
        summary.update(
            {
                "port": port,
                "baud": args.baud,
                "frame_mask": [name for name, value in FRAME_CODES.items() if frame_mask & value],
                "params_before": params_before,
                "params_active": params_active,
            }
        )
        if rspi is not None:
            summary["rspi_sweep_value"] = rspi
        return records, summary
    finally:
        if params_before and not args.no_restore_params:
            records.extend(protocol.restore_params(params_before))
        protocol.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe K-LD7 RADC timing and guarded commands.")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, choices=SUPPORTED_BAUD_RATES)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--frame-mask", default="RADC,DONE")
    parser.add_argument("--rspi-sweep", action="store_true")
    parser.add_argument("--rrai", type=int)
    parser.add_argument("--rbfr", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unsafe-probe", action="store_true")
    parser.add_argument("--probe-command", action="append", default=[])
    parser.add_argument("--allow-factory-reset", action="store_true")
    parser.add_argument("--no-restore-params", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    errors = []
    if args.duration <= 0:
        errors.append("--duration must be positive")
    if args.unsafe_probe and not args.output:
        errors.append("--unsafe-probe requires --output so probe activity is auditable")
    if args.probe_command and not args.unsafe_probe:
        errors.append("--probe-command requires --unsafe-probe")
    if args.no_restore_params and not args.unsafe_probe:
        errors.append("--no-restore-params requires --unsafe-probe")
    try:
        parse_frame_mask(args.frame_mask)
    except argparse.ArgumentTypeError as exc:
        errors.append(str(exc))
    for command_spec in args.probe_command:
        command, _, hex_payload = command_spec.partition(":")
        error = validate_probe_command(command, hex_payload)
        if error:
            errors.append(f"{command_spec}: {error}")
        if command.upper() in DESTRUCTIVE_COMMANDS and not args.allow_factory_reset:
            errors.append("RFSE is refused unless --allow-factory-reset is set")
    return errors


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    errors = validate_args(args)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2
    all_records = []
    summaries = []
    try:
        rspi_values: Iterable[Optional[int]] = range(4) if args.rspi_sweep else [None]
        for rspi in rspi_values:
            records, summary = _run_one_configuration(args, rspi)
            all_records.extend(records)
            summaries.append(summary)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output = summaries[0] if len(summaries) == 1 else {"runs": summaries}
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output:
        write_jsonl(args.output, all_records, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
