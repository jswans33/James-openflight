"""Tests for the guarded K-LD7 timing probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hardware-test" / "probe_kld7_timing.py"
spec = importlib.util.spec_from_file_location("probe_kld7_timing", SCRIPT)
probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


class FakeSerial:
    """Small serial fake that can return deliberately split chunks."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.writes = []
        self.timeout = 0.2
        self.baudrate = 115200
        self.closed = False

    def read(self, n):
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > n:
            self.chunks.insert(0, chunk[n:])
            return chunk[:n]
        return chunk

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        return None

    def reset_input_buffer(self):
        return None

    def close(self):
        self.closed = True


def test_build_packet_uppercases_command_and_packs_length():
    packet = probe.build_packet("gnfd", (0x21).to_bytes(4, "little"))

    assert packet == b"GNFD\x04\x00\x00\x00!\x00\x00\x00"


def test_validate_command_rejects_non_four_byte_command():
    assert probe.validate_probe_command("ABC", "") == "command must be exactly 4 ASCII characters"


def test_validate_command_rejects_non_uppercase_command():
    assert probe.validate_probe_command("test", "") == "command must be uppercase ASCII"


def test_validate_command_rejects_odd_hex_payload():
    assert (
        probe.validate_probe_command("TST1", "abc")
        == "hex payload must have an even number of characters"
    )


def test_read_packet_handles_split_header_and_payload():
    payload = b"\x00\x01\x02\x03"
    header = b"DONE" + len(payload).to_bytes(4, "little")
    fake = FakeSerial([header[:3], header[3:8], payload[:1], payload[1:]])
    protocol = probe.KLD7Protocol.__new__(probe.KLD7Protocol)
    protocol.port = fake

    packet = protocol.read_packet()

    assert packet.code == "DONE"
    assert packet.payload == payload
    assert packet.payload_bytes == 4


def test_summarize_measurements_counts_done_gaps():
    packets = [
        probe.PacketRecord(code="DONE", payload_bytes=4, complete_monotonic=1.0, done_frame=10),
        probe.PacketRecord(
            code="RADC", payload_bytes=3072, complete_monotonic=1.1, read_duration_ms=10.0
        ),
        probe.PacketRecord(code="DONE", payload_bytes=4, complete_monotonic=2.0, done_frame=12),
        probe.PacketRecord(
            code="RADC", payload_bytes=3072, complete_monotonic=2.1, read_duration_ms=20.0
        ),
    ]

    summary = probe.summarize_packets(packets, duration_s=2.0)

    assert summary["radc_frames"] == 2
    assert summary["done_frames"] == 2
    assert summary["done_frame_gaps"] == 1
    assert summary["effective_radc_hz"] == 1.0
    assert summary["read_duration_ms_p95"] == 20.0


def test_unsafe_probe_requires_output():
    parser = probe.build_parser()
    args = parser.parse_args(["--port", "/dev/null", "--unsafe-probe", "--probe-command", "TEST"])

    assert probe.validate_args(args) == [
        "--unsafe-probe requires --output so probe activity is auditable"
    ]


def test_rfse_requires_factory_reset_flag():
    parser = probe.build_parser()
    args = parser.parse_args(
        [
            "--port",
            "/dev/null",
            "--output",
            "/tmp/probe.jsonl",
            "--unsafe-probe",
            "--probe-command",
            "RFSE",
        ]
    )

    assert probe.validate_args(args) == ["RFSE is refused unless --allow-factory-reset is set"]


def test_parse_frame_mask_combines_known_flags():
    assert probe.parse_frame_mask("RADC,DONE") == 0x21


def test_probe_command_requires_unsafe_probe():
    parser = probe.build_parser()
    args = parser.parse_args(["--port", "/dev/null", "--probe-command", "TEST"])

    assert probe.validate_args(args) == ["--probe-command requires --unsafe-probe"]


def test_rfse_allowed_when_factory_reset_flag_is_present():
    parser = probe.build_parser()
    args = parser.parse_args(
        [
            "--port",
            "/dev/null",
            "--output",
            "/tmp/probe.jsonl",
            "--unsafe-probe",
            "--allow-factory-reset",
            "--probe-command",
            "RFSE",
        ]
    )

    assert probe.validate_args(args) == []


def test_send_command_records_response_code_and_written_packet():
    payload = b"\x00"
    header = b"RESP" + len(payload).to_bytes(4, "little")
    fake = FakeSerial([header, payload])
    protocol = probe.KLD7Protocol.__new__(probe.KLD7Protocol)
    protocol.port = fake

    response = protocol.send_command("TEST", b"\x01\x02")

    assert fake.writes == [b"TEST\x02\x00\x00\x00\x01\x02"]
    assert response.command == "TEST"
    assert response.response_code == 0


def test_parse_probe_command_decodes_hex_payload():
    assert probe.parse_probe_command("TEST:0102ff") == ("TEST", b"\x01\x02\xff")


def test_write_jsonl_strips_payload_bytes(tmp_path):
    output = tmp_path / "probe.jsonl"
    records = [probe.PacketRecord(code="RADC", payload_bytes=3072, payload=b"\x00\x01")]
    summary = {"radc_frames": 1}

    probe.write_jsonl(output, records, summary)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert '"payload"' not in lines[0]
    assert '"type": "packet"' in lines[0]
    assert '"type": "summary"' in lines[1]
