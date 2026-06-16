#!/usr/bin/env python3
"""
Debug radar commands to find what rolling buffer mode actually works.
"""

import sys
import time

sys.path.insert(0, "src")

from openflight.ops243 import OPS243Radar


def test_command(radar, cmd, description):
    """Send a command and print the response."""
    print(f"  {cmd:8} ({description})...")
    try:
        response = radar._send_command(cmd)
        print(f"           → {response if response else '(no response)'}")
    except Exception as e:
        print(f"           → ERROR: {e}")
    time.sleep(0.1)


def main():
    print("=" * 70)
    print("  Radar Command Debugger")
    print("=" * 70)
    print()

    radar = OPS243Radar()
    radar.connect()
    print(f"Connected: {radar.port}")

    info = radar.get_info()
    print(f"Firmware: {info.get('Version', 'unknown')}")
    print()

    # Reset to known state
    print("Resetting radar...")
    radar._send_command("PI")
    time.sleep(0.2)

    # Query available settings
    print()
    print("-" * 70)
    print("Querying current settings:")
    print("-" * 70)
    test_command(radar, "??", "all settings")

    # Test different operation modes
    print()
    print("-" * 70)
    print("Testing operation mode commands:")
    print("-" * 70)

    test_command(radar, "G?", "query mode")
    test_command(radar, "GS", "CW streaming mode")
    test_command(radar, "G?", "query mode")
    test_command(radar, "GC", "continuous sampling mode")
    test_command(radar, "G?", "query mode")

    # Try various G commands to see what's supported
    print()
    print("-" * 70)
    print("Testing other G commands:")
    print("-" * 70)

    for cmd in ["G0", "G1", "G2", "G3", "G4", "GB", "GR", "GT"]:
        test_command(radar, cmd, f"mode {cmd}")

    # Back to continuous sampling and test S!
    print()
    print("-" * 70)
    print("Testing S! trigger in continuous sampling mode:")
    print("-" * 70)

    print("  Configuring: PI → GC → PA → S=30...")
    radar._send_command("PI")
    time.sleep(0.1)
    radar._send_command("GC")
    time.sleep(0.1)
    radar._send_command("PA")
    time.sleep(0.1)
    radar._send_command("S=30")
    time.sleep(0.1)
    radar._send_command("S#8")
    time.sleep(0.5)  # Let buffer fill

    print("  Sending S!...")
    radar.serial.reset_input_buffer()
    radar.serial.write(b"S!\r")
    radar.serial.flush()

    # Read response for up to 5 seconds
    start = time.time()
    chunks = []
    while time.time() - start < 5.0:
        if radar.serial.in_waiting:
            chunk = radar.serial.read(radar.serial.in_waiting)
            chunks.append(chunk)
            if b'"Q"' in chunk:
                break
        time.sleep(0.05)

    response = b''.join(chunks).decode('utf-8', errors='ignore')
    print(f"  Response length: {len(response)} bytes")
    print(f"  Response preview: {response[:300] if response else '(empty)'}...")

    has_iq = '"I"' in response and '"Q"' in response
    print(f"  Contains I/Q data: {has_iq}")

    if not has_iq:
        print()
        print("  S! not returning I/Q data in GC mode.")
        print("  Let me try GS mode with O1 output...")

        print()
        print("-" * 70)
        print("Testing raw I/Q output mode (O1):")
        print("-" * 70)

        radar._send_command("PI")
        time.sleep(0.1)
        test_command(radar, "O?", "query output format")
        test_command(radar, "O1", "set raw I/Q output")
        test_command(radar, "O?", "query output format")
        test_command(radar, "GS", "CW streaming mode")
        test_command(radar, "PA", "activate")

        print("  Reading 2 seconds of data...")
        radar.serial.reset_input_buffer()
        start = time.time()
        chunks = []
        while time.time() - start < 2.0:
            if radar.serial.in_waiting:
                chunk = radar.serial.read(radar.serial.in_waiting)
                chunks.append(chunk)
            time.sleep(0.05)

        response = b''.join(chunks).decode('utf-8', errors='ignore')
        print(f"  Received: {len(response)} bytes")
        print(f"  Preview: {response[:300] if response else '(empty)'}...")

    radar._send_command("PI")
    radar.disconnect()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
