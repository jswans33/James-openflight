#!/usr/bin/env python3
"""
GPIO-assisted sound trigger tester for SEN-14262 + OPS243-A.

Uses Pi GPIO to detect the SEN-14262 GATE signal, then sends S! command
to trigger the radar capture. This works around voltage level issues
where GATE doesn't reach the 3.3V threshold required by HOST_INT.

Wiring:
    SEN-14262 VCC  → 3.3V (Pi or radar)
    SEN-14262 GND  → GND
    SEN-14262 GATE → Pi GPIO pin (default: GPIO17, physical pin 11)

How it works:
    1. Pi GPIO detects rising edge on GATE (lower voltage threshold than HOST_INT)
    2. Python sends S! command to radar to trigger buffer dump
    3. Script reads and processes I/Q data

Usage:
    uv run python scripts/test_sound_trigger_gpio.py
    uv run python scripts/test_sound_trigger_gpio.py --gpio 17
    uv run python scripts/test_sound_trigger_gpio.py --swings-only
"""

import argparse
import sys
import time

# Add src to path so we can import openflight
sys.path.insert(0, "src")

# Try to import GPIO libraries
GPIO_LIB = None
try:
    from gpiozero import Button
    GPIO_LIB = "gpiozero"
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO_LIB = "rpigpio"
    except ImportError:
        pass

if not GPIO_LIB:
    print("WARNING: No GPIO library available.")
    print("Install with: pip install gpiozero  OR  sudo apt install python3-rpi-lgpio")

from openflight.ops243 import OPS243Radar  # noqa: E402
from openflight.rolling_buffer.processor import RollingBufferProcessor  # noqa: E402

# Radar constants (AN-027)
SAMPLES_PER_SEGMENT = 128
TOTAL_SEGMENTS = 32
SEGMENT_DURATION_MS = SAMPLES_PER_SEGMENT / 30000 * 1000  # ~4.27ms at 30ksps

# Default GPIO pin for GATE input (BCM numbering)
DEFAULT_GPIO_PIN = 17  # Physical pin 11


def process_capture(processor: RollingBufferProcessor, response: str,
                    trigger_count: int, transfer_ms: float,
                    pre_trigger_segments: int, swings_only: bool = False):
    """
    Process the I/Q buffer dump and display results.
    """
    pre_ms = pre_trigger_segments * SEGMENT_DURATION_MS
    post_ms = (TOTAL_SEGMENTS - pre_trigger_segments) * SEGMENT_DURATION_MS

    print()
    print("=" * 60)
    print(f"  TRIGGER #{trigger_count}")
    print("=" * 60)
    print(f"  Capture window: {pre_ms:.1f}ms pre / {post_ms:.1f}ms post (S#{pre_trigger_segments})")
    print(f"  Buffer received: {len(response)} bytes in {transfer_ms:.0f}ms")

    # Parse the I/Q response
    capture = processor.parse_capture(response)

    if capture is None:
        print("  WARNING: Failed to parse I/Q data from buffer dump")
        print("=" * 60)
        print()
        return False

    print(f"  I/Q samples: {len(capture.i_samples)} I, {len(capture.q_samples)} Q")

    # Run FFT processing
    timeline = processor.process_standard(capture)

    print()
    print(f"  Processed {len(timeline.readings)} speed readings across 32 segments:")
    print()

    # Show readings grouped by segment like AN-027 Appendix A output
    outbound_readings = []
    for reading in timeline.readings:
        direction = "OUT" if reading.is_outbound else " IN"
        marker = " <--" if reading.is_outbound and reading.speed_mph >= 15.0 else ""
        print(f"    Seg {reading.segment_index:>2}: {reading.speed_mph:>6.1f} mph "
              f"{direction} (mag: {reading.magnitude:>6.1f}){marker}")
        if reading.is_outbound and reading.speed_mph >= 15.0:
            outbound_readings.append(reading)

    print()
    if not outbound_readings:
        if swings_only:
            print("  No swing detected — discarding (false trigger / nearby player)")
        else:
            print("  No swing detected (false trigger / nearby player)")
        print("=" * 60)
        print()
        return False

    peak_reading = max(outbound_readings, key=lambda r: r.speed_mph)
    print(f"  Peak outbound:  {peak_reading.speed_mph:.1f} mph "
          f"(segment {peak_reading.segment_index}, mag {peak_reading.magnitude:.1f})")
    print(f"  Outbound count: {len(outbound_readings)} readings >= 15 mph")
    print("=" * 60)
    print()
    return True


def run(port: str, gpio_pin: int, pre_trigger_segments: int,
        swings_only: bool, debounce_ms: int):
    """Connect to radar, configure rolling buffer, wait for GPIO triggers."""

    if not GPIO_LIB:
        print("ERROR: GPIO library required for GPIO-assisted triggering.")
        print("Install with: pip install gpiozero")
        sys.exit(1)

    # --- Set up GPIO ---
    print(f"Setting up GPIO{gpio_pin} for GATE input (using {GPIO_LIB})...")

    # Track trigger events
    trigger_event = {"triggered": False}

    if GPIO_LIB == "gpiozero":
        # gpiozero approach - simpler and more compatible
        button = Button(gpio_pin, pull_up=False, bounce_time=debounce_ms / 1000.0)

        def on_trigger():
            trigger_event["triggered"] = True

        button.when_pressed = on_trigger
        print(f"GPIO{gpio_pin} configured with gpiozero (pull-down, bounce={debounce_ms}ms)")
    else:
        # RPi.GPIO approach
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        print(f"GPIO{gpio_pin} configured with RPi.GPIO (pull-down)")

    # --- Connect to radar ---
    print("Connecting to radar...")
    radar = OPS243Radar(port=port if port else None)
    radar.connect()
    print(f"Connected on {radar.port}")

    # Get firmware info
    info = radar.get_info()
    version = info.get("Version", "unknown")
    print(f"Firmware: {version}")

    # --- Configure rolling buffer mode ---
    print(f"Configuring rolling buffer mode (S#{pre_trigger_segments})...")
    radar.configure_for_rolling_buffer(pre_trigger_segments=pre_trigger_segments)

    # --- Set up FFT processor ---
    processor = RollingBufferProcessor()

    print()
    print(f"Pre-trigger: S#{pre_trigger_segments} "
          f"({pre_trigger_segments * SEGMENT_DURATION_MS:.1f}ms pre, "
          f"{(TOTAL_SEGMENTS - pre_trigger_segments) * SEGMENT_DURATION_MS:.1f}ms post)")
    print(f"Mode: {'swings only' if swings_only else 'all captures'}")
    print(f"Debounce: {debounce_ms}ms")
    print()
    print(f"Wiring: SEN-14262 GATE → GPIO{gpio_pin} (physical pin {gpio_to_physical(gpio_pin)})")
    print()
    print("Waiting for sound trigger on GPIO... (Ctrl+C to quit)")
    print()

    trigger_count = 0
    swing_count = 0
    last_trigger_time = 0

    try:
        while True:
            # Wait for rising edge on GPIO (GATE going HIGH)
            if GPIO_LIB == "gpiozero":
                # gpiozero - check for trigger event
                if not trigger_event["triggered"]:
                    time.sleep(0.1)
                    # Print dot every second
                    if int(time.time()) % 10 == 0 and int(time.time() * 10) % 10 == 0:
                        print(".", end="", flush=True)
                    continue
                trigger_event["triggered"] = False
            else:
                # RPi.GPIO - wait for edge
                channel = GPIO.wait_for_edge(gpio_pin, GPIO.RISING, timeout=1000)
                if channel is None:
                    print(".", end="", flush=True)
                    continue

            # Debounce - ignore triggers too close together (for RPi.GPIO)
            now = time.time()
            if GPIO_LIB != "gpiozero" and (now - last_trigger_time) * 1000 < debounce_ms:
                continue
            last_trigger_time = now

            print()
            print("GPIO edge detected! Triggering radar capture...")

            # Send S! to trigger the radar capture
            start_time = time.time()
            response = radar.trigger_capture(timeout=5.0)
            transfer_ms = (time.time() - start_time) * 1000

            if not response:
                print("WARNING: No response from radar after S! trigger")
                radar.rearm_rolling_buffer()
                continue

            trigger_count += 1

            # Re-arm for next capture
            radar.rearm_rolling_buffer()

            # Process and display
            is_swing = process_capture(
                processor, response, trigger_count, transfer_ms,
                pre_trigger_segments, swings_only
            )

            if is_swing:
                swing_count += 1

            print(f"Waiting for next trigger... ({trigger_count} triggers, {swing_count} swings)")
            print()

    except KeyboardInterrupt:
        print()
        print()
        print(f"Done. {trigger_count} triggers, {swing_count} swings detected.")

    finally:
        print("Cleaning up GPIO...")
        if GPIO_LIB == "gpiozero":
            button.close()
        elif GPIO_LIB == "rpigpio":
            GPIO.cleanup()
        print("Disconnecting radar...")
        radar.disable_rolling_buffer()
        radar.disconnect()
        print("Done.")


def gpio_to_physical(bcm_pin: int) -> int:
    """Convert BCM GPIO number to physical pin number (for common pins)."""
    bcm_to_physical = {
        2: 3, 3: 5, 4: 7, 17: 11, 27: 13, 22: 15,
        10: 19, 9: 21, 11: 23, 5: 29, 6: 31, 13: 33,
        19: 35, 26: 37, 14: 8, 15: 10, 18: 12, 23: 16,
        24: 18, 25: 22, 8: 24, 7: 26, 12: 32, 16: 36,
        20: 38, 21: 40,
    }
    return bcm_to_physical.get(bcm_pin, 0)


def main():
    parser = argparse.ArgumentParser(
        description="GPIO-assisted sound trigger test for SEN-14262 + OPS243-A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Wiring:
  SEN-14262 VCC  → 3.3V (Pi or radar)
  SEN-14262 GND  → GND (shared with Pi)
  SEN-14262 GATE → Pi GPIO17 (physical pin 11)

How it works:
  This script uses Pi GPIO to detect the GATE signal instead of relying
  on the radar's HOST_INT pin. The Pi GPIO has a lower voltage threshold
  (~1.8V) compared to HOST_INT (~2.0V), making it more reliable when
  GATE doesn't reach full 3.3V.

  When GATE goes HIGH (sound detected):
    1. Pi GPIO detects the rising edge
    2. Script sends S! command to radar
    3. Radar dumps rolling buffer
    4. Script processes I/Q data

Examples:
  # Default settings (GPIO17)
  uv run python scripts/test_sound_trigger_gpio.py

  # Use a different GPIO pin
  uv run python scripts/test_sound_trigger_gpio.py --gpio 27

  # Only show real swings
  uv run python scripts/test_sound_trigger_gpio.py --swings-only

  # Increase debounce time for noisy environments
  uv run python scripts/test_sound_trigger_gpio.py --debounce 500

Tips:
  - The script enables internal pull-down on the GPIO pin
  - Adjust SEN-14262 pot to set trigger sensitivity
  - Use --debounce to filter rapid false triggers
        """,
    )
    parser.add_argument(
        "--port", help="Serial port for radar (auto-detect if not specified)"
    )
    parser.add_argument(
        "--gpio", "-g", type=int, default=DEFAULT_GPIO_PIN,
        help=f"GPIO pin (BCM numbering) for GATE input (default: {DEFAULT_GPIO_PIN})"
    )
    parser.add_argument(
        "--pre-trigger", "-p", type=int, default=12,
        help="Pre-trigger segments S#n, 0-32 (default: 12, ~51ms pre-trigger)"
    )
    parser.add_argument(
        "--swings-only", "-s", action="store_true",
        help="Only show captures with real swing data (>= 15 mph outbound)"
    )
    parser.add_argument(
        "--debounce", "-d", type=int, default=200,
        help="Debounce time in ms to ignore rapid triggers (default: 200)"
    )

    args = parser.parse_args()

    if not 0 <= args.pre_trigger <= 32:
        print("Error: pre-trigger must be between 0 and 32")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  OpenFlight GPIO-Assisted Sound Trigger Tester")
    print(f"  (SEN-14262 GATE → Pi GPIO{args.gpio})")
    print("=" * 60)
    print()

    run(args.port, args.gpio, args.pre_trigger, args.swings_only, args.debounce)


if __name__ == "__main__":
    main()
