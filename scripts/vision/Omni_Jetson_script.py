#!/usr/bin/env python3
"""
OPS243-A Rolling Buffer Data Capture
-------------------------------------
Designed for NVIDIA Jetson Nano with OPS243-A radar sensor connected via USB.

Usage:
    python3 ops243a_rolling_buffer.py [options]

Options:
    --port      /dev/ttyACM0    Serial port (auto-detected if omitted)
    --trigger   sw | hw         Trigger mode: sw = software (default), hw = hardware GPIO
    --gpio-pin  29              Jetson Nano physical board pin for hardware trigger (default 29)
    --pre-trigger N             Pre-trigger segments 0-32 (default 8)
    --output    file.csv        Output CSV filename (auto-named if omitted)

Sample rate is fixed at 20ksps (API command S2).
USB port opens directly at 115200. IR is sent as a fire-and-forget API
command to enable the OPS243-A hardware UART pins (sensor-internal only,
does not affect the USB connection).

Keyboard controls (both trigger modes):
    Ctrl+T      Fire trigger (software S! command OR hardware GPIO pulse)
    q           Quit and restore normal radar mode

Trigger modes:
    sw  Software trigger — sends the S! API command over USB serial.
        No extra wiring required.

    hw  Hardware trigger — rising-edge signal on OPS243-A J3 Pin 3 (HOST_INT).
          * sudo privileges for devmem (8mA drive strength register write)
        An external 10kΩ pull-down resistor from the signal line to GND is
        required — the OPS243-A J3 Pin 3 (HOST_INT) is tri-state (hi-Z) so
        without a pull-down the line will float and no edge will be detected.
        Pin behaviour:
          - Initialised LOW at startup
          - Driven LOW before GC command arms the sensor
          - On Ctrl+T: driven HIGH — rising edge triggers the OPS243-A
          - Stays HIGH while the 4096-sample I/Q data burst is read back
          - Driven LOW after data received (or timeout), before PA re-arm
        Requires:
          * Jetson.GPIO  (pip3 install Jetson.GPIO)
          * sudo privileges for devmem (drive strength register write)
          * Physical pin 29 (GPIO01) wired to OPS243-A J3 Pin 3
          * No separate GND wire needed — common ground via USB

Reference: AN-027-B OPS243-A Large Rolling Buffer (OmniPreSense)
"""

import serial
import serial.tools.list_ports
import csv
import time
import sys
import os
import argparse
import termios
import tty
import threading
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BAUD_RATE           = 115200      # USB CDC-ACM port baud rate
READ_TIMEOUT        = 30.0        # seconds to wait for full 4096-sample burst
                                  # At 115200 baud a full I/Q dump can take 10-15s
IDLE_TIMEOUT        = 3.0         # seconds of silence before declaring capture done
ROLLING_BUF_SIZE    = 4096        # fixed samples per capture
SEGMENT_SIZE        = 128         # samples per segment
NUM_SEGMENTS        = 32          # total segments in rolling buffer
SAMPLE_RATE_CMD     = "S2"        # OPS243-A API command for 20ksps
CTRL_T              = "\x14"      # ASCII 0x14 = Ctrl+T

# Tegra X1 pad control register for GPIO01 (physical pin 29 / SPI2_CS1_N_PDD7)
# Base: 0x70003000  Offset: 0x3068  — DRV_SPI2 pad group
# Bits [12:10] = DRVDN (drive-down), Bits [7:5] = DRVUP (drive-up)
# 8mA = code 3 = 0b011 in both DRVDN[12:10] and DRVUP[7:5] fields
GPIO_PAD_REG_ADDR   = 0x70003068  # pad control register for pin 29 pad group
GPIO_PAD_8MA_VALUE  = 0x00001460  # DRVDN=3, DRVUP=3 → 8mA drive strength

# ─────────────────────────────────────────────────────────────────────────────
# GPIO helpers  (hardware trigger mode)
# ─────────────────────────────────────────────────────────────────────────────
def gpio_set_drive_strength(pin):
    """
    Set pin 29 (GPIO01) pad drive strength to 8mA via the Tegra X1 pad
    control register using 'sudo -n busybox devmem'.

    'sudo -n' is non-interactive — it never prompts for a password and
    never hangs. If passwordless sudo is not configured it fails instantly
    with a clear warning instead of blocking the script.

    To enable passwordless sudo for devmem, run this once on the Jetson:
        sudo visudo -f /etc/sudoers.d/jetson-devmem
    Add this line and save:
        rob ALL=(ALL) NOPASSWD: /bin/busybox devmem

    Only applies to physical pin 29 — other pins use different registers.
    """
    if pin != 29:
        print(f"[GPIO]  Drive strength register only configured for pin 29. "
              f"Pin {pin} uses default drive strength.")
        return

    import subprocess
    reg = f"0x{GPIO_PAD_REG_ADDR:08X}"
    val = f"0x{GPIO_PAD_8MA_VALUE:08X}"
    print(f"[GPIO]  Setting pin 29 drive strength to 8mA "
          f"(devmem {reg} 32 {val}) ...")
    try:
        result = subprocess.run(
            ["sudo", "-n", "busybox", "devmem", reg, "32", val],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=5
        )
        if result.returncode == 0:
            print("[GPIO]  Drive strength set to 8mA.")
        else:
            print("[WARN]  devmem needs passwordless sudo. To fix, run:")
            print("[WARN]    sudo visudo -f /etc/sudoers.d/jetson-devmem")
            print("[WARN]  Add line:  rob ALL=(ALL) NOPASSWD: /bin/busybox devmem")
            print("[WARN]  Continuing with default 2mA drive strength.")
    except FileNotFoundError:
        print("[WARN]  busybox not found. Install with: "
              "sudo apt-get install busybox")
        print("[WARN]  Continuing with default drive strength.")
    except Exception as e:
        print(f"[WARN]  Could not set drive strength: {e}")
        print("[WARN]  Continuing with default drive strength.")


def gpio_setup(pin):
    """
    Initialise Jetson.GPIO for output on the given physical board pin.
    Pin is initialised LOW — it will be driven LOW explicitly before the
    GC command arms the sensor, ensuring a clean known state before
    Ctrl+T produces the rising edge trigger.
    Returns the GPIO module on success, or None if unavailable.
    """
    gpio_set_drive_strength(pin)
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        print(f"[GPIO]  Pin {pin} configured as output, initialised LOW.")
        return GPIO
    except ImportError:
        print("[ERROR] Jetson.GPIO is not installed.")
        print("        Run:  pip3 install Jetson.GPIO")
        print("        Then: sudo groupadd -f gpio && sudo usermod -aG gpio $USER")
        return None
    except Exception as e:
        print(f"[ERROR] GPIO setup failed on pin {pin}: {e}")
        return None


def gpio_trigger_high(GPIO, pin):
    """
    Drive the trigger pin HIGH (3.3V) on Ctrl+T.
    Produces the rising edge the OPS243-A detects on HOST_INT.
    Pin stays HIGH until gpio_set_low() is called after data readback.
    """
    GPIO.output(pin, GPIO.HIGH)
    print(f"[GPIO]  Pin {pin} driven HIGH — rising edge trigger sent.")


def gpio_set_low(GPIO, pin):
    """
    Drive the trigger pin LOW (0V).
    Called in two places:
      1. Before the GC command — ensures a clean LOW before arming
      2. After data readback (or timeout) — before PA re-arms the sensor
    """
    GPIO.output(pin, GPIO.LOW)
    print(f"[GPIO]  Pin {pin} driven LOW.")


def gpio_cleanup(GPIO, pin):
    """
    Drive pin LOW then release GPIO resources on exit.
    LOW is the safe idle state — no unintended trigger on cleanup.
    """
    try:
        GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup(pin)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Terminal raw-mode keypress reader
# ─────────────────────────────────────────────────────────────────────────────
class KeyReader:
    """
    Reads individual keypresses from stdin without requiring ENTER.
    Runs a background thread so the main loop is not blocked.

    Usage:
        kr = KeyReader()
        kr.start()
        ch = kr.get(timeout=0.2)   # returns char or None on timeout
        kr.stop()
    """

    def __init__(self):
        self._queue        = []
        self._lock         = threading.Lock()
        self._event        = threading.Event()
        self._active       = False
        self._thread       = None
        self._old_settings = None

    def start(self):
        """Switch stdin to cbreak (raw single-char) mode and start reader thread."""
        self._old_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._active = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reader thread and restore original terminal settings."""
        self._active = False
        if self._old_settings:
            try:
                termios.tcsetattr(sys.stdin.fileno(),
                                  termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def _reader(self):
        import select as sel
        while self._active:
            ready, _, _ = sel.select([sys.stdin], [], [], 0.1)
            if ready:
                ch = sys.stdin.read(1)
                with self._lock:
                    self._queue.append(ch)
                self._event.set()

    def get(self, timeout=None):
        """Block up to timeout seconds for a keypress. Returns char or None."""
        self._event.wait(timeout)
        with self._lock:
            if self._queue:
                ch = self._queue.pop(0)
                if not self._queue:
                    self._event.clear()
                return ch
            self._event.clear()
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Serial port auto-detection
# ─────────────────────────────────────────────────────────────────────────────
def find_ops243_port():
    """
    Scan available serial ports and return the most likely OPS243-A port.
    Prefers /dev/ttyACM* (CDC-ACM), falls back to /dev/ttyUSB*.
    """
    ports     = list(serial.tools.list_ports.comports())
    acm_ports = [p.device for p in ports if "ACM" in p.device]
    usb_ports = [p.device for p in ports if "USB" in p.device]

    if acm_ports:
        return sorted(acm_ports)[0]
    if usb_ports:
        return sorted(usb_ports)[0]

    if ports:
        print("Available ports:")
        for p in ports:
            print(f"  {p.device}  ({p.description})")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sensor communication helpers
# ─────────────────────────────────────────────────────────────────────────────
def send_command(ser, cmd, delay=0.1):
    """Send a command string terminated with newline. Does not flush —
    flush() can block indefinitely on USB CDC-ACM ports."""
    full_cmd = (cmd.strip() + "\n").encode("ascii")
    ser.write(full_cmd)
    time.sleep(delay)
    print(f"  [TX] {cmd.strip()}")


def read_response(ser, timeout=1.0):
    """Read all available lines within timeout seconds."""
    lines    = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if line:
                lines.append(line)
        else:
            time.sleep(0.01)
    return lines


def configure_sensor(ser, pre_trigger_segments):
    """Set 20ksps sample rate (S2) and configure pre-trigger split."""
    print(f"\n[CONFIG] Setting sample rate to 20ksps (API command: {SAMPLE_RATE_CMD})")
    send_command(ser, SAMPLE_RATE_CMD, delay=0.3)
    _ = read_response(ser, timeout=0.5)

    print(f"[CONFIG] Pre-trigger segments (S#{pre_trigger_segments})")
    send_command(ser, f"S#{pre_trigger_segments}", delay=0.2)
    _ = read_response(ser, timeout=0.5)


def enable_rolling_buffer(ser, GPIO=None, gpio_pin=None):
    """
    Drive pin LOW (hw mode), then send GC to enter rolling buffer mode.
    Pin is set LOW before GC so the OPS243-A sees a stable LOW on HOST_INT
    while it arms — Ctrl+T then produces a clean low-to-high rising edge.

    The sensor responds with ~17 lines of configuration info.
    The last line should be:
        {"OperationMode":"Continuous Sampling Mode"}
    This confirms the sensor entered rolling buffer mode successfully.
    """
    if GPIO is not None and gpio_pin is not None:
        gpio_set_low(GPIO, gpio_pin)
        print("[MODE]  Pin LOW — ready for rising-edge trigger.")

    print("[MODE]  Sending GC (Rolling Buffer mode) ...")
    send_command(ser, "GC", delay=0.5)

    print("[MODE]  Reading sensor configuration response ...")
    resp = read_response(ser, timeout=3.0)

    confirmed = False
    for line in resp:
        print(f"[MODE]    {line}")
        if "Continuous Sampling Mode" in line:
            confirmed = True

    if confirmed:
        print("[MODE]  Confirmed: sensor is in Rolling Buffer "
              "(Continuous Sampling Mode).")
    else:
        print("[WARN]  Rolling Buffer mode confirmation not received.")
        print("[WARN]  Expected last line: "
              '{"OperationMode":"Continuous Sampling Mode"}')
        print("[WARN]  Proceeding — check sensor wiring and firmware version.")


def open_serial(port):
    """
    Open the USB serial port to the OPS243-A.
    USB CDC-ACM does not require a specific baud rate setting on the host —
    the OS negotiates it automatically.

    After opening, sends one fire-and-forget API command to the sensor:
      IR — enable the OPS243-A hardware UART interface (sensor-internal,
           does not affect the USB connection in any way)
    No response is expected from IR.

    Returns the open serial.Serial object, or calls sys.exit(1) on failure.
    """
    print(f"[UART]  Opening {port} ...")
    try:
        ser = serial.Serial(
            port     = port,
            baudrate = BAUD_RATE,
            bytesize = serial.EIGHTBITS,
            parity   = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE,
            timeout  = 0.1,
        )
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {port}: {e}")
        sys.exit(1)

    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Fire-and-forget — no response expected
    print("[UART]  Sending IR (enable OPS243-A hardware UART) ...")
    send_command(ser, "IR", delay=0.3)

    ser.reset_input_buffer()   # discard any unexpected noise
    print("[UART]  Serial port ready.")
    return ser


# ─────────────────────────────────────────────────────────────────────────────
# Trigger dispatch
# ─────────────────────────────────────────────────────────────────────────────
def fire_trigger(ser, trigger_mode, GPIO=None, gpio_pin=None):
    """
    Fire the trigger in the selected mode.

    sw mode: sends the S! API command over USB serial.
    hw mode: drives the GPIO pin HIGH to produce a rising edge on
             OPS243-A J3 Pin 3 (HOST_INT). Pin was already LOW before
             the GC arm command, so this is a clean low-to-high transition.
             Pin stays HIGH — caller must call gpio_set_low() after all
             capture data has been read back (or on timeout).

    Returns True if the trigger was fired successfully, False otherwise.
    """
    if trigger_mode == "hw":
        if GPIO is None or gpio_pin is None:
            print("[ERROR] Hardware trigger requested but GPIO is not initialised.")
            return False
        gpio_trigger_high(GPIO, gpio_pin)
        return True
    else:
        print("\n[TRIG]  Sending software trigger (S!) ...")
        send_command(ser, "S!", delay=0.05)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Data capture & parsing
# ─────────────────────────────────────────────────────────────────────────────
def read_capture(ser):
    """
    Read the raw I/Q data burst from the sensor after a trigger event.

    Returns a dict:
        {
            "trigger_time": <float | None>,
            "start_time":   <float | None>,
            "raw_lines":    [str, ...],
            "i_samples":    [int, ...],
            "q_samples":    [int, ...],
        }
    """
    result = {
        "trigger_time": None,
        "start_time":   None,
        "raw_lines":    [],
        "i_samples":    [],
        "q_samples":    [],
    }

    print("[RECV]  Waiting for sensor data burst ...", flush=True)
    t_start     = time.time()
    t_first     = None               # time first byte arrives
    deadline    = t_start + READ_TIMEOUT
    last_data   = t_start
    total_lines = 0

    while True:
        now = time.time()
        if now > deadline:
            elapsed = now - t_start
            print(f"[WARN]  Read timeout reached after {elapsed:.1f}s "
                  f"(READ_TIMEOUT={READ_TIMEOUT}s). "
                  f"Lines received so far: {total_lines}")
            break
        if (now - last_data) > IDLE_TIMEOUT and total_lines > 0:
            elapsed = now - t_start
            print(f"[RECV]  Idle timeout — no data for {IDLE_TIMEOUT}s. "
                  f"Total read time: {elapsed:.2f}s")
            break

        if ser.in_waiting:
            try:
                raw = ser.readline().decode("ascii", errors="replace").strip()
            except Exception:
                continue

            if not raw:
                continue

            # Record time of first data byte
            if t_first is None:
                t_first = time.time()
                print(f"[RECV]  First data arrived {t_first - t_start:.3f}s "
                      f"after trigger. Reading ...", flush=True)

            last_data    = time.time()
            total_lines += 1
            result["raw_lines"].append(raw)

            # Print first 3 lines raw so format is visible in console
            if total_lines <= 3:
                print(f"[RECV]  Line {total_lines}: {raw[:80]}")

            # Parse timestamp header lines
            if raw.lower().startswith("start") or "1st sample" in raw.lower():
                try:
                    result["start_time"] = float(raw.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "trigger" in raw.lower() and "time" in raw.lower():
                try:
                    result["trigger_time"] = float(raw.split(":")[-1].strip())
                except ValueError:
                    pass

            # Parse I/Q data lines
            _parse_iq_line(raw, result)

        else:
            time.sleep(0.005)

    elapsed_total = time.time() - t_start
    print(f"[RECV]  Capture read complete: {total_lines} lines | "
          f"I samples: {len(result['i_samples'])} | "
          f"Q samples: {len(result['q_samples'])} | "
          f"Total time: {elapsed_total:.2f}s")
    return result


def _parse_iq_line(line, result):
    """
    Extract integer I/Q sample values from a raw sensor output line.
    Handles formats:
        "I: 1234 2345 ..."   "Q: 1234 2345 ..."
        "I,1234,2345,..."    "Q,1234,2345,..."
        "1234,5678"          (comma-separated pair per line)
        "1234 5678"          (space-separated pair per line)
    """
    s = line.strip()

    if s.upper().startswith("I:") or s.upper().startswith("I,"):
        result["i_samples"].extend(_extract_ints(s[2:]))
        return
    if s.upper().startswith("Q:") or s.upper().startswith("Q,"):
        result["q_samples"].extend(_extract_ints(s[2:]))
        return

    parts = s.split(",")
    if len(parts) == 2:
        try:
            result["i_samples"].append(int(parts[0]))
            result["q_samples"].append(int(parts[1]))
        except ValueError:
            pass
        return

    parts = s.split()
    if len(parts) == 2:
        try:
            result["i_samples"].append(int(parts[0]))
            result["q_samples"].append(int(parts[1]))
        except ValueError:
            pass


def _extract_ints(text):
    """Return list of integers parsed from a whitespace/comma-separated string."""
    values = []
    for t in text.replace(",", " ").split():
        try:
            values.append(int(t))
        except ValueError:
            pass
    return values


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────
def save_to_csv(capture, output_path, trigger_mode):
    """
    Write captured I/Q samples to a CSV file.
    Columns: sample_index, segment, i_value, q_value
    """
    i_data = capture["i_samples"]
    q_data = capture["q_samples"]
    n      = max(len(i_data), len(q_data))

    if n == 0:
        print("[WARN]  No I/Q samples parsed -- raw lines saved separately.")
        raw_path = output_path.replace(".csv", "_raw.txt")
        with open(raw_path, "w") as f:
            f.write("\n".join(capture["raw_lines"]))
        print(f"[FILE]  Raw lines -> {raw_path}")
        return

    # Pad shorter array with zeros if lengths differ
    if len(i_data) < n:
        i_data += [0] * (n - len(i_data))
    if len(q_data) < n:
        q_data += [0] * (n - len(q_data))

    start_time   = capture.get("start_time",   0.0) or 0.0
    trigger_time = capture.get("trigger_time", 0.0) or 0.0

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["# OPS243-A Rolling Buffer Capture"])
        writer.writerow(["# Capture timestamp",   datetime.now().isoformat()])
        writer.writerow(["# Trigger mode",         trigger_mode])
        writer.writerow(["# 1st sample time (s)",  start_time])
        writer.writerow(["# Trigger time (s)",      trigger_time])
        writer.writerow(["# Total samples",         n])
        writer.writerow([])

        writer.writerow(["sample_index", "segment", "i_value", "q_value"])

        for idx in range(n):
            segment = idx // SEGMENT_SIZE
            writer.writerow([idx, segment, i_data[idx], q_data[idx]])

    print(f"[FILE]  Saved {n} samples -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="OPS243-A Rolling Buffer capture tool for Jetson Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  python3 ops243a_rolling_buffer.py --trigger hw --gpio-pin 11 "
               "--sample-rate 20ksps\n"
    )
    parser.add_argument(
        "--port", default=None,
        help="Serial port (e.g. /dev/ttyACM0). Auto-detected if omitted."
    )
    parser.add_argument(
        "--trigger", default="sw", choices=["sw", "hw"],
        help="sw = software S! trigger (default)  |  hw = GPIO hardware pulse"
    )
    parser.add_argument(
        "--gpio-pin", type=int, default=29, metavar="PIN",
        help="Jetson Nano physical board pin for hardware trigger (default 29 = GPIO01, 3.3V). "
             "WARNING: avoid pin 11 (GPIO17) — it is a 1.8V domain and idles ~1.6V."
    )
    parser.add_argument(
        "--pre-trigger", type=int, default=8, metavar="N",
        help="Pre-trigger segments 0-32 (default 8 = 25%% historical data)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV base filename. Auto-generated with timestamp if omitted."
    )
    args = parser.parse_args()

    # ── Resolve output filename ───────────────────────────────────────────
    if args.output is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"ops243a_capture_{ts}.csv"
    else:
        output_base = args.output

    # ── Locate sensor port ────────────────────────────────────────────────
    port = args.port
    if port is None:
        print("[INIT]  Auto-detecting OPS243-A serial port ...")
        port = find_ops243_port()
        if port is None:
            print("[ERROR] No USB serial port found. "
                  "Ensure the sensor is plugged in and try again.")
            sys.exit(1)
        print(f"[INIT]  Found port: {port}")
    else:
        print(f"[INIT]  Using specified port: {port}")

    # ── Clamp pre-trigger to valid range ──────────────────────────────────
    pre_trig = max(0, min(32, args.pre_trigger))
    if pre_trig != args.pre_trigger:
        print(f"[WARN]  Pre-trigger clamped to {pre_trig} (valid range 0-32)")

    trigger_mode = args.trigger

    # ── Initialise GPIO if hardware trigger requested ─────────────────────
    GPIO     = None
    gpio_pin = args.gpio_pin
    if trigger_mode == "hw":
        GPIO = gpio_setup(gpio_pin)
        if GPIO is None:
            print("[ERROR] Hardware trigger initialisation failed. Exiting.")
            sys.exit(1)

    # ── Open USB serial port and send IR/I4 to sensor ────────────────────
    ser = open_serial(port)

    # ── Start raw keypress reader ─────────────────────────────────────────
    kr            = KeyReader()
    capture_count = 0

    try:
        configure_sensor(ser, pre_trig)
        enable_rolling_buffer(ser, GPIO, gpio_pin)

        trig_label = (
            f"Hardware GPIO pulse  (Jetson pin {gpio_pin} -> OPS243-A J3 Pin 3)"
            if trigger_mode == "hw"
            else "Software S! command over USB serial"
        )

        print("\n" + "=" * 66)
        print("  OPS243-A Rolling Buffer -- Interactive Capture")
        print(f"  Trigger mode : {trig_label}")
        print(f"  Sample rate  : 20ksps (API command: {SAMPLE_RATE_CMD})")
        print(f"  Pre-trigger  : {pre_trig} segments "
              f"({pre_trig * SEGMENT_SIZE} samples = "
              f"{100 * pre_trig // NUM_SEGMENTS}% historical)")
        print("  Controls     : Ctrl+T = fire trigger   q = quit")
        print("=" * 66 + "\n")

        kr.start()
        print("Waiting for Ctrl+T to trigger (or 'q' to quit) ...\n")

        while True:
            ch = kr.get(timeout=0.2)

            if ch is None:
                continue

            # Quit
            if ch.lower() == "q" or ch == "\x03":    # 'q' or Ctrl+C
                print("\n[EXIT]  Quit key pressed.")
                break

            # Ctrl+T -- fire trigger
            if ch == CTRL_T:
                capture_count += 1
                ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n---- Capture #{capture_count}  [{ts_str}] ----")

                # Drive pin HIGH (rising edge) or send S! (sw).
                # Pin was already LOW since before GC — this is a clean
                # low-to-high transition. Pin stays HIGH throughout readback.
                ok = fire_trigger(ser, trigger_mode, GPIO, gpio_pin)
                if not ok:
                    if trigger_mode == "hw":
                        gpio_set_low(GPIO, gpio_pin)
                    continue

                # Read the 4096-sample I/Q data burst — pin stays HIGH in hw mode
                capture = read_capture(ser)

                # Pin goes LOW now — data done (or timeout expired)
                if trigger_mode == "hw":
                    gpio_set_low(GPIO, gpio_pin)

                # Build per-capture output filename
                base, ext = os.path.splitext(output_base)
                ext       = ext if ext else ".csv"
                out_file  = f"{base}_{capture_count:03d}{ext}"

                save_to_csv(capture, out_file, trigger_mode)

                # Re-arm sensor — pin is already LOW, ready for next Ctrl+T
                print("[MODE]  Re-arming rolling buffer (PA) ...")
                send_command(ser, "PA", delay=0.5)
                _ = read_response(ser, timeout=0.5)
                print("[MODE]  Ready -- press Ctrl+T to capture again.\n")


            # All other keys are silently ignored

    finally:
        kr.stop()   # MUST restore terminal before any further print calls

        print("\n[EXIT]  Sending software reset (P!) ...")
        try:
            send_command(ser, "P!", delay=0.3)
        except Exception:
            pass
        ser.close()

        if GPIO:
            gpio_cleanup(GPIO, gpio_pin)

            print("[EXIT]  GPIO released.")

        print(f"[EXIT]  Done.  Total captures this session: {capture_count}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
