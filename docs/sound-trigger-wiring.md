# Sound Trigger Wiring Guide

Step-by-step instructions for wiring the sound trigger that enables spin detection in rolling buffer mode.

> **Parts needed:** See the [Parts List](PARTS.md#sound-trigger-for-rolling-buffer-mode) for what to buy.

## Overview

The SparkFun SEN-14262 sound detector listens for club impact and triggers the OPS243-A radar to dump its I/Q buffer. That captured data is then analyzed for spin rate estimation.

The wiring is simple — three wires:

```
SEN-14262 GATE → OPS243-A HOST_INT (J3 Pin 3)
SEN-14262 VCC  → Raspberry Pi 3.3V
SEN-14262 GND  → Raspberry Pi GND (shared with OPS243-A GND)
```

## Before You Wire: Solder R17

**R17 is not a part already on the board — it's an empty pair of solder pads** (a footprint) that SparkFun left on the Sound Detector so you can tune the microphone preamp. The silkscreen prints `R17` next to the two empty holes, in the preamp section right beside the surface-mount resistor **R3**. You bridge those two pads with one through-hole resistor. Resistors aren't polarized, so orientation doesn't matter.

**Why it's needed:** the SEN-14262 is factory-tuned for 5V. At the 3.3V used here the preamp gain is too high, so the GATE output saturates and gets stuck high (always "triggered"). Adding R17 lowers the gain so GATE fires only on a sharp, loud transient — a club striking the ball.

**How it works:** R17 sits electrically *in parallel* with the onboard 100kΩ R3. Two resistors in parallel combine to less than either one alone, and a smaller feedback resistor means lower amplifier gain:

| R17 value | R17 ∥ R3 (100kΩ) | Effect | When to use |
|-----------|------------------|--------|-------------|
| 47kΩ | ~32kΩ | Moderate gain reduction | Start here |
| 33kΩ | ~25kΩ | More reduction | Still too sensitive / noisy room |

Start with 47kΩ. After wiring, clap near the sensor: the onboard GATE LED should flash briefly then go dark. If the LED stays lit constantly, the gain is still too high — switch to a lower value.

> Board photos showing the exact R17 location are in SparkFun's [Sound Detector Hookup Guide](https://learn.sparkfun.com/tutorials/sound-detector-hookup-guide/all).

---

## Wiring

### Step 1: Identify OPS243-A J3 Header Pins

```
OPS243-A J3 Header:
┌───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │
│GND│   │INT│   │   │   │
└───┴───┴───┴───┴───┴───┘
```

- **Pin 1** = GND
- **Pin 3** = HOST_INT (trigger input)

### Step 2: Connect Power

1. Connect **SEN-14262 VCC** → **Pi 3.3V** (physical pin 1)
2. Connect **SEN-14262 GND** → **Pi GND** (physical pin 6)
3. Connect **OPS243-A GND (J3 Pin 1)** → same **Pi GND** rail

All three boards must share a common ground.

### Step 3: Connect Trigger

1. Connect **SEN-14262 GATE** → **OPS243-A HOST_INT (J3 Pin 3)**

That's it. No level shifter, no MOSFETs, no breadboard needed.

```
SEN-14262               Raspberry Pi           OPS243-A
┌───────────┐          ┌──────────┐          ┌──────────┐
│ VCC ──────┼──────────┤ 3.3V     │          │          │
│           │          │          │          │          │
│ GATE ─────┼──────────┼──────────┼──────────┤ HOST_INT │
│           │          │          │          │ (J3 P3)  │
│ GND ──────┼──────────┤ GND      ├──────────┤ GND      │
│           │          │          │          │ (J3 P1)  │
└───────────┘          └──────────┘          └──────────┘
```

---

## Wiring Checklist

- [ ] R17 resistor soldered on SEN-14262 board
- [ ] SEN-14262 VCC → Pi 3.3V (pin 1)
- [ ] SEN-14262 GND → Pi GND (pin 6)
- [ ] SEN-14262 GATE → OPS243-A HOST_INT (J3 Pin 3)
- [ ] OPS243-A GND (J3 Pin 1) → Pi GND (shared ground)

---

## One-Time Radar Setup

The OPS243-A must have rolling buffer mode saved to persistent memory for HOST_INT triggers to work. This is due to a firmware bug where the HOST_INT pin mode changes when transitioning modes at runtime.

```bash
# Configure and save rolling buffer mode to flash (one-time)
uv run python scripts/hardware-test/test_rolling_buffer_persist.py --setup

# Power cycle the radar (unplug USB, wait 3s, replug)

# Verify
uv run python scripts/hardware-test/test_rolling_buffer_persist.py --test
```

---

## Testing

### Quick Test: Visual

Make a loud sound near the SEN-14262. The onboard LED should flash briefly, then turn off. If the LED stays on constantly, you need a lower-value resistor in R17.

### Full Test: Software

```bash
uv run python scripts/hardware-test/test_sound_trigger_hardware.py
```

You should see:
```
Ready for hardware sound triggers!
Make a sound near the sensor... (Ctrl+C to quit)

[1] Waiting for hardware trigger (timeout=60s)...
  TRIGGER RECEIVED after 0.02s!
  I/Q samples: 4096 I, 4096 Q
```

---

## Troubleshooting

### GATE LED stays on (stuck high)

The preamp gain is too high for 3.3V operation.
- **Fix:** Solder a lower-value resistor into R17 (try 33kΩ instead of 47kΩ)

### No trigger received

1. Check the GATE LED flashes when you clap
2. Verify GND is shared between all three boards (Pi, SEN-14262, OPS243-A)
3. Verify HOST_INT is J3 **Pin 3** (not Pin 2)
4. Run `uv run python scripts/hardware-test/test_rolling_buffer_persist.py --test` to confirm radar is in rolling buffer mode

### Triggers constantly / too sensitive

- Use a lower-value R17 resistor to reduce gain
- Move the sensor further from the hitting area

### Triggers but no I/Q data

- Run the one-time radar setup (see above) — HOST_INT mode must be saved to flash
- Power cycle the radar after setup
