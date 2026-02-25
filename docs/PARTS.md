# OpenFlight Parts List

Hardware components for building the OpenFlight golf launch monitor.

## Core Components

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **OPS243-A Radar** | Doppler radar for ball/club speed detection | [OmniPreSense](https://omnipresense.com/product/ops243-a-doppler-radar-sensor/) | $249 |
| **Raspberry Pi 5** | Main compute unit (4GB+ recommended) | [Adafruit](https://www.adafruit.com/product/5812) | $60 |
| **Raspberry Pi Camera Module 3 Wide** | IMX708 wide-angle (120 FOV) for ball/launch detection | [Adafruit](https://www.adafruit.com/product/5658) | $35 |
| **7" Touchscreen Display** | 800x480 DSI display (connects to Pi 5 DSI/CSI port) | [Amazon](https://www.amazon.com/dp/B0D3QB7X4Z) | $46 |

## Sound Trigger (for Rolling Buffer Mode)

The sound trigger enables precise timing of radar captures by detecting the club impact sound. This is essential for spin detection via rolling buffer mode.

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **SparkFun SEN-14262** | Sound Detector with envelope/gate outputs | [SparkFun](https://www.sparkfun.com/products/14262) | $12 |
| **IRLZ44N MOSFET (2)** | Logic-level N-channel MOSFETs for driver circuit | [Amazon](https://www.amazon.com/dp/B0CBKH4XGL) | $8 |
| **4.7Ω Resistor** | Strong pull-up for HOST_INT drive | Any electronics supplier | $1 |
| **10kΩ Resistor** | Pull-up for first inverter stage | Any electronics supplier | $1 |
| **1kΩ Resistor** | Gate protection (optional) | Any electronics supplier | $1 |
| **Breadboard** | For circuit assembly | Any | $5 |
| **Jumper Wires** | For connections | Any | $5 |

### Sound Trigger Wiring (MOSFET Driver - Recommended)

The MOSFET double-inverter provides high-current 3.3V drive to HOST_INT. This is the recommended method for reliable triggering.

```
SEN-14262                    BREADBOARD                         OPS243-A
┌───────────┐                                                 ┌──────────┐
│           │                  3.3V RAIL                      │          │
│ VCC ──────┼───────════════════╤═══════════════════          │          │
│           │                   │                             │          │
│           │                 [10kΩ]          [4.7Ω]          │          │
│           │                   │               │             │          │
│ GATE ─────┼──[1kΩ]──►Gate    Drain           Drain ────────►│ HOST_INT │
│           │                 ┌─┴─┐    wire   ┌─┴─┐           │ (J3 P3)  │
│           │                 │Q1 │ ─────────►│Q2 │           │          │
│           │                 └─┬─┘  (G to D) └─┬─┘           │          │
│           │                   │               │             │          │
│ GND ──────┼───────════════════╧═══════════════╧═════════════┼── GND    │
│           │                 GND RAIL                        │ (J3 P1)  │
└───────────┘                                                 └──────────┘
```

**Why MOSFETs?** The OPS243-A HOST_INT has very low input impedance (~27Ω) requiring high current drive (~100mA+). The IRLZ44N double-inverter with 4.7Ω pull-up provides ~700mA capability at full 3.3V.

**Trigger Latency:** ~1μs (pure hardware switching).

See [sound-trigger-wiring.md](sound-trigger-wiring.md) for detailed step-by-step instructions.

### Alternative: GPIO Passthrough Method

If you don't have MOSFETs, the Pi can act as a software-controlled trigger with slightly higher latency:

```
SEN-14262 GATE → GPIO17 (pin 11) [input]
GPIO27 (pin 13) → HOST_INT (J3 Pin 3) [output]
```

**Trigger Latency:** ~10μs (lgpio C callback) - use `--trigger sound-passthrough`.

## IR Illumination (for camera)

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **IR LED Camera Module** | OV5647 camera with 2x 3W 850nm IR LEDs (use LEDs only) | [Amazon](https://www.amazon.com/MELIFE-Raspberry-Camera-Adjustable-Focus-Infrared/dp/B08RHZ5BJM) | $16 |

### Alternative IR Options

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| 5W IR LED Module (2-pack) | Higher power, direct Pi connection | [Amazon](https://www.amazon.com/Infrared-Raspberry-Illuminator-Adjustable-Resistor/dp/B0D39S5RLW) | $12 |
| 3W IR LED Module (2-pack) | Standard power, direct Pi connection | [Amazon](https://www.amazon.com/Infrared-Illuminator-Adjustable-Resistor-Raspberry/dp/B07FM6LL3V) | $10 |
| 10W IR LED Chip | For outdoor use (requires 12V driver setup) | [Amazon](https://www.amazon.com/dp/B01DBZK4EM) | $8 |

### 10W LED Driver Setup (if needed for outdoor)

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| 12V 3A Power Supply | Powers Pi + LED | Search "12V 3A barrel jack adapter" | $8 |
| 12V to 5V Buck Converter | Powers Pi from 12V supply | Search "12V to 5V 3A buck converter" | $6 |
| XL4015 CC/CV Module | Constant current driver for 10W LED | [Amazon](https://www.amazon.com/Organizer-Adjustable-Step-Down-Voltmeter-Constant/dp/B07VDKD5YQ) | $8 |
| Heatsink (40x40mm) | Required for 10W LED | Search "aluminum heatsink 40mm" | $3 |

## Power & Accessories

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **27W USB-C Power Supply** | Official Pi 5 power supply (5V 5A) | [Adafruit](https://www.adafruit.com/product/5974) | $12 |
| MicroSD Card (32GB+) | For Pi OS and software | Any Class 10 | $10 |
| USB-A to Micro-USB Cable | For OPS243-A radar connection | Any | $5 |

## Shaft IMU Sensor (for club data)

A shaft-mounted IMU measures club face angle, swing path, attack angle, and tempo directly. Communicates to the Pi over BLE. See [ENGINEERING_BACKLOG.md](ENGINEERING_BACKLOG.md) for integration plan.

**Option A: Adafruit (one vendor, STEMMA QT plug-and-play I2C, ~$41 shipped)**

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **Adafruit QT Py ESP32-C3** | BLE 5.0 + WiFi, STEMMA QT connector | [Adafruit](https://www.adafruit.com/product/5405) | $9.95 |
| **Adafruit MPU-6050** | 6-axis accel + gyro, STEMMA QT | [Adafruit](https://www.adafruit.com/product/3886) | $12.95 |
| **3.7V 400mAh LiPo** | JST-PH 2.0mm, fits Adafruit boards | [Adafruit](https://www.adafruit.com/product/3898) | $6.95 |
| **Micro-Lipo USB-C Charger** | LiPo charger with protection | [Adafruit](https://www.adafruit.com/product/4410) | $5.95 |
| Shipping | Flat rate | Adafruit | ~$5 |

**Option B: Amazon clone boards (bulk packs, free Prime shipping, ~$35-40)**

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| ESP32-C3 SuperMini (5-pack) | BLE 5.0 + WiFi, 22x18mm | Search "ESP32-C3 SuperMini" | ~$12-15 |
| GY-521 MPU-6050 (3-pack) | 6-axis accel + gyro, I2C | Search "GY-521 MPU-6050" | ~$6-8 |
| 3.7V 400mAh LiPo | JST-PH connector | Search "3.7V 400mAh LiPo JST" | ~$8-10 |
| TP4056 USB-C (5-pack) | LiPo charger with protection | Search "TP4056 USB-C charger" | ~$7-8 |

### Shaft IMU Wiring (I2C, 4 wires)

```
ESP32-C3          MPU-6050
  3.3V  ────────  VIN
  GND   ────────  GND
  GPIO6 ────────  SDA
  GPIO7 ────────  SCL
```

With Adafruit STEMMA QT boards (Option A), use a STEMMA QT cable instead of wiring — no soldering needed.

**Power:** LiPo → TP4056/Micro-Lipo → ESP32-C3 (3.3V out powers MPU-6050)

**Battery life:** 400mAh at ~30mA draw (BLE + IMU) = ~13 hours.

## Optional

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| Pi Case with Camera Mount | Enclosure for Pi + HQ Camera | Various | $15-30 |
| Tripod Mount | For positioning the unit | 1/4"-20 mount | $10 |
| **850nm IR Pass Filter** | For outdoor use - blocks visible light, only passes IR | [Kurokesu](https://www.kurokesu.com/shop/D19x1_NIR_SCREWIN) | ~$20 |

---

## Notes

### IR LED Connection
The IR LED modules connect to the Pi's GPIO:
- **5V**: Pin 2 or Pin 4
- **GND**: Pin 6, 9, 14, 20, 25, 30, 34, or 39

### Camera IR Filter
The Raspberry Pi HQ Camera does **not** have a built-in IR filter, making it suitable for IR illumination without modification. Verify your lens doesn't have an IR-cut filter by pointing a TV remote at the camera - you should see the IR LED flash.

### Power Budget
When powering IR LEDs from Pi's 5V GPIO:
- Pi 5 uses ~2-3A under load (more powerful than Pi 4)
- With 5A power supply (official Pi 5 PSU recommended), ~2A available for accessories
- 6W of IR LEDs @ 5V = 1.2A (safe with good power supply)
