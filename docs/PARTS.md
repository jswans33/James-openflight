# OpenFlight Parts List

Hardware components for building the OpenFlight golf launch monitor.

> **Next step after gathering parts:** See the [Raspberry Pi Setup Guide](raspberry-pi-setup.md) for assembly and software installation.

## Core Components

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **OPS243 Radar** | Doppler radar for ball/club speed detection | [OmniPreSense](https://omnipresense.com/product/ops243-doppler-radar-sensor/) | $249 |
| **Raspberry Pi 5** | Main compute unit (4GB+ recommended) | [Adafruit](https://www.adafruit.com/product/5812) | $60 |
| **Raspberry Pi Camera Module 3 Wide** | IMX708 wide-angle (120° FOV), optional — for DTL swing video recording | [Adafruit](https://www.adafruit.com/product/5658) | $35 |

> **WARNING: Do NOT buy the OPS243-A-W (WiFi version).** The WiFi module locks the serial baud rate to 19200, which is far too slow for I/Q data transfer. OpenFlight requires the standard **OPS243** (USB only) which runs at 57600 baud over CDC-ACM. The WiFi version is not compatible.
| **7" Touchscreen Display** | HMTECH 7" 1024x600 IPS display | [Amazon](https://www.amazon.com/dp/B0D3QB7X4Z) | $46 |

## Sound Trigger (for Rolling Buffer Mode)

The sound trigger detects club impact to precisely time radar captures. Essential for spin detection via rolling buffer mode.

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **SparkFun SEN-14262** | Sound Detector with envelope/gate outputs | [SparkFun](https://www.sparkfun.com/products/14262) | $12 |
| **Through-hole resistor** | For R17 pad on SEN-14262 to reduce sensitivity (see note) | Any electronics supplier | $1 |
| **Jumper Wires** | 3 wires: GATE → HOST_INT, VCC → 3.3V, GND → GND | Any | $5 |

> **R17 resistor:** The SEN-14262 is rated for 5V but runs at 3.3V in this setup, which can cause the GATE output to stick high. Soldering a resistor into the R17 through-hole position (in parallel with the onboard 100kΩ R3) reduces preamp gain and fixes this. Start with 47kΩ; use a lower value (e.g. 33kΩ) if the sensor is still too sensitive for your environment.

### Sound Trigger Wiring

```
SEN-14262               Raspberry Pi           OPS243
┌───────────┐          ┌──────────┐          ┌──────────┐
│ VCC ──────┼──────────┤ 3.3V     │          │          │
│           │          │          │          │          │
│ GATE ─────┼──────────┼──────────┼──────────┤ HOST_INT │
│           │          │          │          │ (J3 P3)  │
│ GND ──────┼──────────┤ GND      ├──────────┤ GND      │
│           │          │          │          │ (J3 P1)  │
└───────────┘          └──────────┘          └──────────┘
```

See [sound-trigger-wiring.md](sound-trigger-wiring.md) for detailed instructions and troubleshooting.

## Angle Radar (K-LD7)

Two K-LD7 modules measure launch angle (vertical) and club path / aim direction (horizontal). The OPS243 handles speed; the K-LD7s provide **angle and distance only** (speed data aliases above 62 mph).

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **RFbeam K-LD7 (×2)** | 24 GHz FMCW radar for angle + distance | [RFbeam](https://rfbeam.ch/product/k-ld7-radar-transceiver/) | ~$60 ea |
| **FTDI USB-to-Serial adapter (×2)** | 3.3V FTDI board for K-LD7 UART (e.g. FT232RL) | [Amazon](https://www.amazon.com/s?k=ftdi+3.3v+usb+serial) | ~$10 |

> **EVAL board not required.** The K-LD7 bare module communicates over 3.3V UART (TX, RX, VCC, GND). Any 3.3V FTDI USB-to-serial adapter works. The official K-LD7 EVAL board (~$120 each) is only needed if you want the RFbeam GUI software for configuration — OpenFlight configures the radar over serial automatically.

### K-LD7 Connection

Each K-LD7 connects via a 3.3V FTDI adapter, appearing as `/dev/ttyUSB*` on Linux.

```
K-LD7 Module (UART) → FTDI 3.3V Adapter → USB → Raspberry Pi
```

One unit is mounted vertically (launch angle), one horizontally (club path / aim direction). A `--kld7-angle-offset` parameter corrects for mounting geometry — see the [setup guide](raspberry-pi-setup.md) for calibration.

## Power & Accessories

| Part | Description | Link | ~Price |
|------|-------------|------|--------|
| **27W USB-C Power Supply** | Official Pi 5 power supply (5V 5A) | [Adafruit](https://www.adafruit.com/product/5974) | $12 |
| MicroSD Card (32GB+) | For Pi OS and software | Any Class 10 | $10 |
| USB-A to Micro-USB Cable | For OPS243 radar connection | Any | $5 |

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
| Tripod Mount | For positioning the unit | 1/4"-20 mount | $10 |

---

## Cost Summary

| Category | ~Price |
|----------|--------|
| Core (OPS243, Pi 5, Display) | $355 |
| Sound Trigger (SEN-14262 + resistor + wires) | $18 |
| Angle Radar (2× K-LD7 + FTDI adapters) | $140 |
| Power & Accessories | $27 |
| **Total** | **~$540** |

> The angle radar is the most expensive component. OpenFlight works without it — you'll get ball speed, club speed, smash factor, and estimated carry. The K-LD7s add measured launch angle and club path data.
