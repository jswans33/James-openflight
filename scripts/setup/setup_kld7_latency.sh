#!/bin/bash
#
# Persistently set FTDI latency_timer=1ms for K-LD7 USB serial adapters.
#
# This installs a udev rule targeted at the current /dev/kld7_vertical and
# /dev/kld7_horizontal adapters by USB serial number, then applies the same
# latency value to currently connected devices.
#
# Usage:
#   sudo scripts/setup/setup_kld7_latency.sh
#   sudo scripts/setup/setup_kld7_latency.sh --latency 1
#   sudo scripts/setup/setup_kld7_latency.sh --all-ftdi
#   scripts/setup/setup_kld7_latency.sh --dry-run

set -euo pipefail

LATENCY_MS=1
RULE_FILE="/etc/udev/rules.d/99-openflight-kld7-latency.rules"
DRY_RUN=false
ALL_FTDI=false
DEVICES=("/dev/kld7_vertical" "/dev/kld7_horizontal")

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --latency MS       FTDI latency_timer value in ms (default: 1)
  --dev PATH         Add a device to target, e.g. /dev/ttyUSB0
  --all-ftdi         Install a broad rule for all FTDI USB serial adapters
  --rule-file PATH   Destination udev rule file
  --dry-run          Print the rule and current-device actions without writing
  --help             Show this help

Default targets are /dev/kld7_vertical and /dev/kld7_horizontal.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --latency)
            LATENCY_MS="$2"
            shift 2
            ;;
        --dev)
            DEVICES+=("$2")
            shift 2
            ;;
        --all-ftdi)
            ALL_FTDI=true
            shift
            ;;
        --rule-file)
            RULE_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$LATENCY_MS" =~ ^[0-9]+$ ]] || [ "$LATENCY_MS" -lt 1 ]; then
    echo "latency must be a positive integer in milliseconds" >&2
    exit 2
fi

if ! command -v udevadm >/dev/null 2>&1; then
    echo "udevadm not found; this script is intended for Linux/Pi systems" >&2
    exit 1
fi

run_root() {
    if [ "$DRY_RUN" = true ]; then
        printf '[dry-run] '
        printf '%q ' "$@"
        printf '\n'
    elif [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

device_serial() {
    local dev="$1"
    local serial
    serial="$(udevadm info -q property -n "$dev" 2>/dev/null | sed -n 's/^ID_SERIAL_SHORT=//p' | head -1)"
    if [ -n "$serial" ]; then
        echo "$serial"
        return 0
    fi
    udevadm info -a -n "$dev" 2>/dev/null \
        | sed -n 's/.*ATTRS{serial}=="\([^"]*\)".*/\1/p' \
        | head -1
}

sysfs_latency_path() {
    local dev="$1"
    local resolved
    resolved="$(readlink -f "$dev" 2>/dev/null || true)"
    if [ -z "$resolved" ]; then
        return 1
    fi
    local tty_name
    tty_name="$(basename "$resolved")"
    local path="/sys/bus/usb-serial/devices/${tty_name}/latency_timer"
    [ -e "$path" ] || return 1
    echo "$path"
}

rule_lines=()

if [ "$ALL_FTDI" = true ]; then
    rule_lines+=(
        "ACTION==\"add|change\", SUBSYSTEM==\"usb-serial\", DRIVERS==\"ftdi_sio\", ATTR{latency_timer}=\"$LATENCY_MS\""
    )
else
    seen_serials=""
    for dev in "${DEVICES[@]}"; do
        if [ ! -e "$dev" ]; then
            echo "Skipping missing device: $dev" >&2
            continue
        fi
        serial="$(device_serial "$dev")"
        if [ -z "$serial" ]; then
            echo "Could not determine USB serial for $dev; use --all-ftdi or --dev /dev/ttyUSBx" >&2
            continue
        fi
        case " $seen_serials " in
            *" $serial "*) continue ;;
        esac
        seen_serials="$seen_serials $serial"
        rule_lines+=(
            "ACTION==\"add|change\", SUBSYSTEM==\"usb-serial\", DRIVERS==\"ftdi_sio\", ATTRS{serial}==\"$serial\", ATTR{latency_timer}=\"$LATENCY_MS\""
        )
    done
fi

if [ "${#rule_lines[@]}" -eq 0 ]; then
    echo "No K-LD7 FTDI devices found. Check /dev/kld7_* symlinks or pass --all-ftdi." >&2
    exit 1
fi

tmp_rule="$(mktemp)"
{
    echo "# OpenFlight K-LD7 FTDI low-latency serial rule"
    echo "# Installed by scripts/setup/setup_kld7_latency.sh"
    echo "# Keeps RADC packet timestamps less exposed to FTDI buffering."
    for line in "${rule_lines[@]}"; do
        echo "$line"
    done
} > "$tmp_rule"

echo "Installing udev rule:"
echo "  $RULE_FILE"
echo
cat "$tmp_rule"
echo

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] would install $RULE_FILE"
else
    run_root install -m 0644 "$tmp_rule" "$RULE_FILE"
fi
rm -f "$tmp_rule"

echo "Applying latency_timer=$LATENCY_MS to currently connected target devices..."
for dev in "${DEVICES[@]}"; do
    [ -e "$dev" ] || continue
    path="$(sysfs_latency_path "$dev" || true)"
    if [ -z "$path" ]; then
        echo "  $dev: latency_timer not exposed"
        continue
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] echo $LATENCY_MS > $path"
        continue
    fi
    echo "$LATENCY_MS" | run_root tee "$path" >/dev/null
    echo "  $dev -> $(readlink -f "$dev"): $(cat "$path")ms ($path)"
done

run_root udevadm control --reload-rules
run_root udevadm trigger --subsystem-match=usb-serial --action=change

echo
echo "Done. Replug the K-LD7 adapters or reboot to verify persistence."
echo "Runtime logs should show: [KLD7:*] USB serial latency_timer=${LATENCY_MS}ms"
