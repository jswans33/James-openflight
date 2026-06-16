#!/usr/bin/env bash
# Analyze a K-LD7 RADC capture.
# Usage: ./scripts/analyze-radc.sh capture.pkl [--shot-windows] [--csv]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <capture.pkl> [--shot-windows] [--csv]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

uv run --no-project --with numpy --with matplotlib \
    python "$SCRIPT_DIR/analyze_kld7_radc.py" "$@"
