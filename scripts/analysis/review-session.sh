#!/usr/bin/env bash
# Review a K-LD7 session log and export shot-profile plots.
# Usage: ./scripts/review-session.sh session_logs/session_*.jsonl [--clean]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <session.jsonl> [--clean]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

uv run --no-project --with numpy --with matplotlib \
    python "$SCRIPT_DIR/review_kld7_session.py" "$@"
