#!/usr/bin/env python
"""Live-sync K-LD7 frame reports from a Pi session log.

The Pi should stay focused on kiosk capture. This helper runs on the Mac,
polls the Pi over SSH for the newest session log, SCPs it only when the shot
count changes, regenerates the K-LD7 geometry report locally, and optionally
serves the report directory for the timing visualizer's auto-reload mode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "kld7_geometry_selection_report.py"
VISUALIZER = REPO_ROOT / "scripts" / "analysis" / "kld7_timing_shift_visualizer.html"


def _run(command: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _remote_latest_session(pi_host: str, session_dir: str, connect_timeout: int) -> dict[str, str]:
    remote_cmd = (
        "set -e; "
        f"latest=$(ls -t {session_dir.rstrip('/')}/session_*.jsonl 2>/dev/null | head -n 1); "
        'if [ -z "$latest" ]; then exit 2; fi; '
        'shots=$(grep -c \'"type": "shot_detected"\' "$latest" || true); '
        'size=$(stat -c %s "$latest" 2>/dev/null || wc -c < "$latest"); '
        'mtime=$(stat -c %Y "$latest" 2>/dev/null || echo 0); '
        'printf "%s\\t%s\\t%s\\t%s\\n" "$latest" "$shots" "$size" "$mtime"'
    )
    result = _run(
        ["ssh", "-o", f"ConnectTimeout={connect_timeout}", pi_host, remote_cmd],
        timeout=max(connect_timeout + 5, 10),
    )
    parts = result.stdout.strip().split("\t")
    if len(parts) != 4:
        raise RuntimeError(f"Unexpected SSH response: {result.stdout!r}")
    return {
        "remote_path": parts[0],
        "shot_count": parts[1],
        "size": parts[2],
        "mtime": parts[3],
    }


def _scp_session(pi_host: str, remote_path: str, local_path: Path) -> None:
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    _run(["scp", f"{pi_host}:{remote_path}", str(tmp_path)])
    tmp_path.replace(local_path)


def _copy_visualizer(output_dir: Path) -> None:
    if VISUALIZER.exists():
        shutil.copy2(VISUALIZER, output_dir / "index.html")


def _run_report(args: argparse.Namespace, session_path: Path, output_dir: Path) -> None:
    command = [
        sys.executable,
        str(REPORT_SCRIPT),
        str(session_path),
        "--output-dir",
        str(output_dir),
        "--orientation",
        args.orientation,
        "--ball-distance-ft",
        str(args.ball_distance_ft),
        "--mount-deg",
        str(args.mount_deg),
        "--angle-offset-deg",
        str(args.angle_offset_deg),
        "--ball-above-radar-ft",
        str(args.ball_above_radar_ft),
    ]
    if args.report_arg:
        command.extend(args.report_arg)
    _run(command)


def _write_state(
    output_dir: Path,
    *,
    pi_host: str,
    remote_info: dict[str, str],
    local_session: Path,
    report_started_at: float,
) -> None:
    state = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_seconds": round(time.time() - report_started_at, 3),
        "pi_host": pi_host,
        "remote_session": remote_info["remote_path"],
        "local_session": str(local_session),
        "shot_count": int(remote_info["shot_count"]),
        "remote_size_bytes": int(remote_info["size"]),
        "remote_mtime": int(remote_info["mtime"]),
        "frames_live_csv": "frames_live.csv",
        "shots_live_csv": "shots_live.csv",
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _serve_directory(output_dir: Path, port: int) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def sync_once(
    args: argparse.Namespace, last_key: tuple[str, str, str] | None
) -> tuple[str, str, str] | None:
    remote_info = _remote_latest_session(args.pi_host, args.session_dir, args.connect_timeout)
    key = (remote_info["remote_path"], remote_info["shot_count"], remote_info["size"])
    shot_count = int(remote_info["shot_count"])

    if shot_count <= 0:
        print(
            f"[live-sync] Latest session has no shots yet: {remote_info['remote_path']}",
            flush=True,
        )
        return key

    if key == last_key:
        return last_key

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_session = args.output_dir / Path(remote_info["remote_path"]).name
    latest_session = args.output_dir / "latest_session.jsonl"

    print(
        f"[live-sync] Pulling shot_count={shot_count} size={int(remote_info['size']) / 1_000_000:.1f}MB "
        f"from {remote_info['remote_path']}",
        flush=True,
    )
    report_started_at = time.time()
    _scp_session(args.pi_host, remote_info["remote_path"], local_session)
    shutil.copy2(local_session, latest_session)
    _run_report(args, latest_session, args.output_dir)
    _copy_visualizer(args.output_dir)
    _write_state(
        args.output_dir,
        pi_host=args.pi_host,
        remote_info=remote_info,
        local_session=latest_session,
        report_started_at=report_started_at,
    )
    print(
        f"[live-sync] Updated {args.output_dir / 'frames_live.csv'} "
        f"({time.time() - report_started_at:.1f}s)",
        flush=True,
    )
    return key


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SCP the latest Pi session and regenerate live K-LD7 frame CSVs."
    )
    parser.add_argument(
        "--pi-host",
        required=True,
        help="SSH destination for the Pi, for example openflight@openflight.local.",
    )
    parser.add_argument("--session-dir", default="~/openflight_sessions")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "openflight_sessions" / "live_kld7_report",
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--connect-timeout", type=int, default=8)
    parser.add_argument("--once", action="store_true", help="Run one sync check and exit.")
    parser.add_argument("--serve-port", type=int, default=8765)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--orientation", default="vertical", choices=["vertical", "horizontal"])
    parser.add_argument("--ball-distance-ft", type=float, default=5.0)
    parser.add_argument("--mount-deg", type=float, default=10.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--ball-above-radar-ft", type=float, default=-4.0 / 12.0)
    parser.add_argument(
        "--report-arg",
        action="append",
        default=[],
        help="Extra argument passed through to kld7_geometry_selection_report.py. Repeat as needed.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    server = None
    if not args.no_serve:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _copy_visualizer(args.output_dir)
        server = _serve_directory(args.output_dir, args.serve_port)
        print(
            "[live-sync] Visualizer: "
            f"http://127.0.0.1:{args.serve_port}/index.html?csv=frames_live.csv&auto=1",
            flush=True,
        )

    last_key = None
    try:
        while True:
            try:
                last_key = sync_once(args, last_key)
            except subprocess.CalledProcessError as error:
                message = error.stderr.strip() or error.stdout.strip() or str(error)
                print(f"[live-sync] Command failed: {message}", file=sys.stderr, flush=True)
            except Exception as error:
                print(f"[live-sync] {error}", file=sys.stderr, flush=True)

            if args.once:
                break
            time.sleep(max(args.interval, 0.5))
    finally:
        if server:
            server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
