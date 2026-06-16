"""Replay a session log through a proposed MEDIAN club-speed picker and
report before/after vs the logged (magnitude-pick) club speed.

Background
----------
The production `find_club_speed` in `rolling_buffer/processor.py` picks
the candidate reading with the **highest magnitude** in the 100 ms
pre-ball window. On a behind-the-ball setup this systematically
under-reports club speed by 5-15 mph because the clubhead's radar
return peaks when it crosses beam center (mid-swing), not at impact.
By the time the club is approaching the ball, it's in marginal beam
coverage — speed is higher but magnitude is lower, so the mid-swing
reading wins.

This script evaluates an alternative picker without changing
production code, so it can be reviewed against any existing session
log:

    MEDIAN strategy
        1. Build the same candidate set the production picker uses
           (outbound, 67-85 % of ball speed, within 100 ms of ball,
           not the ball itself).
        2. Apply a magnitude floor: keep only candidates whose
           magnitude is ≥ 30 % of the peak candidate magnitude.
           Rejects sidelobe spikes.
        3. Pick the median speed across surviving candidates.
           Pair it with the timestamp of the candidate whose speed
           is closest to the median (so we keep a usable timestamp).

Why MEDIAN
----------
Per-shot inspection of capture data shows the candidate set is
bimodal:
    - Mid-swing cluster: 13-18 ms before ball, high magnitude,
      lower-speed (beam center).
    - Approach cluster: 5-12 ms before ball, lower magnitude,
      higher-speed (clubhead descending toward impact).
TrackMan's impact-speed truth tends to sit at the boundary between
the two clusters. The median naturally lands near that boundary
without needing tuned time windows or smash brackets.

Usage
-----
    .venv/bin/python scripts/analysis/replay_club_speed.py \\
        session_logs/session_<timestamp>_range.jsonl

    .venv/bin/python scripts/analysis/replay_club_speed.py \\
        session_logs/session_<timestamp>_range.jsonl --mag-floor 0.5

    .venv/bin/python scripts/analysis/replay_club_speed.py \\
        session_logs/session_<timestamp>_range.jsonl --csv out.csv

What it prints
--------------
Per-shot:  ball speed, OLD club / OLD smash, NEW club / NEW smash, Δ.
Aggregate: mean / median / range of NEW - OLD across all shots.

The smash-factor columns are an internal sanity check: physical iron
smash factors are 1.18-1.42. OLD often produces > 1.45 (impossible
contact quality) on shots where the picker has landed on a mid-swing
sidelobe; NEW pulls these back to plausible ranges.
"""
# ruff: noqa: I001
# Import ordering is intentional: openflight package needs path setup
# and lightweight stubs before importing the processor on dev machines
# without pyserial / picamera2 installed.

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

if "openflight" not in sys.modules:
    pkg = types.ModuleType("openflight")
    pkg.__path__ = [str(REPO_ROOT / "src" / "openflight")]
    sys.modules["openflight"] = pkg
if "openflight.rolling_buffer" not in sys.modules:
    rb_pkg = types.ModuleType("openflight.rolling_buffer")
    rb_pkg.__path__ = [str(REPO_ROOT / "src" / "openflight" / "rolling_buffer")]
    sys.modules["openflight.rolling_buffer"] = rb_pkg

ops243_stub = types.ModuleType("openflight.ops243")


class _Stub: ...


for _name in ("SpeedReading", "OPS243Radar", "SpeedUnit", "Direction"):
    setattr(ops243_stub, _name, _Stub)
sys.modules.setdefault("openflight.ops243", ops243_stub)

_processor_spec = importlib.util.spec_from_file_location(
    "openflight.rolling_buffer.processor",
    REPO_ROOT / "src" / "openflight" / "rolling_buffer" / "processor.py",
)
_processor_mod = importlib.util.module_from_spec(_processor_spec)
sys.modules["openflight.rolling_buffer.processor"] = _processor_mod
_processor_spec.loader.exec_module(_processor_mod)
IQCapture = _processor_mod.IQCapture
RollingBufferProcessor = _processor_mod.RollingBufferProcessor


# Production-picker defaults from find_club_speed in processor.py.
SPEED_FLOOR_FRAC = 0.67  # smash 1.50 ceiling
SPEED_CEILING_FRAC = 0.85  # smash 1.18 floor
MAX_WINDOW_MS = 100.0


def median_club_speed(
    timeline,
    ball_speed_mph: float,
    ball_timestamp_ms: float,
    *,
    mag_floor_frac: float = 0.30,
):
    """Median speed of candidates above the magnitude floor.

    Returns (speed_mph, timestamp_ms) or (None, None).
    """
    min_club = ball_speed_mph * SPEED_FLOOR_FRAC
    max_club = ball_speed_mph * SPEED_CEILING_FRAC
    pre_ball = [r for r in timeline.readings if r.timestamp_ms <= ball_timestamp_ms]
    candidates = [
        r
        for r in pre_ball
        if r.is_outbound
        and min_club <= r.speed_mph <= max_club
        and ball_timestamp_ms - r.timestamp_ms <= MAX_WINDOW_MS
        and abs(r.speed_mph - ball_speed_mph) > 1.0
    ]
    if not candidates:
        return None, None
    peak_mag = max(r.magnitude for r in candidates)
    eligible = [r for r in candidates if r.magnitude >= peak_mag * mag_floor_frac]
    if not eligible:
        return None, None
    median = float(np.median([r.speed_mph for r in eligible]))
    # Pair the median speed with the timestamp of the closest reading.
    pick = min(eligible, key=lambda r: abs(r.speed_mph - median))
    return median, pick.timestamp_ms


def iter_captures(session_path: Path):
    """Yield rolling_buffer_capture entries from a session log."""
    with session_path.open() as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "rolling_buffer_capture":
                yield entry


def replay_one(processor, entry: dict, mag_floor_frac: float) -> dict | None:
    """Run MEDIAN on a single capture entry, reusing logged ball metadata."""
    if "i_samples" not in entry or "q_samples" not in entry:
        return None
    ball_speed = entry.get("ball_speed_mph")
    ball_ts = entry.get("ball_timestamp_ms")
    if ball_speed is None or ball_ts is None:
        return None

    capture = IQCapture(
        sample_time=float(entry.get("sample_time", 0.0)),
        trigger_time=float(entry.get("trigger_time", 0.0)),
        i_samples=entry["i_samples"],
        q_samples=entry["q_samples"],
    )
    timeline = processor.process_overlapping(capture)
    new_speed, new_ts = median_club_speed(
        timeline,
        float(ball_speed),
        float(ball_ts),
        mag_floor_frac=mag_floor_frac,
    )
    return {
        "shot": entry.get("shot_number"),
        "ball_speed": float(ball_speed),
        "old_club": entry.get("club_speed_mph"),
        "old_club_ts": entry.get("club_timestamp_ms"),
        "new_club": new_speed,
        "new_club_ts": new_ts,
    }


def _fmt(v, width: int = 9, fmt: str = ".2f") -> str:
    if v is None:
        return f"{'—':>{width}}"
    return format(v, f">{width}{fmt}")


def _fmt_signed(v, width: int = 8) -> str:
    if v is None:
        return f"{'—':>{width}}"
    return format(v, f">+{width}.2f")


def print_table(rows: list[dict], session_name: str) -> None:
    print(f"Session: {session_name}")
    print(
        f"{'shot':>4} {'ball':>7} {'old_club':>9} {'old_smash':>10} "
        f"{'new_club':>9} {'new_smash':>10} {'Δ_club':>8}"
    )
    print("-" * 65)
    for r in rows:
        old, new, ball = r["old_club"], r["new_club"], r["ball_speed"]
        old_smash = ball / old if old else None
        new_smash = ball / new if new else None
        delta = (new - old) if (new is not None and old is not None) else None
        print(
            f"{r['shot']:>4} {ball:>7.2f} {_fmt(old)} {_fmt(old_smash, 10, '.3f')} "
            f"{_fmt(new)} {_fmt(new_smash, 10, '.3f')} {_fmt_signed(delta)}"
        )


def aggregate(rows: list[dict]) -> None:
    deltas = [
        r["new_club"] - r["old_club"]
        for r in rows
        if r["new_club"] is not None and r["old_club"] is not None
    ]
    old_smash = [r["ball_speed"] / r["old_club"] for r in rows if r["old_club"]]
    new_smash = [r["ball_speed"] / r["new_club"] for r in rows if r["new_club"]]
    print()
    if deltas:
        print(
            f"Δ club speed (new - old): n={len(deltas)}  "
            f"mean={np.mean(deltas):+.2f}  median={np.median(deltas):+.2f}  "
            f"min={min(deltas):+.2f}  max={max(deltas):+.2f}"
        )
    impl_old = sum(1 for s in old_smash if s > 1.42)
    impl_new = sum(1 for s in new_smash if s > 1.42)
    print(
        f"Implausibly high smash (> 1.42, iron/wedge max): "
        f"OLD {impl_old}/{len(old_smash)}   NEW {impl_new}/{len(new_smash)}"
    )


def write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "shot",
                "ball_speed_mph",
                "old_club_speed_mph",
                "old_club_timestamp_ms",
                "old_smash",
                "new_club_speed_mph",
                "new_club_timestamp_ms",
                "new_smash",
                "delta_club_mph",
            ]
        )
        for r in rows:
            old, new, ball = r["old_club"], r["new_club"], r["ball_speed"]
            w.writerow(
                [
                    r["shot"],
                    ball,
                    old,
                    r["old_club_ts"],
                    (ball / old) if old else "",
                    new,
                    r["new_club_ts"],
                    (ball / new) if new else "",
                    (new - old) if (new is not None and old is not None) else "",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="path to session_*.jsonl")
    parser.add_argument(
        "--mag-floor",
        type=float,
        default=0.30,
        help="magnitude floor as fraction of peak candidate magnitude (default 0.30)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional CSV output path",
    )
    args = parser.parse_args()
    if not args.session.exists():
        print(f"Session not found: {args.session}", file=sys.stderr)
        return 2

    processor = RollingBufferProcessor()
    rows: list[dict] = []
    for entry in iter_captures(args.session):
        result = replay_one(processor, entry, args.mag_floor)
        if result is not None:
            rows.append(result)

    if not rows:
        print("No replayable rolling_buffer_capture entries found.")
        return 1

    print_table(rows, args.session.name)
    aggregate(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
