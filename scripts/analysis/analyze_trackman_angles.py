#!/usr/bin/env python3
"""Analyze OpenFlight launch-angle accuracy against a TrackMan comparison CSV.

This complements compare_trackman.py by joining the comparison rows back to the
OpenFlight JSONL log so angle errors can be sliced by angle source, confidence,
K-LD7 frame count, and simple candidate gating policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ErrorStats:
    n: int
    bias: float
    mae: float
    rmse: float
    p90_abs: float
    corr: float | None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(p * (len(ordered) - 1))))
    return ordered[idx]


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    cov = statistics.fmean((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def _stats(pairs: list[tuple[float, float]]) -> ErrorStats | None:
    if not pairs:
        return None
    of_vals = [p[0] for p in pairs]
    tm_vals = [p[1] for p in pairs]
    deltas = [of_v - tm_v for of_v, tm_v in pairs]
    return ErrorStats(
        n=len(deltas),
        bias=statistics.fmean(deltas),
        mae=statistics.fmean(abs(d) for d in deltas),
        rmse=math.sqrt(statistics.fmean(d * d for d in deltas)),
        p90_abs=_percentile([abs(d) for d in deltas], 0.9),
        corr=_corr(of_vals, tm_vals),
    )


def _fmt_stats(stats: ErrorStats | None) -> str:
    if stats is None:
        return "| 0 |  |  |  |  |  |"
    corr = "" if stats.corr is None else f"{stats.corr:+.2f}"
    return (
        f"| {stats.n} | {stats.bias:+.2f} | {stats.mae:.2f} | "
        f"{stats.rmse:.2f} | {stats.p90_abs:.2f} | {corr} |"
    )


def load_openflight_details(path: Path) -> dict[int, dict[str, Any]]:
    shots: dict[int, dict[str, Any]] = {}
    pending_buffers: dict[tuple[int, str], dict[str, Any]] = {}

    def buffer_summary(entry: dict[str, Any]) -> dict[str, Any]:
        angle = entry.get("ball_angle") or {}
        return {
            "frame_count": entry.get("frame_count"),
            "angle": angle,
            "club_angle": entry.get("club_angle"),
        }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = entry.get("type")
            shot_number = _to_int(entry.get("shot_number"))
            if shot_number is None:
                continue

            if entry_type == "kld7_buffer":
                orientation = entry.get("orientation")
                if orientation in {"vertical", "horizontal"}:
                    pending_buffers[(shot_number, orientation)] = buffer_summary(entry)
                continue

            if entry_type != "shot_detected":
                continue

            shot = dict(entry)
            for orientation in ("vertical", "horizontal"):
                shot[f"{orientation}_buffer"] = pending_buffers.get(
                    (shot_number, orientation)
                )
            shots[shot_number] = shot

    return shots


def load_rows(comparison: Path, shots: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with comparison.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("match_quality") != "good":
                continue
            shot_number = _to_int(row.get("shot_number_of"))
            if shot_number is None:
                continue
            detail = shots.get(shot_number, {})
            row["_shot_number"] = shot_number
            row["_detail"] = detail
            rows.append(row)
    return rows


def angle_pair(row: dict[str, Any], axis: str) -> tuple[float, float] | None:
    of_value = _to_float(row.get(f"launch_{axis}_of"))
    tm_value = _to_float(row.get(f"launch_{axis}_tm"))
    if of_value is None or tm_value is None:
        return None
    return of_value, tm_value


def detail_value(row: dict[str, Any], key: str) -> Any:
    detail = row.get("_detail") or {}
    return detail.get(key)


def kld7_value(row: dict[str, Any], orientation: str, key: str) -> Any:
    detail = row.get("_detail") or {}
    buffer = detail.get(f"{orientation}_buffer") or {}
    angle = buffer.get("angle") or {}
    return angle.get(key)


def _bucket_conf(conf: float | None) -> str:
    if conf is None:
        return "missing"
    if conf < 0.5:
        return "<0.5"
    if conf < 0.7:
        return "0.5-0.7"
    return ">=0.7"


def _bucket_frames(frames: float | None) -> str:
    if frames is None:
        return "missing"
    if frames <= 1:
        return "1"
    if frames <= 3:
        return "2-3"
    return "4+"


def grouped_stats(
    rows: list[dict[str, Any]],
    axis: str,
    label_fn,
) -> dict[str, ErrorStats | None]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        pair = angle_pair(row, axis)
        if pair is None:
            continue
        grouped[label_fn(row)].append(pair)
    return {label: _stats(pairs) for label, pairs in sorted(grouped.items())}


def gate_rows(rows: list[dict[str, Any]], axis: str, policy: str) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        pair = angle_pair(row, axis)
        if pair is None:
            continue
        if policy == "all":
            kept.append(row)
            continue
        if axis == "v":
            source = detail_value(row, "angle_source")
            conf = _to_float(kld7_value(row, "vertical", "confidence"))
            frames = _to_float(kld7_value(row, "vertical", "num_frames"))
        else:
            source = detail_value(row, "angle_source")
            conf = _to_float(kld7_value(row, "horizontal", "confidence"))
            frames = _to_float(kld7_value(row, "horizontal", "num_frames"))

        if policy == "radar-only" and source == "radar":
            kept.append(row)
        elif policy == "conf>=0.7" and source == "radar" and conf is not None and conf >= 0.7:
            kept.append(row)
        elif policy == "frames>=4" and source == "radar" and frames is not None and frames >= 4:
            kept.append(row)
    return kept


def write_report(rows: list[dict[str, Any]], output: Path) -> None:
    lines: list[str] = [
        "# TrackMan Angle Analysis",
        "",
        f"Good matched pairs: {len(rows)}",
        "",
        "## Overall",
        "",
        "| Axis | n | Bias OF-TM | MAE | RMSE | P90 abs | Corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for axis, label in (("v", "Vertical launch"), ("h", "Horizontal launch")):
        pairs = [p for row in rows if (p := angle_pair(row, axis)) is not None]
        lines.append(f"| {label} {_fmt_stats(_stats(pairs))}")

    for axis, label, orientation in (
        ("v", "Vertical Launch", "vertical"),
        ("h", "Horizontal Launch", "horizontal"),
    ):
        lines.extend([
            "",
            f"## {label} Slices",
            "",
            "### By Club",
            "",
            "| Group | n | Bias OF-TM | MAE | RMSE | P90 abs | Corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for group, stats in grouped_stats(rows, axis, lambda r: r.get("club") or "(none)").items():
            lines.append(f"| {group} {_fmt_stats(stats)}")

        lines.extend([
            "",
            "### By OpenFlight Source",
            "",
            "| Group | n | Bias OF-TM | MAE | RMSE | P90 abs | Corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for group, stats in grouped_stats(
            rows, axis, lambda r: detail_value(r, "angle_source") or "(none)"
        ).items():
            lines.append(f"| {group} {_fmt_stats(stats)}")

        lines.extend([
            "",
            "### By K-LD7 Confidence",
            "",
            "| Group | n | Bias OF-TM | MAE | RMSE | P90 abs | Corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for group, stats in grouped_stats(
            rows,
            axis,
            lambda r, orient=orientation: _bucket_conf(
                _to_float(kld7_value(r, orient, "confidence"))
            ),
        ).items():
            lines.append(f"| {group} {_fmt_stats(stats)}")

        lines.extend([
            "",
            "### By K-LD7 Frames",
            "",
            "| Group | n | Bias OF-TM | MAE | RMSE | P90 abs | Corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for group, stats in grouped_stats(
            rows,
            axis,
            lambda r, orient=orientation: _bucket_frames(
                _to_float(kld7_value(r, orient, "num_frames"))
            ),
        ).items():
            lines.append(f"| {group} {_fmt_stats(stats)}")

        lines.extend([
            "",
            "### Candidate Gating Policies",
            "",
            "| Policy | n | Bias OF-TM | MAE | RMSE | P90 abs | Corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for policy in ("all", "radar-only", "conf>=0.7", "frames>=4"):
            gated = gate_rows(rows, axis, policy)
            pairs = [p for row in gated if (p := angle_pair(row, axis)) is not None]
            lines.append(f"| {policy} {_fmt_stats(_stats(pairs))}")

    lines.extend([
        "",
        "## Worst Errors",
        "",
    ])
    for axis, label in (("v", "Vertical"), ("h", "Horizontal")):
        errors: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            pair = angle_pair(row, axis)
            if pair is None:
                continue
            errors.append((pair[0] - pair[1], row))
        errors.sort(key=lambda item: abs(item[0]), reverse=True)
        lines.extend([
            f"### {label}",
            "",
            "| OF shot | Club | OF | TM | Delta | Source | Conf | Frames |",
            "|---:|---|---:|---:|---:|---|---:|---:|",
        ])
        orient = "vertical" if axis == "v" else "horizontal"
        for delta, row in errors[:10]:
            pair = angle_pair(row, axis)
            assert pair is not None
            lines.append(
                "| "
                f"{row['_shot_number']} | {row.get('club') or ''} | "
                f"{pair[0]:.1f} | {pair[1]:.1f} | {delta:+.1f} | "
                f"{detail_value(row, 'angle_source') or ''} | "
                f"{_to_float(kld7_value(row, orient, 'confidence')) or 0:.2f} | "
                f"{_to_int(kld7_value(row, orient, 'num_frames')) or 0} |"
            )
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--openflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shots = load_openflight_details(args.openflight)
    rows = load_rows(args.comparison, shots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(rows, args.output)
    print(f"Wrote {args.output} ({len(rows)} good pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
