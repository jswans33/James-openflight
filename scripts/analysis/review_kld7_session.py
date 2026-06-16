#!/usr/bin/env python3
"""Review a K-LD7 session JSONL file and export shot-profile plots."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from kld7_session_review_lib import ShotReview, analyze_session  # noqa: E402

PROFILE_GRID_MS = np.arange(0.0, 261.0, 10.0)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_pyplot():
    """Import matplotlib only when plot export is requested."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def session_output_dir(session_path: Path) -> Path:
    """Default output directory for generated review artifacts."""
    return REPO_ROOT / "shots" / f"session_review_{session_path.stem}"


def _is_safe_review_output_dir(path: Path) -> bool:
    """Allow cleanup only for session review directories under shots/."""
    try:
        resolved = path.resolve()
        shots_root = (REPO_ROOT / "shots").resolve()
        resolved.relative_to(shots_root)
    except ValueError:
        return False
    return resolved.name.startswith("session_review_") and resolved.parent == shots_root


def ensure_output_dir(path: Path, *, clean: bool) -> None:
    """Create the output directory and optionally remove prior generated files."""
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not clean:
        return
    if not _is_safe_review_output_dir(path):
        raise ValueError(
            f"Refusing to clean unsafe output directory: {path}. "
            "Use a directory under shots/session_review_* or omit --clean."
        )
    for child in path.iterdir():
        if child.is_dir():
            raise ValueError(
                f"Output directory contains nested paths and cannot be cleaned safely: {child}"
            )
        child.unlink()


def plot_shot_profile(result: ShotReview, session_name: str, output_path: Path) -> None:
    """Plot one reviewed shot path against all post-impact detections."""
    plt = _load_pyplot()
    path = result.anchor.path
    times = np.array([d.time_ms for d in path], dtype=float)
    distances = np.array([d.distance_m for d in path], dtype=float)
    angles = np.array([d.angle_deg for d in path], dtype=float)
    magnitudes = np.array([d.magnitude for d in path], dtype=float)

    background_times = np.array(
        [d.time_ms for d in result.anchor.all_post_hits],
        dtype=float,
    )
    background_distances = np.array(
        [d.distance_m for d in result.anchor.all_post_hits],
        dtype=float,
    )

    fig, (distance_axis, magnitude_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    if len(background_times):
        distance_axis.scatter(
            background_times,
            background_distances,
            s=20,
            color="0.82",
            alpha=0.55,
            label="all post-impact hits",
            zorder=1,
        )

    scatter = distance_axis.scatter(
        times,
        distances,
        s=np.maximum(90.0, magnitudes / 22.0),
        c=magnitudes,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
        label="selected path",
    )
    distance_axis.plot(times, distances, color="black", linewidth=2.1, zorder=2)

    for index, detection in enumerate(path):
        distance_axis.annotate(
            str(index),
            (detection.time_ms, detection.distance_m),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=9,
        )

    angle_axis = distance_axis.twinx()
    angle_axis.plot(
        times,
        angles,
        color="tab:orange",
        linestyle="--",
        marker="o",
        linewidth=1.5,
    )
    angle_axis.set_ylabel("angle (deg)", color="tab:orange")
    angle_axis.tick_params(axis="y", labelcolor="tab:orange")

    distance_axis.set_title(
        f"{session_name} | shot {result.shot_number:02d} | "
        f"{result.club_label} | launch conf {result.logged_launch_confidence or 0.0:.2f}"
    )
    distance_axis.set_ylabel("distance (m)")
    max_background = float(background_distances.max()) if len(background_distances) else 0.0
    distance_axis.set_xlim(-10.0, max(150.0, float(times[-1] + 40.0)))
    distance_axis.set_ylim(
        0.0,
        max(5.4, float(max(max_background, float(np.max(distances))) + 0.4)),
    )
    distance_axis.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
    distance_axis.grid(True, alpha=0.25)
    distance_axis.legend(loc="upper left")

    summary_lines = [
        f"quality: {result.quality}",
        (
            f"path: {result.anchor.metrics.point_count} pts, "
            f"{result.anchor.metrics.distance_gain_m:.2f} m gain, "
            f"{result.anchor.metrics.duration_ms:.0f} ms"
        ),
        f"anchor: club event {result.anchor.club_event_index}, lingering {result.anchor.lingering_hits}",
    ]
    if result.logged_launch_angle_deg is not None:
        summary_lines.append(f"logged launch: {result.logged_launch_angle_deg:.1f} deg")
    if result.rolling_ball_dt_ms is not None:
        summary_lines.append(f"rolling ball-club dt: {result.rolling_ball_dt_ms:.1f} ms")
    distance_axis.text(
        0.015,
        0.98,
        "\n".join(summary_lines),
        transform=distance_axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
    )

    colorbar = fig.colorbar(scatter, ax=distance_axis, pad=0.02)
    colorbar.set_label("magnitude (raw)")

    magnitude_axis.plot(times, magnitudes, color="tab:green", marker="s", linewidth=1.7)
    magnitude_axis.set_xlabel("time from inferred impact (ms)")
    magnitude_axis.set_ylabel("magnitude")
    magnitude_axis.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def interpolate_profile(result: ShotReview) -> np.ndarray:
    """Interpolate a reviewed path onto a fixed time grid."""
    interpolated = np.full(PROFILE_GRID_MS.shape, np.nan, dtype=float)
    times = np.array([d.time_ms for d in result.anchor.path], dtype=float)
    distances = np.array([d.distance_m for d in result.anchor.path], dtype=float)
    if len(times) == 1:
        nearest = int(np.argmin(np.abs(PROFILE_GRID_MS - times[0])))
        interpolated[nearest] = distances[0]
        return interpolated
    mask = (PROFILE_GRID_MS >= times[0]) & (PROFILE_GRID_MS <= times[-1])
    interpolated[mask] = np.interp(PROFILE_GRID_MS[mask], times, distances)
    return interpolated


def aggregate_band(
    results: list[ShotReview],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate all reviewed paths into median and IQR bands."""
    stacked = np.vstack([interpolate_profile(result) for result in results])
    counts = np.sum(~np.isnan(stacked), axis=0)
    median = np.full(PROFILE_GRID_MS.shape, np.nan, dtype=float)
    q25 = np.full(PROFILE_GRID_MS.shape, np.nan, dtype=float)
    q75 = np.full(PROFILE_GRID_MS.shape, np.nan, dtype=float)
    for index, count in enumerate(counts):
        if count == 0:
            continue
        values = stacked[:, index]
        values = values[~np.isnan(values)]
        median[index] = float(np.median(values))
        q25[index] = float(np.percentile(values, 25))
        q75[index] = float(np.percentile(values, 75))
    return counts, median, q25, q75


def plot_overlay(results: list[ShotReview], session_name: str, output_path: Path) -> None:
    """Plot all selected shot profiles on one overlay."""
    plt = _load_pyplot()
    fig, axis = plt.subplots(figsize=(12, 7))
    quality_colors = {"strong": "tab:blue", "partial": "tab:orange", "weak": "tab:red"}

    for result in results:
        times = np.array([d.time_ms for d in result.anchor.path], dtype=float)
        distances = np.array([d.distance_m for d in result.anchor.path], dtype=float)
        axis.plot(
            times,
            distances,
            marker="o",
            linewidth=1.8,
            alpha=0.78,
            color=quality_colors[result.quality],
        )
        axis.annotate(
            f"{result.shot_number:02d}",
            (times[-1], distances[-1]),
            textcoords="offset points",
            xytext=(5, 2),
            fontsize=8,
        )

    counts, median, q25, q75 = aggregate_band(results)
    band_mask = counts >= 3
    line_mask = counts >= 2
    axis.fill_between(
        PROFILE_GRID_MS[band_mask],
        q25[band_mask],
        q75[band_mask],
        color="gold",
        alpha=0.28,
        label="median IQR",
    )
    axis.plot(
        PROFILE_GRID_MS[line_mask],
        median[line_mask],
        color="black",
        linewidth=3.0,
        label="median path",
    )

    axis.set_title(f"{session_name} | extracted shot profiles")
    axis.set_xlabel("time from inferred impact (ms)")
    axis.set_ylabel("distance (m)")
    axis.set_xlim(0.0, float(PROFILE_GRID_MS[-1]))
    axis.set_ylim(0.0, 5.4)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    axis.text(
        0.015,
        0.98,
        "labels show shot number\nblue=strong, orange=partial, red=weak",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_launch_angle_review(results: list[ShotReview], output_path: Path) -> None:
    """Compare logged launch angles against extracted path angles."""
    plt = _load_pyplot()
    shots = np.array([result.shot_number for result in results], dtype=float)
    logged = np.array(
        [
            np.nan if result.logged_launch_angle_deg is None else result.logged_launch_angle_deg
            for result in results
        ],
        dtype=float,
    )
    start = np.array(
        [result.anchor.metrics.start_angle_deg for result in results],
        dtype=float,
    )
    mean = np.array(
        [result.anchor.metrics.mean_angle_deg for result in results],
        dtype=float,
    )
    end = np.array(
        [result.anchor.metrics.end_angle_deg for result in results],
        dtype=float,
    )

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(shots, logged, marker="o", linewidth=2.0, color="black", label="logged launch angle")
    axis.plot(shots, start, marker="s", linewidth=1.4, color="tab:blue", label="path start angle")
    axis.plot(shots, mean, marker="^", linewidth=1.4, color="tab:orange", label="path mean angle")
    axis.plot(shots, end, marker="D", linewidth=1.2, color="tab:green", label="path end angle")
    axis.set_title("Logged launch angle vs extracted path angle traces")
    axis.set_xlabel("shot number")
    axis.set_ylabel("angle (deg)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _round_or_blank(value: float | None, digits: int) -> float | str:
    """Round floats for CSV output while preserving blank fields."""
    if value is None:
        return ""
    return round(value, digits)


def write_csv(results: list[ShotReview], output_path: Path) -> None:
    """Write reviewed shot metrics to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "shot_number",
                "club_label",
                "quality",
                "selected_club_event",
                "selection_score",
                "track_score",
                "lingering_hits",
                "point_count",
                "start_ms",
                "duration_ms",
                "start_distance_m",
                "end_distance_m",
                "distance_gain_m",
                "monotonicity",
                "angle_span_deg",
                "start_angle_deg",
                "mean_angle_deg",
                "end_angle_deg",
                "logged_launch_angle_deg",
                "logged_launch_confidence",
                "logged_ball_angle_frames",
                "logged_ball_angle_accepted",
                "expected_launch_deg",
                "allowed_delta_deg",
                "observed_delta_deg",
                "logged_ball_speed_mph",
                "logged_club_speed_mph",
                "rolling_ball_dt_ms",
            ],
        )
        writer.writeheader()
        for result in results:
            metrics = result.anchor.metrics
            writer.writerow(
                {
                    "shot_number": result.shot_number,
                    "club_label": result.club_label,
                    "quality": result.quality,
                    "selected_club_event": result.anchor.club_event_index,
                    "selection_score": round(result.anchor.selection_score, 2),
                    "track_score": round(result.anchor.track_score, 2),
                    "lingering_hits": result.anchor.lingering_hits,
                    "point_count": metrics.point_count,
                    "start_ms": round(metrics.start_ms, 1),
                    "duration_ms": round(metrics.duration_ms, 1),
                    "start_distance_m": round(metrics.start_distance_m, 2),
                    "end_distance_m": round(metrics.end_distance_m, 2),
                    "distance_gain_m": round(metrics.distance_gain_m, 2),
                    "monotonicity": round(metrics.monotonicity, 2),
                    "angle_span_deg": round(metrics.angle_span_deg, 2),
                    "start_angle_deg": round(metrics.start_angle_deg, 2),
                    "mean_angle_deg": round(metrics.mean_angle_deg, 2),
                    "end_angle_deg": round(metrics.end_angle_deg, 2),
                    "logged_launch_angle_deg": _round_or_blank(result.logged_launch_angle_deg, 2),
                    "logged_launch_confidence": _round_or_blank(result.logged_launch_confidence, 2),
                    "logged_ball_angle_frames": result.logged_ball_angle_frames or "",
                    "logged_ball_angle_accepted": (
                        result.logged_ball_angle_accepted
                        if result.logged_ball_angle_accepted is not None
                        else ""
                    ),
                    "expected_launch_deg": _round_or_blank(result.expected_launch_deg, 2),
                    "allowed_delta_deg": _round_or_blank(result.allowed_delta_deg, 2),
                    "observed_delta_deg": _round_or_blank(result.observed_delta_deg, 2),
                    "logged_ball_speed_mph": _round_or_blank(result.logged_ball_speed_mph, 2),
                    "logged_club_speed_mph": _round_or_blank(result.logged_club_speed_mph, 2),
                    "rolling_ball_dt_ms": _round_or_blank(result.rolling_ball_dt_ms, 1),
                }
            )


def write_summary(
    session_meta: dict,
    session_path: Path,
    results: list[ShotReview],
    output_path: Path,
) -> None:
    """Write a short markdown summary for the reviewed session."""
    quality_counts: dict[str, int] = defaultdict(int)
    by_club: dict[str, list[ShotReview]] = defaultdict(list)
    for result in results:
        quality_counts[result.quality] += 1
        by_club[result.club_label].append(result)

    logged_angles = [
        result.logged_launch_angle_deg
        for result in results
        if result.logged_launch_angle_deg is not None
    ]
    strong_results = [result for result in results if result.quality == "strong"]
    partial_results = [result for result in results if result.quality == "partial"]
    weak_results = [result for result in results if result.quality == "weak"]

    lines = [
        f"# Session Review: {session_path.name}",
        "",
        "## Method",
        "",
        "1. Parse `rolling_buffer_capture`, `kld7_buffer`, and `shot_detected` rows by `shot_number`.",
        "2. Re-detect close/high-magnitude club anchors from each `kld7_buffer`.",
        "3. For each anchor, score outward post-impact `pdat` paths by timing, distance growth, angle continuity, magnitude, and lingering-clutter penalties.",
        "4. Keep the best anchor/path per shot and export plots plus a CSV summary.",
        "",
        "## Session Summary",
        "",
        f"- session mode: `{session_meta.get('mode', 'unknown')}`",
        f"- total analyzed shots: `{len(results)}`",
        f"- strong profiles: `{quality_counts['strong']}`",
        f"- partial profiles: `{quality_counts['partial']}`",
        f"- weak profiles: `{quality_counts['weak']}`",
    ]
    if logged_angles:
        lines.append(
            f"- logged launch-angle range: `{min(logged_angles):.1f}°` to `{max(logged_angles):.1f}°`"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Strong profiles: `{', '.join(f'{result.shot_number:02d}' for result in strong_results) or 'none'}`",
            f"- Partial profiles: `{', '.join(f'{result.shot_number:02d}' for result in partial_results) or 'none'}`",
            f"- Weak profiles: `{', '.join(f'{result.shot_number:02d}' for result in weak_results) or 'none'}`",
        ]
    )

    for club_label in sorted(by_club):
        club_results = by_club[club_label]
        gains = [result.anchor.metrics.distance_gain_m for result in club_results]
        durations = [result.anchor.metrics.duration_ms for result in club_results]
        lines.append(
            f"- `{club_label}` shots: `{len(club_results)}` total, "
            f"median distance gain `{np.median(gains):.2f} m`, "
            f"median tracked duration `{np.median(durations):.0f} ms`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The `strong`, `partial`, and `weak` labels are review grades for profile recoverability and coherence, not grades of swing quality or ball quality.",
            "The strongest profiles show the expected outward progression over roughly `50-225 ms` with about `1-2.5 m` of range gain.",
            "The weaker profiles either stay in a far clutter band or expose only a short segment of the shot.",
            "Use this report as an empirical review aid for K-LD7 tuning, not as ground-truth validation of launch angle by itself.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Review a K-LD7 session JSONL file and export shot-profile plots.",
    )
    parser.add_argument(
        "session_file",
        type=Path,
        help="Path to a session JSONL file (for example session_logs/session_20260403_133805_range.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated review files (default: shots/session_review_<session>).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing generated files in the output directory before writing new ones.",
    )
    args = parser.parse_args()

    try:
        output_dir = args.output_dir or session_output_dir(args.session_file)
        session_meta, results = analyze_session(args.session_file)
        ensure_output_dir(output_dir, clean=args.clean)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if not results:
        raise SystemExit(f"No reviewable K-LD7 shots found in {args.session_file}.")

    for warning in session_meta.get("_review_warnings", []):
        print(f"Warning: {warning}")

    for result in results:
        plot_shot_profile(
            result,
            args.session_file.stem,
            output_dir / f"shot_{result.shot_number:02d}_profile.png",
        )

    plot_overlay(
        results,
        args.session_file.stem,
        output_dir / "all_shot_profiles_overlay.png",
    )
    plot_launch_angle_review(results, output_dir / "launch_angle_review.png")
    write_csv(results, output_dir / "shot_profiles.csv")
    write_summary(session_meta, args.session_file, results, output_dir / "summary.md")

    print(f"Analyzed {len(results)} shots from {args.session_file}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
