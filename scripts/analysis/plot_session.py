#!/usr/bin/env python3
"""Generate session analysis plots from a JSONL session log.

Usage:
    python scripts/plot_session.py session_logs/session_20260409_104308_range.jsonl
    python scripts/plot_session.py session_logs/session_*.jsonl --output-dir ~/plots
    python scripts/plot_session.py session_logs/session_20260409_104308_range.jsonl --shot 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Color scheme
C_BALL = "#2196F3"
C_CLUB = "#FF9800"
C_ANGLE = "#4CAF50"
C_HORIZ = "#9C27B0"
C_SPIN = "#F44336"
C_CARRY = "#009688"


def load_session(path: Path) -> list[dict]:
    shots = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "shot_detected":
                shots.append(entry)
    return shots


def plot_radar_card(shot: dict, output: Path):
    """Single-shot radar card with all available metrics."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(f"Shot {shot['shot_number']} — Radar Card", fontsize=18, fontweight="bold", pad=20)

    metrics = [
        ("Ball Speed", f"{shot['ball_speed_mph']:.1f} mph", C_BALL),
    ]
    if shot.get("club_speed_mph") is not None:
        metrics.append(("Club Speed", f"{shot['club_speed_mph']:.1f} mph", C_CLUB))
    if shot.get("smash_factor") is not None:
        metrics.append(("Smash Factor", f"{shot['smash_factor']:.2f}", C_BALL))
    if shot.get("launch_angle_vertical") is not None:
        metrics.append(("V. Launch", f"{shot['launch_angle_vertical']:.1f}°", C_ANGLE))
    if shot.get("launch_angle_horizontal") is not None:
        metrics.append(("H. Launch", f"{shot['launch_angle_horizontal']:+.1f}°", C_HORIZ))
    if shot.get("club_angle_deg") is not None:
        metrics.append(("Club AoA", f"{shot['club_angle_deg']:+.1f}°", C_CLUB))
    if shot.get("club_path_deg") is not None:
        metrics.append(("Club Path", f"{shot['club_path_deg']:+.1f}°", C_HORIZ))
    if shot.get("spin_axis_deg") is not None:
        sa = shot["spin_axis_deg"]
        shape = "fade" if sa > 2 else "draw" if sa < -2 else "straight"
        metrics.append(("Spin Axis", f"{sa:+.1f}° ({shape})", C_SPIN))
    elif shot.get("launch_angle_horizontal") is not None and shot.get("club_path_deg") is not None:
        sa = shot["launch_angle_horizontal"] - shot["club_path_deg"]
        shape = "fade" if sa > 2 else "draw" if sa < -2 else "straight"
        metrics.append(("Spin Axis", f"{sa:+.1f}° ({shape})", C_SPIN))
    if shot.get("spin_rpm") is not None:
        metrics.append(("Spin Rate", f"{shot['spin_rpm']:.0f} rpm", C_SPIN))
    if shot.get("estimated_carry_yards") is not None:
        metrics.append(("Carry", f"{shot['estimated_carry_yards']:.0f} yds", C_CARRY))
    if shot.get("carry_spin_adjusted") is not None:
        metrics.append(("Carry (adj)", f"{shot['carry_spin_adjusted']:.0f} yds", C_CARRY))
    if shot.get("angle_source"):
        metrics.append(("Angle Source", shot["angle_source"], "#666666"))

    for i, (label, value, color) in enumerate(metrics):
        row = 9.2 - i * (8.0 / max(len(metrics), 1))
        ax.text(1.5, row, label, fontsize=14, ha="right", va="center", color="#555")
        ax.text(2.0, row, value, fontsize=16, ha="left", va="center", fontweight="bold", color=color)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speeds(shots: list[dict], output: Path):
    """Ball and club speed bar chart with smash factor."""
    fig, ax = plt.subplots(figsize=(10, 5))
    shot_nums = [s["shot_number"] for s in shots]
    ball = [s["ball_speed_mph"] for s in shots]
    club = [s.get("club_speed_mph", 0) or 0 for s in shots]
    smash = [s.get("smash_factor") for s in shots]

    x = np.arange(len(shots))
    w = 0.35
    ax.bar(x - w / 2, ball, w, label="Ball Speed", color=C_BALL, alpha=0.85)
    ax.bar(x + w / 2, club, w, label="Club Speed", color=C_CLUB, alpha=0.85)
    for i, sf in enumerate(smash):
        if sf:
            ax.text(i, ball[i] + 2, f"{sf:.2f}x", ha="center", fontsize=9, color="#333")

    ax.set_xlabel("Shot")
    ax.set_ylabel("Speed (mph)")
    ax.set_title("Ball & Club Speed (smash factor annotated)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{n}" for n in shot_nums])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_launch_angles(shots: list[dict], output: Path):
    """Vertical and horizontal launch angles side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    shot_nums = [s["shot_number"] for s in shots]

    v = [s.get("launch_angle_vertical", 0) or 0 for s in shots]
    ax1.bar(shot_nums, v, color=C_ANGLE, alpha=0.85)
    for i, val in enumerate(v):
        if val:
            ax1.text(shot_nums[i], val + 0.3, f"{val:.1f}°", ha="center", fontsize=10)
    ax1.set_xlabel("Shot")
    ax1.set_ylabel("Degrees")
    ax1.set_title("V. Launch Angle")
    ax1.grid(axis="y", alpha=0.3)

    h = [s.get("launch_angle_horizontal") for s in shots]
    colors = [C_HORIZ if val is not None else "#ccc" for val in h]
    ax2.bar(shot_nums, [val if val is not None else 0 for val in h], color=colors, alpha=0.85)
    ax2.axhline(y=0, color="#333", linewidth=0.8, linestyle="--")
    for i, val in enumerate(h):
        if val is not None:
            ax2.text(shot_nums[i], val + (0.5 if val >= 0 else -1.5), f"{val:+.1f}°", ha="center", fontsize=10)
        else:
            ax2.text(shot_nums[i], 0.3, "—", ha="center", fontsize=10, color="#999")
    ax2.set_xlabel("Shot")
    ax2.set_ylabel("Degrees (+ right, - left)")
    ax2.set_title("H. Launch Angle")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_club_metrics(shots: list[dict], output: Path):
    """Club AoA and club path side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    shot_nums = [s["shot_number"] for s in shots]

    aoa = [s.get("club_angle_deg") for s in shots]
    ax1.bar(shot_nums, [a if a is not None else 0 for a in aoa],
            color=[C_CLUB if a is not None else "#ccc" for a in aoa], alpha=0.85)
    ax1.axhline(y=0, color="#333", linewidth=0.8, linestyle="--")
    for i, a in enumerate(aoa):
        if a is not None:
            ax1.text(shot_nums[i], a + (0.3 if a >= 0 else -0.8), f"{a:+.1f}°", ha="center", fontsize=10)
        else:
            ax1.text(shot_nums[i], 0.3, "—", ha="center", fontsize=10, color="#999")
    ax1.set_xlabel("Shot")
    ax1.set_ylabel("Degrees (- = descending)")
    ax1.set_title("Club Angle of Attack")
    ax1.grid(axis="y", alpha=0.3)

    path = [s.get("club_path_deg") for s in shots]
    ax2.bar(shot_nums, [p if p is not None else 0 for p in path],
            color=[C_HORIZ if p is not None else "#ccc" for p in path], alpha=0.85)
    ax2.axhline(y=0, color="#333", linewidth=0.8, linestyle="--")
    for i, p in enumerate(path):
        if p is not None:
            ax2.text(shot_nums[i], p + (0.3 if p >= 0 else -1.0), f"{p:+.1f}°", ha="center", fontsize=10)
        else:
            ax2.text(shot_nums[i], 0.3, "—", ha="center", fontsize=10, color="#999")
    ax2.set_xlabel("Shot")
    ax2.set_ylabel("Degrees (+ right, - left)")
    ax2.set_title("Club Path")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_overhead(shot: dict, output: Path):
    """Overhead view showing ball direction, club path, and spin axis."""
    h_launch = shot.get("launch_angle_horizontal")
    c_path = shot.get("club_path_deg")
    if h_launch is None:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-5, 50)
    ax.set_aspect("equal")
    ax.set_title(f"Shot {shot['shot_number']} — Overhead View", fontsize=16, fontweight="bold")

    ax.plot([0, 0], [0, 45], "k--", alpha=0.3, linewidth=1)
    ax.text(0.5, 44, "Target", fontsize=10, color="#999")
    ax.plot(0, 0, "ko", markersize=8)
    ax.text(1, -1.5, "Ball", fontsize=10)

    ball_dx = 40 * np.sin(np.radians(h_launch))
    ball_dy = 40 * np.cos(np.radians(h_launch))
    ax.annotate("", xy=(ball_dx, ball_dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_BALL, lw=2.5))
    ax.text(ball_dx + 1, ball_dy - 2, f"Ball: {h_launch:+.1f}°", fontsize=11, color=C_BALL, fontweight="bold")

    if c_path is not None:
        club_dx = 35 * np.sin(np.radians(c_path))
        club_dy = 35 * np.cos(np.radians(c_path))
        ax.annotate("", xy=(club_dx, club_dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=C_CLUB, lw=2.5, linestyle="--"))
        ax.text(club_dx - 5, club_dy - 2, f"Club: {c_path:+.1f}°", fontsize=11, color=C_CLUB, fontweight="bold")

        spin_axis = h_launch - c_path
        shape = "fade" if spin_axis > 2 else "draw" if spin_axis < -2 else "straight"
        ax.text(0, -3.5, f"Spin Axis: {spin_axis:+.1f}° ({shape})", fontsize=12,
                ha="center", fontweight="bold", color=C_SPIN,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_SPIN, alpha=0.9))

    ax.set_xlabel("Yards left/right")
    ax.set_ylabel("Yards downrange")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_carry(shots: list[dict], output: Path):
    """Carry distance comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    shot_nums = [s["shot_number"] for s in shots]
    carry = [s.get("estimated_carry_yards", 0) for s in shots]
    carry_adj = [s.get("carry_spin_adjusted", 0) for s in shots]

    x = np.arange(len(shots))
    w = 0.35
    ax.bar(x - w / 2, carry, w, label="Estimated Carry", color=C_CARRY, alpha=0.7)
    ax.bar(x + w / 2, carry_adj, w, label="Spin-Adjusted", color=C_CARRY, alpha=1.0)
    for i in range(len(shots)):
        ax.text(i, max(carry[i], carry_adj[i]) + 2, f"{carry_adj[i]:.0f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Shot")
    ax.set_ylabel("Carry (yards)")
    ax.set_title("Carry Distance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{n}" for n in shot_nums])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_completeness(shots: list[dict], output: Path):
    """Metric completeness heatmap."""
    metric_names = ["Ball\nSpeed", "Club\nSpeed", "Smash", "V.\nLaunch", "H.\nLaunch",
                    "Club\nAoA", "Club\nPath", "Spin\nAxis", "Spin", "Carry"]
    metric_keys = ["ball_speed_mph", "club_speed_mph", "smash_factor", "launch_angle_vertical",
                   "launch_angle_horizontal", "club_angle_deg", "club_path_deg", None, "spin_rpm",
                   "estimated_carry_yards"]

    grid = np.zeros((len(shots), len(metric_keys)))
    for i, s in enumerate(shots):
        for j, k in enumerate(metric_keys):
            if k is None:
                has = (s.get("spin_axis_deg") is not None or
                       (s.get("launch_angle_horizontal") is not None and s.get("club_path_deg") is not None))
                grid[i, j] = 1 if has else 0
            else:
                grid[i, j] = 1 if s.get(k) is not None else 0

    fig, ax = plt.subplots(figsize=(11, max(3, len(shots) * 0.8 + 1)))
    cmap = ListedColormap(["#ffcdd2", "#c8e6c9"])
    ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest")

    for i, s in enumerate(shots):
        for j, k in enumerate(metric_keys):
            if k is None:
                sa = s.get("spin_axis_deg")
                if sa is None:
                    h = s.get("launch_angle_horizontal")
                    p = s.get("club_path_deg")
                    sa = h - p if h is not None and p is not None else None
                if sa is not None:
                    ax.text(j, i, f"{sa:+.1f}", ha="center", va="center", fontsize=7, color="#333")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999")
            else:
                v = s.get(k)
                if v is not None:
                    txt = f"{v:.1f}" if isinstance(v, float) and abs(v) < 200 else str(v)[:6]
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="#333")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999")

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticks(range(len(shots)))
    ax.set_yticklabels([f"Shot {s['shot_number']}" for s in shots])
    ax.set_title("Metric Completeness (green = captured)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate session analysis plots.")
    parser.add_argument("session", type=Path, help="Path to session JSONL file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: alongside session file)")
    parser.add_argument("--shot", type=int, default=None, help="Generate radar card + overhead for a specific shot number")
    args = parser.parse_args()

    if not args.session.exists():
        print(f"Error: {args.session} not found")
        sys.exit(1)

    shots = load_session(args.session)
    if not shots:
        print(f"No shots found in {args.session}")
        sys.exit(1)

    output_dir = args.output_dir or args.session.parent / f"plots_{args.session.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.shot:
        shot = next((s for s in shots if s["shot_number"] == args.shot), None)
        if not shot:
            print(f"Shot {args.shot} not found (available: {[s['shot_number'] for s in shots]})")
            sys.exit(1)
        plot_radar_card(shot, output_dir / f"shot{args.shot}_radar_card.png")
        plot_overhead(shot, output_dir / f"shot{args.shot}_overhead.png")
        print(f"Shot {args.shot} plots saved to {output_dir}/")
    else:
        # All plots
        plot_speeds(shots, output_dir / "speeds.png")
        plot_launch_angles(shots, output_dir / "launch_angles.png")
        plot_club_metrics(shots, output_dir / "club_metrics.png")
        plot_carry(shots, output_dir / "carry.png")
        plot_completeness(shots, output_dir / "completeness.png")

        # Radar card + overhead for the most complete shot
        best = max(shots, key=lambda s: sum(1 for k in [
            "ball_speed_mph", "club_speed_mph", "launch_angle_vertical",
            "launch_angle_horizontal", "club_angle_deg", "club_path_deg", "spin_rpm",
        ] if s.get(k) is not None))
        best_n = best["shot_number"]
        plot_radar_card(best, output_dir / f"shot{best_n}_radar_card.png")
        plot_overhead(best, output_dir / f"shot{best_n}_overhead.png")

        # Overhead for any shot with horizontal data
        for s in shots:
            if s.get("launch_angle_horizontal") is not None and s["shot_number"] != best_n:
                plot_overhead(s, output_dir / f"shot{s['shot_number']}_overhead.png")

        print(f"{len(list(output_dir.glob('*.png')))} plots saved to {output_dir}/")

    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
