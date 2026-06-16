"""Generate scatter + Bland-Altman plots from a Trackman ↔ OpenFlight
comparison CSV (output of ``compare_trackman.py``).

Two plots per metric:

1. **Scatter** — Trackman on x-axis, OpenFlight on y-axis, with the
   y=x identity line. Shows whether OpenFlight systematically over- or
   under-reads at each value of the true (Trackman) measurement.
2. **Bland-Altman** — Mean of (OF, TM) on x-axis, delta (OF − TM) on
   y-axis, with horizontal reference lines for mean delta and ±1.96·σ
   limits of agreement. Separates bias from spread, and reveals
   value-dependent bias (e.g. larger errors at higher speeds).

Both plots are color-coded by club so club-dependent bias is visible.

Usage::

    uv run python scripts/analysis/plot_comparison.py \\
        session_logs/comparison_20260506.csv \\
        --output-dir session_logs/comparison_20260506_plots/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position


# Each tuple: (csv-suffix, friendly label, unit). The csv-suffix
# corresponds to the column-stem in the comparison CSV — appending
# "_of"/"_tm"/"_delta" gets the actual columns.
_METRICS: List[Tuple[str, str, str]] = [
    ("ball_speed", "Ball speed",       "mph"),
    ("club_speed", "Club speed",       "mph"),
    ("launch_v",   "Launch angle V",   "deg"),
    ("launch_h",   "Launch direction", "deg"),
    ("spin",       "Spin rate",        "rpm"),
    ("carry",      "Carry",            "yds"),
]


_CLUB_COLORS = {
    "driver":   "#FF9800",
    "3-wood":   "#8BC34A",
    "5-wood":   "#4CAF50",
    "3-iron":   "#3F51B5",
    "5-iron":   "#2196F3",
    "6-iron":   "#03A9F4",
    "7-iron":   "#00BCD4",
    "8-iron":   "#009688",
    "9-iron":   "#26A69A",
    "pw":       "#9C27B0",
    "gw":       "#AB47BC",
    "sw":       "#7B1FA2",
    "lw":       "#4A148C",
}
_DEFAULT_COLOR = "#777777"


def _color_for_club(club: str) -> str:
    return _CLUB_COLORS.get(club, _DEFAULT_COLOR)


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_comparison(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("match_quality") != "good":
                continue  # only chart confidently-paired shots
            rows.append(r)
    return rows


def _gather_metric(rows: List[Dict], stem: str
                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return aligned (of, tm, club) arrays for rows where both values
    are present for this metric."""
    of_vals, tm_vals, clubs = [], [], []
    for r in rows:
        of = _to_float(r.get(f"{stem}_of"))
        tm = _to_float(r.get(f"{stem}_tm"))
        if of is None or tm is None:
            continue
        of_vals.append(of)
        tm_vals.append(tm)
        clubs.append(r.get("club") or "(no club)")
    return np.array(of_vals), np.array(tm_vals), clubs


def _scatter(ax, of: np.ndarray, tm: np.ndarray, clubs: List[str],
             label: str, unit: str) -> None:
    """OpenFlight vs Trackman scatter with y=x reference line."""
    if len(of) == 0:
        ax.text(0.5, 0.5, "no paired data",
                ha="center", va="center", transform=ax.transAxes,
                color="#999")
        ax.set_title(f"{label} — agreement")
        return

    # One scatter call per club so the legend entries are clean.
    by_club: Dict[str, Tuple[List[float], List[float]]] = {}
    for x, y, c in zip(tm, of, clubs):
        by_club.setdefault(c, ([], []))[0].append(x)
        by_club[c][1].append(y)
    for club, (xs, ys) in sorted(by_club.items()):
        ax.scatter(xs, ys, label=f"{club} (n={len(xs)})",
                   color=_color_for_club(club),
                   alpha=0.85, s=55, edgecolor="white", linewidth=0.7)

    lo = float(min(of.min(), tm.min()))
    hi = float(max(of.max(), tm.max()))
    pad = (hi - lo) * 0.05 + 1e-9
    line = np.array([lo - pad, hi + pad])
    ax.plot(line, line, "--", color="#888", linewidth=1, label="y = x")
    ax.set_xlim(line[0], line[1])
    ax.set_ylim(line[0], line[1])
    ax.set_xlabel(f"Trackman {label} ({unit})")
    ax.set_ylabel(f"OpenFlight {label} ({unit})")
    ax.set_title(f"{label} — agreement (n={len(of)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)


def _bland_altman(ax, of: np.ndarray, tm: np.ndarray, clubs: List[str],
                  label: str, unit: str) -> None:
    """Bland-Altman plot: mean on x, delta on y, with ±1.96σ limits."""
    if len(of) == 0:
        ax.text(0.5, 0.5, "no paired data",
                ha="center", va="center", transform=ax.transAxes,
                color="#999")
        ax.set_title(f"{label} — Bland-Altman")
        return

    means = (of + tm) / 2.0
    deltas = of - tm

    by_club: Dict[str, Tuple[List[float], List[float]]] = {}
    for x, y, c in zip(means, deltas, clubs):
        by_club.setdefault(c, ([], []))[0].append(x)
        by_club[c][1].append(y)
    for club, (xs, ys) in sorted(by_club.items()):
        ax.scatter(xs, ys, label=f"{club} (n={len(xs)})",
                   color=_color_for_club(club),
                   alpha=0.85, s=55, edgecolor="white", linewidth=0.7)

    bias = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=0))
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    ax.axhline(0, color="#888", linewidth=0.7)
    ax.axhline(bias, color="#d32f2f", linestyle="-", linewidth=1.2,
               label=f"bias = {bias:+.2f} {unit}")
    ax.axhline(upper, color="#d32f2f", linestyle="--", linewidth=0.8,
               label=f"+1.96σ = {upper:+.2f}")
    ax.axhline(lower, color="#d32f2f", linestyle="--", linewidth=0.8,
               label=f"-1.96σ = {lower:+.2f}")

    ax.set_xlabel(f"mean of OF and TM ({unit})")
    ax.set_ylabel(f"OF − TM ({unit})")
    ax.set_title(f"{label} — Bland-Altman (bias={bias:+.2f}, σ={sd:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)


def plot_metric(rows: List[Dict], stem: str, label: str, unit: str,
                output_dir: Path) -> Optional[Path]:
    of, tm, clubs = _gather_metric(rows, stem)
    if len(of) == 0:
        print(f"  skipping {label}: no paired data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], of, tm, clubs, label, unit)
    _bland_altman(axes[1], of, tm, clubs, label, unit)
    fig.suptitle(f"{label}: OpenFlight vs Trackman", fontsize=14, y=1.00)
    fig.tight_layout()

    output = output_dir / f"compare_{stem}.png"
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output}  (n={len(of)})")
    return output


# ---------------------------------------------------------------------------
# Per-metric takeaway classification
# ---------------------------------------------------------------------------

# Threshold table: (metric_stem, "good" max-stddev, "good" max-|bias|,
#                   "noisy" max-stddev, unit, severity-suffix-text).
# Rules:
#   * If |bias| <= good_bias and stddev <= good_sd → good
#   * Else if stddev > noisy_sd                     → noisy (per-shot noise dominates)
#   * Else                                          → biased (clean offset, easy to fix)
_TAKEAWAY_RULES: Dict[str, Tuple[float, float, float]] = {
    # stem:        (good_sd, good_bias, noisy_sd)
    "ball_speed":  (1.5,  1.5,   3.0),
    "club_speed":  (3.0,  3.0,   6.0),
    "launch_v":    (3.0,  3.0,   5.0),
    "launch_h":    (3.0,  3.0,   5.0),
    "spin":        (500.0, 500.0, 1500.0),
    "carry":       (10.0, 10.0,  20.0),
}

# Severity → (background, accent, label).
_SEVERITY_STYLE = {
    "good":   ("#E8F5E9", "#2E7D32", "ON TARGET"),
    "biased": ("#FFF3E0", "#E65100", "SYSTEMATIC BIAS"),
    "noisy":  ("#FFEBEE", "#C62828", "HIGH NOISE"),
    "thin":   ("#F5F5F5", "#616161", "LOW DATA"),
}


def _classify_metric(stem: str, deltas: List[float],
                     total_pairs: int) -> Tuple[str, str]:
    """Classify the metric into a severity bucket and produce a short
    plain-English verdict line.

    ``total_pairs`` is the number of good comparable pairs in the whole
    session — used to call out metrics that are missing from most
    shots (e.g. spin where the OpenFlight side often has no value).
    """
    n = len(deltas)
    if n < 3:
        return "thin", f"only {n} comparable shot{'s' if n != 1 else ''}"
    # If we have data on fewer than half of all session pairs, the
    # metric is thinly covered no matter what the stats say — readings
    # that exist might happen to be good but we can't generalize.
    if total_pairs > 0 and n < 0.5 * total_pairs:
        coverage = 100 * n / total_pairs
        return "thin", (f"only {n}/{total_pairs} shots had a value "
                        f"({coverage:.0f}% coverage — most shots had no reading)")
    bias = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=0))
    good_sd, good_bias, noisy_sd = _TAKEAWAY_RULES.get(
        stem, (1e9, 1e9, 1e9))

    if abs(bias) <= good_bias and sd <= good_sd:
        return "good", f"matches Trackman within ±{sd:.1f} (no fix needed)"
    if sd > noisy_sd:
        # Noise dominates — bias is in the noise.
        if abs(bias) > 2 * sd:
            return "noisy", (f"bias {bias:+.1f} but per-shot σ is {sd:.1f} "
                             f"— mostly noise")
        return "noisy", (f"per-shot σ is {sd:.1f} — readings vary a lot, "
                         f"can't trust individual shots")
    # Else: biased — bias dominates the noise.
    direction = "low" if bias < 0 else "high"
    return "biased", (f"reads {abs(bias):.1f} {direction} on average "
                      f"(σ={sd:.1f} — clean miscalibration)")


def _per_club_summary(rows: List[Dict], stem: str
                      ) -> Tuple[List[str], List[float], List[float],
                                 List[int]]:
    """Per-club (sorted): names, mean deltas, stddevs, sample counts."""
    of, tm, clubs = _gather_metric(rows, stem)
    if len(of) == 0:
        return [], [], [], []
    deltas_by_club: Dict[str, List[float]] = {}
    for v_of, v_tm, c in zip(of, tm, clubs):
        deltas_by_club.setdefault(c, []).append(v_of - v_tm)
    names = sorted(deltas_by_club)
    return (names,
            [float(np.mean(deltas_by_club[c])) for c in names],
            [float(np.std(deltas_by_club[c], ddof=0))
             if len(deltas_by_club[c]) > 1 else 0.0 for c in names],
            [len(deltas_by_club[c]) for c in names])


def plot_takeaways(rows: List[Dict], output_dir: Path) -> Path:
    """One-page summary card laying out a plain-English verdict for
    each metric. Built for a quick read of "where is OpenFlight
    actually working / not working." No dense scatter plots; just the
    headline finding plus a small per-club bar.
    """
    n_metrics = len(_METRICS)
    fig = plt.figure(figsize=(13, 11))

    # Reserve a top strip for the title + overall match counts.
    gs = fig.add_gridspec(
        n_metrics + 1, 2,
        height_ratios=[0.55] + [1.0] * n_metrics,
        width_ratios=[2.4, 1.0],
        hspace=0.45, wspace=0.25,
        left=0.04, right=0.98, top=0.96, bottom=0.04,
    )

    # --- header strip --------------------------------------------------
    header = fig.add_subplot(gs[0, :])
    header.axis("off")
    header.text(0.0, 0.65, "OpenFlight vs Trackman — Key Takeaways",
                fontsize=22, fontweight="bold", color="#222")
    n_pairs = len(rows)
    clubs_list = sorted({r.get("club", "") for r in rows if r.get("club")})
    header.text(0.0, 0.18,
                f"{n_pairs} good shot pairs across "
                f"{len(clubs_list)} clubs ({', '.join(clubs_list)}).  "
                f"Severity ranking: bias size & per-shot variance "
                f"vs. metric-specific thresholds.",
                fontsize=11, color="#555")

    # --- one row per metric --------------------------------------------
    for idx, (stem, label, unit) in enumerate(_METRICS):
        verdict_ax = fig.add_subplot(gs[idx + 1, 0])
        bar_ax = fig.add_subplot(gs[idx + 1, 1])

        of, tm, _clubs = _gather_metric(rows, stem)
        deltas = [a - b for a, b in zip(of, tm)]
        severity, sentence = _classify_metric(stem, deltas, len(rows))
        bg, accent, badge = _SEVERITY_STYLE[severity]

        # Verdict card.
        verdict_ax.axis("off")
        verdict_ax.add_patch(plt.Rectangle(
            (0.0, 0.0), 1.0, 1.0, transform=verdict_ax.transAxes,
            facecolor=bg, edgecolor=accent, linewidth=1.6, zorder=0))
        verdict_ax.text(
            0.025, 0.78, label.upper(), fontsize=15,
            fontweight="bold", color="#222", transform=verdict_ax.transAxes,
        )
        verdict_ax.text(
            0.97, 0.78, badge, fontsize=10, fontweight="bold",
            color="white", ha="right",
            transform=verdict_ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=accent,
                      edgecolor="none"),
        )

        # Headline numbers row.
        if len(deltas) >= 1:
            bias = float(np.mean(deltas))
            sd = float(np.std(deltas, ddof=0)) if len(deltas) > 1 else 0.0
            num_line = (f"mean Δ (OF−TM) = {bias:+.2f} {unit}    "
                        f"σ = {sd:.2f} {unit}    n = {len(deltas)}")
        else:
            num_line = "no comparable shots"
        verdict_ax.text(
            0.025, 0.50, num_line, fontsize=11,
            color="#333", family="monospace",
            transform=verdict_ax.transAxes,
        )

        # Plain-English sentence.
        verdict_ax.text(
            0.025, 0.20, sentence, fontsize=12, fontstyle="italic",
            color=accent, transform=verdict_ax.transAxes,
        )

        # Per-club mini bar chart on the right.
        names, means, sds, ns = _per_club_summary(rows, stem)
        if not names:
            bar_ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        color="#999", transform=bar_ax.transAxes)
            bar_ax.axis("off")
            continue
        x = np.arange(len(names))
        bar_ax.bar(x, means, yerr=sds, color=[_color_for_club(c) for c in names],
                   edgecolor="white", linewidth=0.6, capsize=4)
        bar_ax.axhline(0, color="#444", linewidth=0.6)
        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels([f"{n}\nn={ni}" for n, ni in zip(names, ns)],
                               fontsize=8)
        bar_ax.set_ylabel(f"Δ ({unit})", fontsize=9)
        bar_ax.tick_params(axis="y", labelsize=8)
        bar_ax.grid(True, alpha=0.25, axis="y")
        for spine in ("top", "right"):
            bar_ax.spines[spine].set_visible(False)

    output = output_dir / "compare_takeaways.png"
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output}")
    return output


def plot_overview(rows: List[Dict], output_dir: Path) -> Path:
    """A single 3x2 grid showing the bias bar per metric per club —
    quick at-a-glance summary."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    for idx, (stem, label, unit) in enumerate(_METRICS):
        ax = axes[idx]
        of, tm, clubs = _gather_metric(rows, stem)
        if len(of) == 0:
            ax.text(0.5, 0.5, f"{label}: no data",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#999")
            ax.axis("off")
            continue

        # Group deltas by club.
        deltas_by_club: Dict[str, List[float]] = {}
        for v_of, v_tm, c in zip(of, tm, clubs):
            deltas_by_club.setdefault(c, []).append(v_of - v_tm)

        names = sorted(deltas_by_club)
        means = [float(np.mean(deltas_by_club[c])) for c in names]
        stds = [float(np.std(deltas_by_club[c], ddof=0))
                if len(deltas_by_club[c]) > 1 else 0.0
                for c in names]
        ns = [len(deltas_by_club[c]) for c in names]
        colors = [_color_for_club(c) for c in names]

        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, color=colors,
               edgecolor="white", linewidth=0.8, capsize=5)
        ax.axhline(0, color="#444", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(n={ni})" for n, ni in zip(names, ns)],
                           fontsize=9)
        ax.set_ylabel(f"OF − TM ({unit})")
        ax.set_title(f"{label}: bias ± 1σ per club")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("OpenFlight vs Trackman — bias overview by club",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    output = output_dir / "compare_overview.png"
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output}")
    return output


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate scatter + Bland-Altman plots from a "
                    "compare_trackman comparison CSV.",
    )
    parser.add_argument("comparison_csv", type=Path,
                        help="Comparison CSV (output of compare_trackman.py)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: alongside the CSV)")
    args = parser.parse_args(argv)

    if not args.comparison_csv.exists():
        print(f"Comparison CSV not found: {args.comparison_csv}",
              file=sys.stderr)
        return 2

    output_dir = (args.output_dir
                  or args.comparison_csv.with_suffix("").parent
                  / (args.comparison_csv.stem + "_plots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    rows = load_comparison(args.comparison_csv)
    print(f"Loaded {len(rows)} 'good' pair rows")

    plot_takeaways(rows, output_dir)
    plot_overview(rows, output_dir)
    for stem, label, unit in _METRICS:
        plot_metric(rows, stem, label, unit, output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
