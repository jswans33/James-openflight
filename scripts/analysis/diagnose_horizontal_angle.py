#!/usr/bin/env python3
"""Diagnose horizontal launch angle inconsistency from session logs.

Two complementary modes:

1. JSONL session log mode (default).
   Mines `shot_detected` and `kld7_buffer` events to characterize how the
   horizontal radar's angle output behaves shot-to-shot. This mode cannot
   replay the FFT pipeline because the live tracker strips raw RADC bytes
   from snapshot_buffer (see KLD7Tracker.snapshot_buffer). It uses the live
   diagnostics that ARE logged: per-shot ball_angle (horizontal_deg,
   confidence, magnitude/SNR, num_frames), per-frame timestamps, and
   downstream Shot fields.

2. RADC capture mode (--radc PATH).
   Operates on `.pkl` files produced by scripts/analysis/capture_kld7_radc.py
   which DO contain raw 3072-byte RADC frames. In this mode we re-run the
   real `extract_launch_angle` pipeline per shot, plus a per-frame breakdown
   of the ball-band spectrum, picked peak bin, per-bin angle, and SNR. This
   is the right tool to confirm whether the horizontal radar's peak-bin
   selection is locking onto the ball or onto club/multipath/sidelobes.

Usage:
    # JSONL mode
    python scripts/analysis/diagnose_horizontal_angle.py \\
        session_logs/session_2026042[12]_*_range.jsonl \\
        --output-dir session_logs/h_angle_diag

    # RADC capture mode
    python scripts/analysis/diagnose_horizontal_angle.py \\
        --radc session_logs/kld7_radc_20260406_161627-7i.pkl \\
        --output-dir session_logs/radc_diag
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_FACE = "#9C27B0"
C_PATH = "#FF9800"
C_DIFF = "#F44336"
C_GRID = "#cccccc"


# ---------- data model ----------


@dataclass
class ShotRow:
    session_id: str
    shot_number: int
    ts: str
    ball_speed_mph: Optional[float]
    club_speed_mph: Optional[float]
    angle_source: Optional[str]
    h_angle: Optional[float]
    v_angle: Optional[float]
    confidence: Optional[float]
    club_path_deg: Optional[float]
    spin_axis_deg: Optional[float]
    # From horizontal kld7_buffer (live diagnostics)
    h_ball_angle: Optional[float] = None
    h_confidence: Optional[float] = None
    h_num_frames: Optional[int] = None
    h_avg_snr_db: Optional[float] = None
    h_buffer_frame_count: Optional[int] = None  # total frames in ring buffer
    h_buffer_span_s: Optional[float] = None     # time span across buffer
    # From vertical kld7_buffer
    v_buffer_frame_count: Optional[int] = None
    v_ball_angle: Optional[float] = None
    v_confidence: Optional[float] = None


def load_sessions(paths: list[Path]) -> list[ShotRow]:
    rows: list[ShotRow] = []
    for p in paths:
        sid = p.stem.replace("_range", "")
        shots_by_num: dict[int, ShotRow] = {}
        # Buffer kld7_buffer entries (they appear before shot_detected in the log)
        # keyed by (shot_number, orientation) -> dict of fields
        pending_bufs: dict[tuple[int, str], dict] = {}

        def buf_fields(entry: dict) -> dict:
            frames = entry.get("frames") or []
            fc = len(frames)
            span = None
            if fc >= 2:
                ts_list = [f.get("timestamp") for f in frames
                           if isinstance(f.get("timestamp"), (int, float))]
                if len(ts_list) >= 2:
                    span = max(ts_list) - min(ts_list)
            ba = entry.get("ball_angle") or {}
            return {
                "frame_count": fc,
                "span_s": span,
                "ball_angle": ba,
            }

        def apply_buf(row: ShotRow, orientation: str, fields: dict) -> None:
            ba = fields["ball_angle"]
            if orientation == "horizontal":
                row.h_buffer_frame_count = fields["frame_count"]
                row.h_buffer_span_s = fields["span_s"]
                row.h_ball_angle = ba.get("horizontal_deg")
                row.h_confidence = ba.get("confidence")
                row.h_num_frames = ba.get("num_frames")
                row.h_avg_snr_db = ba.get("magnitude")
            elif orientation == "vertical":
                row.v_buffer_frame_count = fields["frame_count"]
                row.v_ball_angle = ba.get("vertical_deg")
                row.v_confidence = ba.get("confidence")

        with p.open() as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = e.get("type")
                if t == "shot_detected":
                    sn = e.get("shot_number")
                    row = ShotRow(
                        session_id=sid,
                        shot_number=sn,
                        ts=e.get("ts", ""),
                        ball_speed_mph=e.get("ball_speed_mph"),
                        club_speed_mph=e.get("club_speed_mph"),
                        angle_source=e.get("angle_source"),
                        h_angle=e.get("launch_angle_horizontal"),
                        v_angle=e.get("launch_angle_vertical"),
                        confidence=e.get("launch_angle_confidence"),
                        club_path_deg=e.get("club_path_deg"),
                        spin_axis_deg=e.get("spin_axis_deg"),
                    )
                    # Apply any buffered kld7_buffer entries we already saw
                    for orient in ("horizontal", "vertical"):
                        fields = pending_bufs.pop((sn, orient), None)
                        if fields is not None:
                            apply_buf(row, orient, fields)
                    shots_by_num[sn] = row
                elif t == "kld7_buffer":
                    sn = e.get("shot_number")
                    orient = e.get("orientation")
                    fields = buf_fields(e)
                    row = shots_by_num.get(sn)
                    if row is not None:
                        apply_buf(row, orient, fields)
                    else:
                        pending_bufs[(sn, orient)] = fields
        rows.extend(shots_by_num.values())
    return rows


# ---------- aggregate stats ----------


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def fmt_stat(name: str, xs: list[float]) -> str:
    if not xs:
        return f"{name}: no data"
    return (
        f"{name}: n={len(xs):4d}  mean={stats.mean(xs):+7.2f}  "
        f"stdev={stats.pstdev(xs):5.2f}  min={min(xs):+6.1f}  "
        f"p10={percentile(xs, 0.10):+6.1f}  p50={percentile(xs, 0.50):+6.1f}  "
        f"p90={percentile(xs, 0.90):+6.1f}  max={max(xs):+6.1f}"
    )


def print_summary(rows: list[ShotRow]) -> None:
    print("=" * 100)
    print(f"DIAGNOSTIC SUMMARY  —  {len(rows)} shots, "
          f"{len({r.session_id for r in rows})} sessions")
    print("=" * 100)

    h_radar = [r for r in rows if r.angle_source == "radar" and r.h_angle is not None]
    h_est = [r for r in rows if r.angle_source == "estimated" and r.h_angle is not None]
    h_camera = [r for r in rows if r.angle_source == "camera" and r.h_angle is not None]
    h_none = [r for r in rows if r.h_angle is None]

    print(f"angle_source: radar={len(h_radar)}  estimated={len(h_est)}  "
          f"camera={len(h_camera)}  none={len(h_none)}")
    print()

    # Detection rate of horizontal radar (h_buffer present but live_h missing)
    have_buf = sum(1 for r in rows if r.h_buffer_frame_count)
    have_live = sum(1 for r in rows if r.h_ball_angle is not None)
    print("Horizontal radar detection rate:")
    print(f"  shots with horizontal kld7 buffer logged: {have_buf}")
    print(f"  shots where live ball_angle was returned: {have_live}  "
          f"({have_live*100/max(1, have_buf):.1f}% of bufs)")
    miss = have_buf - have_live
    print(f"  shots where horizontal radar saw nothing: {miss}  "
          f"({miss*100/max(1, have_buf):.1f}%)")
    print()

    print(fmt_stat("h_angle ALL    ", [r.h_angle for r in rows if r.h_angle is not None]))
    print(fmt_stat("h_angle radar  ", [r.h_angle for r in h_radar]))
    if h_radar:
        absh = [abs(r.h_angle) for r in h_radar]
        wall_close = sum(1 for v in absh if v >= 14.0)
        wall_at = sum(1 for v in absh if v >= 14.9)
        print(f"   |h|>=14°: {wall_close} ({wall_close*100/len(absh):.1f}%)   "
              f"|h|>=14.9°: {wall_at}")
    print()

    # Confidence vs |h|
    print("Confidence vs |h_angle| (radar-sourced):")
    cb = defaultdict(list)
    for r in h_radar:
        if r.confidence is not None:
            cb[round(r.confidence, 1)].append(abs(r.h_angle))
    print(f"  {'conf':>6} {'n':>5} {'mean|h|':>9} {'stdev':>7} {'max':>7}")
    for k in sorted(cb):
        xs = cb[k]
        print(f"  {k:>6.1f} {len(xs):>5} {stats.mean(xs):>9.2f} "
              f"{stats.pstdev(xs):>7.2f} {max(xs):>7.1f}")
    print()

    # Shot-to-shot deltas
    print("Shot-to-shot |Δh_angle| (radar-sourced, same session):")
    deltas = []
    for sid in {r.session_id for r in h_radar}:
        seq = sorted([r for r in h_radar if r.session_id == sid],
                     key=lambda r: r.shot_number)
        for a, b in zip(seq, seq[1:]):
            deltas.append(abs(b.h_angle - a.h_angle))
    print("  " + fmt_stat("|delta|", deltas))
    print()

    # Face vs path
    pairs = [(r.h_angle, r.club_path_deg, r.spin_axis_deg)
             for r in rows
             if r.h_angle is not None and r.club_path_deg is not None]
    if pairs:
        diffs = [f - p for f, p, _ in pairs]
        print(f"Face vs path (radar): n={len(pairs)}  "
              f"mean(face-path)={stats.mean(diffs):+.2f}  "
              f"stdev={stats.pstdev(diffs):.2f}  "
              f"min={min(diffs):+.1f}  max={max(diffs):+.1f}")
        ext15 = sum(1 for d in diffs if abs(d) > 15.0)
        ext20 = sum(1 for d in diffs if abs(d) > 20.0)
        print(f"  |face - path| > 15°: {ext15} ({ext15*100/len(diffs):.1f}%)")
        print(f"  |face - path| > 20°: {ext20} ({ext20*100/len(diffs):.1f}%)")

        # corr
        xs = [p for _, p, _ in pairs]
        ys = [f for f, _, _ in pairs]
        if len(xs) >= 5 and stats.pstdev(xs) > 0 and stats.pstdev(ys) > 0:
            mx, my = stats.mean(xs), stats.mean(ys)
            cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
            r = cov / (stats.pstdev(xs) * stats.pstdev(ys))
            print(f"  corr(face, path) = {r:+.3f}  "
                  f"(near 0 = independent peaks; near 1 = same target)")
    print()

    # Per-session
    print("Per-session radar h_angle stats:")
    print(f"  {'session':<24}{'n':>4}{'detect%':>9}{'mean':>8}"
          f"{'stdev':>8}{'med|Δ|':>9}{'p90|Δ|':>9}")
    for sid in sorted({r.session_id for r in rows}):
        ses = [r for r in rows if r.session_id == sid]
        rad = [r for r in ses
               if r.angle_source == "radar" and r.h_angle is not None]
        if not rad:
            continue
        bufd = [r for r in ses if r.h_buffer_frame_count]
        live = [r for r in ses if r.h_ball_angle is not None]
        seq = sorted(rad, key=lambda r: r.shot_number)
        deltas = [abs(b.h_angle - a.h_angle) for a, b in zip(seq, seq[1:])]
        dr = (len(live) / len(bufd) * 100) if bufd else float("nan")
        med = percentile(deltas, 0.5) if deltas else float("nan")
        p90 = percentile(deltas, 0.9) if deltas else float("nan")
        print(f"  {sid:<24}{len(rad):>4}{dr:>8.1f}%"
              f"{stats.mean([r.h_angle for r in rad]):>+8.2f}"
              f"{stats.pstdev([r.h_angle for r in rad]):>8.2f}"
              f"{med:>9.2f}{p90:>9.2f}")
    print()


# ---------- plots ----------


def plot_distribution(rows: list[ShotRow], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    h_radar = [r for r in rows
               if r.angle_source == "radar" and r.h_angle is not None]

    # 1) Histogram with ±15° wall
    ax = axes[0, 0]
    ax.hist([r.h_angle for r in h_radar],
            bins=np.arange(-16, 17, 1), color=C_FACE, edgecolor="k", alpha=0.8)
    ax.axvline(-15, color=C_DIFF, linestyle="--", label="±15° rejection wall")
    ax.axvline(15, color=C_DIFF, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("horizontal launch angle (deg)")
    ax.set_ylabel("count")
    ax.set_title(f"Horizontal angle distribution (radar-sourced, n={len(h_radar)})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # 2) Confidence vs |h|
    ax = axes[0, 1]
    confs = [r.confidence for r in h_radar if r.confidence is not None]
    abs_h = [abs(r.h_angle) for r in h_radar if r.confidence is not None]
    ax.scatter(confs, abs_h, alpha=0.4, color=C_FACE, s=20)
    # binned mean overlay
    cb = defaultdict(list)
    for c, h in zip(confs, abs_h):
        cb[round(c, 1)].append(h)
    if cb:
        ks = sorted(cb)
        ax.plot(ks, [stats.mean(cb[k]) for k in ks],
                color=C_DIFF, marker="o", linewidth=2, label="binned mean")
        ax.legend()
    ax.set_xlabel("launch_angle_confidence")
    ax.set_ylabel("|h_angle| (deg)")
    ax.set_title("Confidence vs |h_angle|")
    ax.grid(alpha=0.3)

    # 3) Shot-to-shot |delta| sequence per session
    ax = axes[1, 0]
    sessions = sorted({r.session_id for r in h_radar})
    cmap = plt.get_cmap("tab10")
    for i, sid in enumerate(sessions):
        seq = sorted([r for r in h_radar if r.session_id == sid],
                     key=lambda r: r.shot_number)
        ds = [abs(b.h_angle - a.h_angle) for a, b in zip(seq, seq[1:])]
        if ds:
            ax.plot(ds, alpha=0.55, color=cmap(i % 10),
                    label=sid.replace("session_", ""))
    ax.set_xlabel("consecutive shot pair index")
    ax.set_ylabel("|Δ h_angle| (deg)")
    ax.set_title("Shot-to-shot volatility")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # 4) face - path
    ax = axes[1, 1]
    diffs = [(r.h_angle - r.club_path_deg) for r in rows
             if r.h_angle is not None and r.club_path_deg is not None]
    ax.hist(diffs, bins=np.arange(-30, 31, 2), color=C_DIFF, edgecolor="k", alpha=0.8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.axvline(15, color="k", linestyle="--", alpha=0.5)
    ax.axvline(-15, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("face - path (deg)  ≈ derived spin axis")
    ax.set_ylabel("count")
    ax.set_title(f"Face vs path consistency (n={len(diffs)})")
    ax.grid(alpha=0.3)

    fig.suptitle("Horizontal launch angle diagnostics", fontsize=14)
    fig.tight_layout()
    out = out_dir / "h_angle_overview.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_per_session(rows: list[ShotRow], out_dir: Path) -> None:
    sessions = sorted({r.session_id for r in rows})
    for sid in sessions:
        ses = [r for r in rows if r.session_id == sid]
        ses.sort(key=lambda r: r.shot_number)
        if not ses:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        xs = [r.shot_number for r in ses]
        ax = axes[0]
        ax.plot(xs, [r.h_angle for r in ses], "o-",
                color=C_FACE, markersize=4, label="face (h_angle)", alpha=0.8)
        ax.plot(xs, [r.club_path_deg for r in ses], "s-",
                color=C_PATH, markersize=4, label="path", alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axhline(15, color=C_DIFF, linestyle="--", alpha=0.4, label="±15° wall")
        ax.axhline(-15, color=C_DIFF, linestyle="--", alpha=0.4)
        ax.set_ylabel("angle (deg)")
        ax.set_title(f"{sid}  —  face/path per shot")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ball_speed = [r.ball_speed_mph or np.nan for r in ses]
        ax.plot(xs, ball_speed, "-", color="#2196F3", label="ball speed")
        ax2 = ax.twinx()
        snrs = [r.h_avg_snr_db if r.h_avg_snr_db is not None else np.nan
                for r in ses]
        ax2.plot(xs, snrs, "-", color="#4CAF50", alpha=0.7, label="h-radar avg SNR")
        ax.set_ylabel("ball speed (mph)", color="#2196F3")
        ax2.set_ylabel("avg SNR (dB)", color="#4CAF50")
        ax.grid(alpha=0.3)

        ax = axes[2]
        confs = [r.confidence if r.confidence is not None else 0 for r in ses]
        sources = [r.angle_source or "none" for r in ses]
        cmap = {"radar": "#4CAF50", "estimated": "#9E9E9E",
                "camera": "#2196F3", "none": "#F44336"}
        ax.bar(xs, confs, color=[cmap.get(s, "#F44336") for s in sources],
               edgecolor="k", linewidth=0.3)
        ax.set_ylabel("confidence")
        ax.set_xlabel("shot number")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        # legend for source colors
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=v, label=k) for k, v in cmap.items()],
                  fontsize=7, loc="upper right")

        fig.tight_layout()
        out = out_dir / f"{sid}_h_angle.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


def plot_face_vs_path_scatter(rows: list[ShotRow], out_dir: Path) -> None:
    pairs = [(r.h_angle, r.club_path_deg, r.confidence or 0)
             for r in rows
             if r.h_angle is not None and r.club_path_deg is not None]
    if not pairs:
        return
    xs = [p for _, p, _ in pairs]
    ys = [f for f, _, _ in pairs]
    cs = [c for _, _, c in pairs]
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=30,
                    edgecolor="k", linewidth=0.3)
    lim = max(15, max(abs(min(xs + ys)), abs(max(xs + ys))))
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5,
            label="face = path (zero spin axis)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("club path (deg)")
    ax.set_ylabel("face / horizontal launch angle (deg)")
    ax.set_title(f"Face vs path scatter (n={len(pairs)})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.colorbar(sc, ax=ax, label="confidence")
    out = out_dir / "face_vs_path_scatter.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- CSV dump ----------


def write_csv(rows: list[ShotRow], out_dir: Path) -> None:
    out = out_dir / "h_angle_diag.csv"
    fields = [
        "session_id", "shot_number", "ts", "ball_speed_mph", "club_speed_mph",
        "angle_source", "h_angle", "v_angle", "confidence",
        "club_path_deg", "spin_axis_deg",
        "h_ball_angle", "h_confidence", "h_num_frames", "h_avg_snr_db",
        "h_buffer_frame_count", "h_buffer_span_s",
        "v_buffer_frame_count", "v_ball_angle", "v_confidence",
    ]
    with out.open("w") as fh:
        fh.write(",".join(fields) + "\n")
        for r in rows:
            fh.write(",".join(_csv(getattr(r, f)) for f in fields) + "\n")
    print(f"wrote {out}  ({len(rows)} rows)")


def _csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v).replace(",", ";")


# ---------- RADC capture mode (raw bytes — full FFT replay) ----------


def _import_radc():
    """Import openflight.kld7.radc lazily so JSONL mode doesn't pay the cost."""
    import sys

    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from openflight.kld7 import radc  # type: ignore

    return radc


def load_radc_capture(path: Path) -> dict:
    """Load a kld7_radc_*.pkl capture (output of capture_kld7_radc.py)."""
    import pickle

    with path.open("rb") as fh:
        return pickle.load(fh)


def group_frames_around_shots(
    capture: dict, ms_before: float = 1500.0, ms_after: float = 700.0,
) -> list[dict]:
    """For each OPS243 shot in the capture, gather K-LD7 frames around it.

    Mirrors the live tracker's get_angle_for_shot window so the offline
    replay sees the same frame set that the live pipeline would have.
    """
    shots = capture.get("ops243_shots") or []
    frames = capture.get("frames") or []
    out = []
    for s in shots:
        t = s.get("timestamp")
        if t is None:
            continue
        sel = [
            f for f in frames
            if f.get("radc") is not None
            and f.get("timestamp") is not None
            and (t - ms_before / 1000.0) <= f["timestamp"] <= (t + ms_after / 1000.0)
        ]
        out.append({
            "shot_timestamp": t,
            "ball_speed_mph": s.get("ball_speed_mph"),
            "club_speed_mph": s.get("club_speed_mph"),
            "frames": sel,
        })
    return out


def per_frame_breakdown(
    frames: list[dict],
    ball_speed_mph: Optional[float],
    fft_size: int = 2048,
    max_speed_kmh: float = 100.0,
    band_tol_mph: float = 10.0,
) -> list[dict]:
    """For each frame in the impact window, compute the ball-band peak bin,
    per-bin angle at that peak, peak SNR, and the velocity that bin maps to.

    This is the diagnostic version of what extract_launch_angle does
    internally — but exposes the per-frame decisions instead of returning
    a single weighted answer.
    """
    radc = _import_radc()

    if ball_speed_mph is not None:
        ball_bands = radc.ball_bin_range_from_speed(
            ball_speed_mph, band_tol_mph, fft_size, max_speed_kmh,
        )
    else:
        ball_bands = [(
            radc._velocity_to_bin(-39.0, fft_size, max_speed_kmh),
            radc._velocity_to_bin(-7.0, fft_size, max_speed_kmh),
        )]

    rows = []
    for fi, frame in enumerate(frames):
        rb = frame.get("radc")
        if rb is None:
            continue
        try:
            channels = radc.parse_radc_payload(rb)
        except ValueError:
            continue
        f1a_iq = radc.to_complex_iq(channels["f1a_i"], channels["f1a_q"])
        f2a_iq = radc.to_complex_iq(channels["f2a_i"], channels["f2a_q"])
        spec = radc.compute_spectrum(f1a_iq, fft_size=fft_size)
        # Find the global peak across all sub-bands (handles wrap)
        peak_bin = -1
        peak_val = 0.0
        for sub_lo, sub_hi in ball_bands:
            sub = spec[sub_lo:sub_hi]
            if sub.size == 0:
                continue
            sub_max = float(sub.max())
            if sub_max > peak_val:
                peak_val = sub_max
                peak_bin = sub_lo + int(np.argmax(sub))
        if peak_val <= 0 or peak_bin < 0:
            continue
        full_pos = spec[spec > 0]
        full_median = float(np.median(full_pos)) if full_pos.size else 0.0
        snr = peak_val / full_median if full_median > 0 else 0.0

        f1a_fft = radc.compute_fft_complex(f1a_iq, fft_size=fft_size)
        f2a_fft = radc.compute_fft_complex(f2a_iq, fft_size=fft_size)
        angles = radc.per_bin_angle_deg(f1a_fft, f2a_fft)
        peak_angle = float(angles[peak_bin])
        # All angles inside the ball band (concatenated across sub-ranges)
        ball_angles_parts = [angles[lo:hi] for lo, hi in ball_bands]
        ball_angles = np.concatenate(ball_angles_parts) if ball_angles_parts else np.array([])

        rows.append({
            "frame_index": fi,
            "timestamp": frame.get("timestamp"),
            "peak_bin": peak_bin,
            "peak_velocity_kmh": radc.bin_to_velocity_kmh(
                peak_bin, fft_size, max_speed_kmh,
            ),
            "peak_snr": snr,
            "peak_angle_deg": peak_angle,
            "ball_bands": ball_bands,
            "ball_band_max_db": 20.0 * np.log10(peak_val) if peak_val > 0 else 0.0,
            "ball_band_angles_min": float(ball_angles.min()) if ball_angles.size else 0.0,
            "ball_band_angles_max": float(ball_angles.max()) if ball_angles.size else 0.0,
            "ball_band_angle_p50": float(np.percentile(ball_angles, 50)) if ball_angles.size else 0.0,
        })
    return rows


def strict_snr_replay(
    breakdown: list[dict],
    snr_floor: float = 8.0,
    cluster_bin_tol: int = 10,
    angle_offset_deg: float = 0.0,
) -> Optional[dict]:
    """Strict-SNR variant of the live aggregation step.

    Operates on the per-frame breakdown produced by `per_frame_breakdown`
    rather than re-running the FFT, so it tests *frame-selection* changes
    in isolation while leaving spectrum/peak/angle math identical to live.

    Pipeline:
      1. Drop frames with peak SNR < `snr_floor`.
      2. Require >=2 surviving frames whose peak bins agree within
         +/- `cluster_bin_tol`. Pick the densest such cluster
         (greedy: for each anchor frame, count frames within tol; the
         anchor with the largest count wins; ties broken by SNR sum).
      3. Take the SNR^2-weighted angle across the cluster only.

    Returns None when the strict criteria fail.
    """
    if not breakdown:
        return None

    survivors = [b for b in breakdown if b["peak_snr"] >= snr_floor]
    if len(survivors) < 2:
        return None

    bins = np.array([b["peak_bin"] for b in survivors], dtype=int)
    snrs = np.array([b["peak_snr"] for b in survivors], dtype=float)
    angs = np.array([b["peak_angle_deg"] for b in survivors], dtype=float)

    # Greedy densest-cluster: for each frame, count how many other
    # frames have peak_bin within +/- cluster_bin_tol of it.
    best_anchor = -1
    best_count = 0
    best_snr_sum = -1.0
    for i in range(len(survivors)):
        mask = np.abs(bins - bins[i]) <= cluster_bin_tol
        count = int(mask.sum())
        snr_sum = float(snrs[mask].sum())
        if count > best_count or (count == best_count and snr_sum > best_snr_sum):
            best_anchor = i
            best_count = count
            best_snr_sum = snr_sum

    if best_count < 2:
        return None

    cluster_mask = np.abs(bins - bins[best_anchor]) <= cluster_bin_tol
    cluster_bins = bins[cluster_mask]
    cluster_snrs = snrs[cluster_mask]
    cluster_angs = angs[cluster_mask]

    w = cluster_snrs ** 2
    total_w = float(w.sum())
    if total_w <= 0:
        return None
    weighted_angle = float(np.sum(cluster_angs * w) / total_w)
    corrected_angle = weighted_angle + angle_offset_deg

    return {
        "launch_angle_deg": round(corrected_angle, 1),
        "raw_angle_deg": round(weighted_angle, 1),
        "frame_count": int(best_count),
        "frames_total": len(breakdown),
        "frames_above_snr": len(survivors),
        "avg_snr_db": round(float(cluster_snrs.mean()), 1),
        "min_snr_db": round(float(cluster_snrs.min()), 1),
        "angle_std_deg": round(float(cluster_angs.std()), 1),
        "cluster_bin_center": int(np.median(cluster_bins)),
        "cluster_bin_span": int(cluster_bins.max() - cluster_bins.min()),
        "snr_floor": snr_floor,
        "cluster_bin_tol": cluster_bin_tol,
    }


def anchored_replay(
    breakdown: list[dict],
    ops_expected_bin: int,
    ops_bin_tol: int = 25,
    snr_floor: float = 2.0,
    angle_offset_deg: float = 0.0,
) -> Optional[dict]:
    """OPS-bin-anchored variant.

    Reject any frame whose peak bin is more than `ops_bin_tol` bins away
    from `ops_expected_bin`. This kills persistent clutter stripes that
    sit outside the actual ball location, even when their SNR is large.
    Then SNR^2-weight the survivors the same way live does.
    """
    if not breakdown or ops_expected_bin is None:
        return None
    survivors = [
        b for b in breakdown
        if b["peak_snr"] >= snr_floor
        and abs(b["peak_bin"] - ops_expected_bin) <= ops_bin_tol
    ]
    if not survivors:
        return None
    snrs = np.array([b["peak_snr"] for b in survivors], dtype=float)
    angs = np.array([b["peak_angle_deg"] for b in survivors], dtype=float)
    bins = np.array([b["peak_bin"] for b in survivors], dtype=int)
    w = snrs ** 2
    total_w = float(w.sum())
    if total_w <= 0:
        return None
    weighted_angle = float(np.sum(angs * w) / total_w)
    return {
        "launch_angle_deg": round(weighted_angle + angle_offset_deg, 1),
        "raw_angle_deg": round(weighted_angle, 1),
        "frame_count": len(survivors),
        "frames_total": len(breakdown),
        "avg_snr_db": round(float(snrs.mean()), 1),
        "max_snr_db": round(float(snrs.max()), 1),
        "angle_std_deg": round(float(angs.std()), 1),
        "bin_offset_mean": int(round(float(np.mean(bins - ops_expected_bin)))),
        "bin_offset_max": int(np.max(np.abs(bins - ops_expected_bin))),
        "ops_expected_bin": int(ops_expected_bin),
        "ops_bin_tol": int(ops_bin_tol),
    }


def replay_capture(
    path: Path,
    out_dir: Path,
    dc_mask_bins: Optional[int] = None,
    aim_tags: Optional[list[str]] = None,
    strict_snr: bool = False,
    strict_snr_floor: float = 8.0,
    strict_cluster_tol: int = 10,
    band_tol_mph: float = 10.0,
    ops_bin_tol: Optional[int] = None,
) -> None:
    radc = _import_radc()

    # Optional DC-mask override (affects compute_spectrum and compute_fft_complex
    # via their default dc_mask_bins=DC_MASK_BINS arguments — but those defaults
    # are bound at function-def time, so we patch by name instead).
    original_dc = radc.DC_MASK_BINS
    if dc_mask_bins is not None and dc_mask_bins != original_dc:
        radc.DC_MASK_BINS = dc_mask_bins
        # Patch the function-level defaults so they actually use the new value.
        # (Python keyword defaults are bound at def time; we have to rebind.)
        radc.compute_spectrum.__defaults__ = (
            radc.compute_spectrum.__defaults__[0], dc_mask_bins,
        )
        radc.compute_fft_complex.__defaults__ = (
            radc.compute_fft_complex.__defaults__[0], dc_mask_bins,
        )
        print(f"NOTE: DC mask overridden to {dc_mask_bins} bins "
              f"(default {original_dc})")

    try:
        cap = load_radc_capture(path)
        md = cap.get("metadata", {})
        orientation = md.get("orientation", "?")
        print()
        print("=" * 100)
        print(f"RADC CAPTURE REPLAY  —  {path.name}")
        print(f"  orientation = {orientation}")
        print(f"  ops243 shots in capture = "
              f"{len(cap.get('ops243_shots') or [])}")
        print(f"  total frames = {md.get('total_frames')}  "
              f"(with RADC: {md.get('radc_frames')})")
        print(f"  dc_mask_bins = {radc.DC_MASK_BINS}")
        print("=" * 100)

        grouped = group_frames_around_shots(cap)
        if not grouped:
            print("No OPS243 shots found in capture; nothing to replay.")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        summary = []

        for shot_idx, g in enumerate(grouped, start=1):
            ball_mph = g["ball_speed_mph"]
            frames = g["frames"]
            if not frames:
                continue

            # Live pipeline result
            results = radc.extract_launch_angle(
                frames=frames,
                ops243_ball_speed_mph=ball_mph,
                speed_tolerance_mph=band_tol_mph,
                orientation=None,  # don't apply hard bounds — we want to see them
            )
            live = results[0] if results else None

            # Per-frame breakdown — must match the live band tolerance
            breakdown = per_frame_breakdown(
                frames, ball_mph, band_tol_mph=band_tol_mph,
            )

            tag = aim_tags[shot_idx - 1] if (
                aim_tags and shot_idx - 1 < len(aim_tags)
            ) else None

            strict = (
                strict_snr_replay(
                    breakdown,
                    snr_floor=strict_snr_floor,
                    cluster_bin_tol=strict_cluster_tol,
                )
                if strict_snr else None
            )

            anchored = None
            ops_bin = (
                _ops243_expected_bin(ball_mph)
                if (ball_mph is not None and ops_bin_tol is not None)
                else None
            )
            if ops_bin_tol is not None and ops_bin is not None:
                anchored = anchored_replay(
                    breakdown,
                    ops_expected_bin=ops_bin,
                    ops_bin_tol=ops_bin_tol,
                    snr_floor=2.0,
                )

            summary.append({
                "shot_idx": shot_idx,
                "ball_mph": ball_mph,
                "aim_tag": tag,
                "frames_in_window": len(frames),
                "frames_with_ball_detection": len(breakdown),
                "live_angle": live["launch_angle_deg"] if live else None,
                "live_confidence": live["confidence"] if live else None,
                "live_snr": live["avg_snr_db"] if live else None,
                "frame_count": live["frame_count"] if live else 0,
                "angle_std_deg": live["angle_std_deg"] if live else None,
                "strict_angle": strict["launch_angle_deg"] if strict else None,
                "strict_frame_count": strict["frame_count"] if strict else 0,
                "strict_avg_snr": strict["avg_snr_db"] if strict else None,
                "strict_min_snr": strict["min_snr_db"] if strict else None,
                "strict_angle_std": strict["angle_std_deg"] if strict else None,
                "strict_cluster_bin": (
                    strict["cluster_bin_center"] if strict else None
                ),
                "strict_cluster_span": (
                    strict["cluster_bin_span"] if strict else None
                ),
                "strict_frames_above_snr": (
                    strict["frames_above_snr"] if strict else 0
                ),
                "anchored_angle": (
                    anchored["launch_angle_deg"] if anchored else None
                ),
                "anchored_frame_count": (
                    anchored["frame_count"] if anchored else 0
                ),
                "anchored_avg_snr": (
                    anchored["avg_snr_db"] if anchored else None
                ),
                "anchored_max_snr": (
                    anchored["max_snr_db"] if anchored else None
                ),
                "anchored_angle_std": (
                    anchored["angle_std_deg"] if anchored else None
                ),
                "anchored_bin_offset_mean": (
                    anchored["bin_offset_mean"] if anchored else None
                ),
                "anchored_bin_offset_max": (
                    anchored["bin_offset_max"] if anchored else None
                ),
                "ops_expected_bin": ops_bin,
                "band_tol_mph": band_tol_mph,
                "ops_bin_tol": ops_bin_tol,
                "per_frame_angles": [b["peak_angle_deg"] for b in breakdown],
                "per_frame_snr": [b["peak_snr"] for b in breakdown],
                "per_frame_bins": [b["peak_bin"] for b in breakdown],
            })

            # Plot per-shot diagnostics
            if breakdown:
                _plot_radc_shot(shot_idx, g, breakdown, live, out_dir, tag)
                _plot_radc_shot_spectrum(shot_idx, g, breakdown, out_dir, tag=tag)
                _plot_radc_shot_spectrogram(shot_idx, g, breakdown, out_dir, tag=tag)

        _print_radc_summary(summary, orientation)
        if strict_snr:
            _print_strict_snr_summary(
                summary, strict_snr_floor, strict_cluster_tol,
            )
            _plot_strict_vs_live(summary, out_dir)
        if ops_bin_tol is not None:
            _print_anchored_summary(summary, band_tol_mph, ops_bin_tol)
            _plot_anchored_vs_live(summary, out_dir)
        _plot_radc_overview(summary, orientation, out_dir)
        if any(s.get("aim_tag") for s in summary):
            _plot_radc_aim_vs_angle(summary, out_dir)
        _write_radc_csv(
            summary, path, out_dir,
            strict_snr=strict_snr, anchored=ops_bin_tol is not None,
        )
    finally:
        if dc_mask_bins is not None and dc_mask_bins != original_dc:
            radc.DC_MASK_BINS = original_dc
            radc.compute_spectrum.__defaults__ = (
                radc.compute_spectrum.__defaults__[0], original_dc,
            )
            radc.compute_fft_complex.__defaults__ = (
                radc.compute_fft_complex.__defaults__[0], original_dc,
            )


def _print_radc_summary(summary: list[dict], orientation: str) -> None:
    if not summary:
        print("\nno shots produced any frames; cannot summarize")
        return

    detected = [s for s in summary if s["live_angle"] is not None]
    print()
    print("Replay summary:")
    print(f"  shots replayed          : {len(summary)}")
    print(f"  shots with live angle   : {len(detected)}  "
          f"({len(detected)*100/len(summary):.1f}%)")

    if detected:
        angles = [s["live_angle"] for s in detected]
        confs = [s["live_confidence"] for s in detected]
        snrs = [s["live_snr"] for s in detected]
        print(f"  live angle              : "
              f"mean={np.mean(angles):+.2f}  std={np.std(angles):.2f}  "
              f"min={min(angles):+.1f}  max={max(angles):+.1f}")
        print(f"  live confidence         : "
              f"mean={np.mean(confs):.2f}  min={min(confs):.2f}")
        print(f"  live avg SNR (dB)       : "
              f"mean={np.mean(snrs):.1f}  min={min(snrs):.1f}")

    # Per-frame stability — are peaks bouncing across bins?
    per_shot_bin_jumps = []
    per_shot_angle_jumps = []
    for s in summary:
        bins = s["per_frame_bins"]
        angs = s["per_frame_angles"]
        if len(bins) >= 2:
            per_shot_bin_jumps.append(
                float(np.mean(np.abs(np.diff(bins))))
            )
            per_shot_angle_jumps.append(
                float(np.mean(np.abs(np.diff(angs))))
            )
    if per_shot_bin_jumps:
        print(f"  mean |Δ peak_bin| / shot   : "
              f"{np.mean(per_shot_bin_jumps):.1f} bins  "
              f"(small = stable peak; large = jumping target)")
        print(f"  mean |Δ peak_angle| / shot : "
              f"{np.mean(per_shot_angle_jumps):.1f}°  "
              f"(small = same target; large = noise-dominated)")


def _plot_radc_shot(
    shot_idx: int, group: dict, breakdown: list[dict],
    live_result: Optional[dict], out_dir: Path,
    tag: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    fis = [b["frame_index"] for b in breakdown]
    bins = [b["peak_bin"] for b in breakdown]
    snrs = [b["peak_snr"] for b in breakdown]
    angs = [b["peak_angle_deg"] for b in breakdown]
    bands = breakdown[0].get("ball_bands") or []

    axes[0].plot(fis, bins, "o-", color="#2196F3")
    for i, (rng_lo, rng_hi) in enumerate(bands):
        axes[0].axhline(rng_lo, color="k", linestyle="--", alpha=0.4,
                        label=(f"band lo={rng_lo}" if i == 0 else None))
        axes[0].axhline(rng_hi, color="k", linestyle="--", alpha=0.4,
                        label=(f"band hi={rng_hi}" if i == 0 else None))
    axes[0].set_ylabel("peak FFT bin")
    title = (f"shot {shot_idx}  —  ball_speed={group['ball_speed_mph']:.1f} mph"
             f"  —  {len(breakdown)} of {len(group['frames'])} frames detected")
    if tag:
        title += f"  —  aim={tag}"
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(fis, snrs, "o-", color="#4CAF50")
    axes[1].axhline(2.0, color="k", linestyle="--", alpha=0.4,
                    label="multi-frame SNR floor (2.0)")
    axes[1].axhline(5.0, color="r", linestyle="--", alpha=0.4,
                    label="single-frame SNR floor (5.0)")
    axes[1].set_ylabel("peak SNR (linear)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(fis, angs, "o-", color=C_FACE)
    if live_result is not None:
        axes[2].axhline(live_result["launch_angle_deg"], color=C_DIFF,
                        linestyle="--", alpha=0.8,
                        label=(f"live angle {live_result['launch_angle_deg']:+.1f}°  "
                               f"(conf {live_result['confidence']:.2f})"))
    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].set_ylabel("peak-bin angle (deg)")
    axes[2].set_xlabel("frame index in window")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"radc_shot_{shot_idx:02d}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _ops243_expected_bin(ball_speed_mph: float, fft_size: int = 2048,
                        max_speed_kmh: float = 100.0) -> int:
    """Bin index where the OPS243-measured ball speed should appear in the
    aliased K-LD7 spectrum. Lets us see whether the radar's strongest
    return is actually at the predicted ball location, or drifting elsewhere.
    """
    radc = _import_radc()
    ball_kmh = ball_speed_mph * 1.609
    aliased = ball_kmh % (2.0 * max_speed_kmh)
    if aliased > max_speed_kmh:
        aliased -= 2.0 * max_speed_kmh
    return radc._velocity_to_bin(aliased, fft_size, max_speed_kmh)


def _plot_radc_shot_spectrum(
    shot_idx: int, group: dict, breakdown: list[dict], out_dir: Path,
    fft_size: int = 2048, tag: Optional[str] = None,
) -> None:
    """Single-frame deep dive on the highest-SNR frame.

    Top panel: full magnitude spectrum (log scale) with DC mask, ball band,
    OPS243-expected bin, and picked peak overlaid.
    Bottom panel: per-bin angle on the same x-axis. Lets us see whether the
    'angle' the algorithm is reading is from a ball-shaped spectral peak or
    from a near-DC clutter region.
    """
    if not breakdown:
        return
    radc = _import_radc()

    # Choose the highest-SNR frame
    best_idx = max(range(len(breakdown)), key=lambda i: breakdown[i]["peak_snr"])
    bd = breakdown[best_idx]
    frame = group["frames"][bd["frame_index"]]
    rb = frame.get("radc")
    if rb is None:
        return
    ch = radc.parse_radc_payload(rb)
    iq1 = radc.to_complex_iq(ch["f1a_i"], ch["f1a_q"])
    iq2 = radc.to_complex_iq(ch["f2a_i"], ch["f2a_q"])
    spec = radc.compute_spectrum(iq1, fft_size=fft_size)
    fft1 = radc.compute_fft_complex(iq1, fft_size=fft_size)
    fft2 = radc.compute_fft_complex(iq2, fft_size=fft_size)
    angles = radc.per_bin_angle_deg(fft1, fft2)

    ball_mph = group["ball_speed_mph"]
    expected_bin = _ops243_expected_bin(ball_mph) if ball_mph else None
    bands = bd.get("ball_bands") or []
    peak_bin = bd["peak_bin"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    bins = np.arange(fft_size)

    # Magnitude spectrum (log)
    mag_db = 20.0 * np.log10(np.maximum(spec, 1e-3))
    axes[0].plot(bins, mag_db, color="#1565C0", linewidth=0.8)
    for i, (b_lo, b_hi) in enumerate(bands):
        axes[0].axvspan(b_lo, b_hi, color="#FFC107", alpha=0.18,
                        label=(f"ball band {bands}" if i == 0 else None))
    axes[0].axvspan(0, radc.DC_MASK_BINS, color="#F44336", alpha=0.25,
                    label=f"DC mask (±{radc.DC_MASK_BINS} bins)")
    axes[0].axvspan(fft_size - radc.DC_MASK_BINS, fft_size,
                    color="#F44336", alpha=0.25)
    if expected_bin is not None:
        axes[0].axvline(expected_bin, color="#2E7D32", linestyle="--",
                        linewidth=2,
                        label=f"OPS243-expected bin {expected_bin} ({ball_mph:.1f} mph)")
    axes[0].axvline(peak_bin, color="#7B1FA2", linestyle=":", linewidth=2,
                    label=f"picked peak bin {peak_bin}  "
                          f"(SNR {bd['peak_snr']:.1f}, ang {bd['peak_angle_deg']:+.1f}°)")
    axes[0].set_ylabel("magnitude (dB)")
    title = (f"shot {shot_idx} frame {bd['frame_index']} (highest SNR)  —  "
             f"ball_speed {ball_mph:.1f} mph")
    if tag:
        title += f"  —  aim={tag}"
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)

    # Per-bin angle
    axes[1].plot(bins, angles, color="#7B1FA2", linewidth=0.6, alpha=0.7)
    for b_lo, b_hi in bands:
        axes[1].axvspan(b_lo, b_hi, color="#FFC107", alpha=0.18)
    if expected_bin is not None:
        axes[1].axvline(expected_bin, color="#2E7D32", linestyle="--", linewidth=2)
    axes[1].axvline(peak_bin, color="#7B1FA2", linestyle=":", linewidth=2)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].axhline(15, color="#F44336", linestyle="--", alpha=0.4)
    axes[1].axhline(-15, color="#F44336", linestyle="--", alpha=0.4)
    axes[1].set_xlim(0, fft_size)
    axes[1].set_ylim(-60, 60)
    axes[1].set_xlabel("FFT bin")
    axes[1].set_ylabel("per-bin angle (deg)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"radc_shot_{shot_idx:02d}_spectrum.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_radc_shot_spectrogram(
    shot_idx: int, group: dict, breakdown: list[dict], out_dir: Path,
    fft_size: int = 2048, tag: Optional[str] = None,
) -> None:
    """2-D spectrogram across the impact window: frame index (y) × bin (x).

    Reveals ball-transit signatures that single-frame snapshots can't show.
    A real ball transit traces a streak that arrives, peaks, and fades over
    several frames; a stationary clutter peak shows as a vertical bar.
    """
    if not breakdown:
        return
    radc = _import_radc()

    frames = group["frames"]
    if not frames:
        return
    n_frames = len(frames)
    spec_grid = np.zeros((n_frames, fft_size), dtype=np.float32)
    valid_rows = []
    for fi, frame in enumerate(frames):
        rb = frame.get("radc")
        if rb is None:
            continue
        try:
            ch = radc.parse_radc_payload(rb)
        except ValueError:
            continue
        iq = radc.to_complex_iq(ch["f1a_i"], ch["f1a_q"])
        spec_grid[fi] = radc.compute_spectrum(iq, fft_size=fft_size)
        valid_rows.append(fi)

    if not valid_rows:
        return

    # log scale, clipped to avoid -inf at masked bins
    mag_db = 20.0 * np.log10(np.maximum(spec_grid, 1e-3))
    floor = float(np.percentile(mag_db[mag_db > 0], 50))
    ceiling = float(np.percentile(mag_db, 99.5))

    ball_mph = group["ball_speed_mph"]
    bands = breakdown[0].get("ball_bands") or []
    expected_bin = _ops243_expected_bin(ball_mph) if ball_mph else None
    peak_bins = [b["peak_bin"] for b in breakdown]
    peak_frames = [b["frame_index"] for b in breakdown]

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=floor,
        vmax=ceiling,
        extent=(0, fft_size, 0, n_frames),
        interpolation="nearest",
    )
    for i, (b_lo, b_hi) in enumerate(bands):
        ax.axvspan(b_lo, b_hi, color="#FFC107", alpha=0.10, lw=0)
        ax.axvline(b_lo, color="#FFC107", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(b_hi, color="#FFC107", linestyle="--", linewidth=1, alpha=0.7,
                   label=(f"ball band {bands}" if i == 0 else None))
    if expected_bin is not None:
        ax.axvline(expected_bin, color="#4CAF50", linestyle="--",
                   linewidth=2,
                   label=f"OPS243-expected bin {expected_bin} ({ball_mph:.1f} mph)")
    ax.axvspan(0, radc.DC_MASK_BINS, color="#F44336", alpha=0.20, lw=0)
    ax.axvspan(fft_size - radc.DC_MASK_BINS, fft_size,
               color="#F44336", alpha=0.20, lw=0,
               label="DC mask")
    ax.scatter(peak_bins, [f + 0.5 for f in peak_frames],
               color="#00E5FF", s=20, edgecolor="k", linewidth=0.4,
               label="picked peak per frame")
    ax.set_xlabel("FFT bin")
    ax.set_ylabel("frame index in window")
    title = f"shot {shot_idx} spectrogram  —  ball_speed {ball_mph:.1f} mph"
    if tag:
        title += f"  —  aim={tag}"
    ax.set_title(title)
    ax.set_xlim(0, fft_size)
    ax.set_ylim(0, n_frames)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    fig.colorbar(im, ax=ax, label="magnitude (dB)")
    fig.tight_layout()
    out = out_dir / f"radc_shot_{shot_idx:02d}_spectrogram.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_radc_aim_vs_angle(
    summary: list[dict], out_dir: Path,
) -> None:
    """Plot picked peak angle and bin grouped by aim_tag.

    Two side-by-side panels:
      - left: per-shot live angle (and per-frame distribution as a strip)
              colored by aim_tag — does the angle move with intended aim?
      - right: per-shot median picked peak bin colored by aim_tag — does
               the dominant bin shift with aim, or stay fixed?
    """
    tagged = [s for s in summary if s.get("aim_tag")]
    if not tagged:
        return

    # Stable color per tag
    unique_tags = []
    for s in tagged:
        if s["aim_tag"] not in unique_tags:
            unique_tags.append(s["aim_tag"])
    cmap = plt.get_cmap("tab10")
    color_for = {t: cmap(i % 10) for i, t in enumerate(unique_tags)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: live angle per shot, plus per-frame angle as strip
    ax = axes[0]
    for s in tagged:
        c = color_for[s["aim_tag"]]
        # All per-frame peak angles as faint dots
        if s["per_frame_angles"]:
            ax.scatter([s["shot_idx"]] * len(s["per_frame_angles"]),
                       s["per_frame_angles"], color=c, alpha=0.25, s=18)
        # Live (aggregated) angle as filled marker
        if s["live_angle"] is not None:
            ax.scatter([s["shot_idx"]], [s["live_angle"]],
                       color=c, s=140, edgecolor="k", linewidth=1.0,
                       zorder=5)
    # Legend by tag
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=color_for[t], markersize=10,
                      markeredgecolor="k", label=t)
               for t in unique_tags]
    ax.legend(handles=handles, fontsize=10, title="aim", loc="best")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(15, color="#F44336", linestyle="--", alpha=0.4)
    ax.axhline(-15, color="#F44336", linestyle="--", alpha=0.4)
    ax.set_xlabel("shot index")
    ax.set_ylabel("angle (deg)  —  big dot = live, small = per-frame")
    ax.set_title("Aim vs picked angle")
    ax.grid(alpha=0.3)
    ax.set_xticks([s["shot_idx"] for s in tagged])

    # Right panel: median picked bin per shot
    ax = axes[1]
    for s in tagged:
        c = color_for[s["aim_tag"]]
        if s["per_frame_bins"]:
            med = float(np.median(s["per_frame_bins"]))
            ax.scatter([s["shot_idx"]] * len(s["per_frame_bins"]),
                       s["per_frame_bins"], color=c, alpha=0.3, s=18)
            ax.scatter([s["shot_idx"]], [med],
                       color=c, s=140, edgecolor="k", linewidth=1.0,
                       zorder=5)
    ax.set_xlabel("shot index")
    ax.set_ylabel("picked peak FFT bin  —  big dot = median across frames")
    ax.set_title("Aim vs picked peak bin (does aim move the dominant peak?)")
    ax.grid(alpha=0.3)
    ax.set_xticks([s["shot_idx"] for s in tagged])

    fig.suptitle(
        "Aim verification — if hypothesis 'small radial velocity ball' holds, "
        "BOTH panels should show grouping by aim_tag.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "radc_aim_vs_angle.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _plot_radc_overview(
    summary: list[dict], orientation: str, out_dir: Path,
) -> None:
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    detected = [s for s in summary if s["live_angle"] is not None]
    if detected:
        axes[0].hist([s["live_angle"] for s in detected],
                     bins=np.arange(-20, 21, 1),
                     color=C_FACE, edgecolor="k", alpha=0.8)
        axes[0].axvline(0, color="k", linewidth=0.5)
        axes[0].set_xlabel(f"replayed live angle ({orientation}, deg)")
        axes[0].set_ylabel("count")
        axes[0].set_title("Replay angle distribution")
        axes[0].grid(alpha=0.3)

    # Frames detected vs frames in window — detection rate per shot
    in_win = [s["frames_in_window"] for s in summary]
    detected_frames = [s["frames_with_ball_detection"] for s in summary]
    rate = [
        d / max(1, w) for d, w in zip(detected_frames, in_win)
    ]
    axes[1].hist(rate, bins=20, color="#4CAF50", edgecolor="k", alpha=0.8)
    axes[1].set_xlabel("ball-detection rate per shot  (frames with peak ÷ frames in window)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Per-shot detection rate")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "radc_overview.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def _write_radc_csv(
    summary: list[dict], src: Path, out_dir: Path,
    strict_snr: bool = False,
    anchored: bool = False,
) -> None:
    out = out_dir / "radc_replay.csv"
    fields = [
        "shot_idx", "ball_mph", "aim_tag", "frames_in_window",
        "frames_with_ball_detection",
        "live_angle", "live_confidence", "live_snr", "frame_count",
        "angle_std_deg",
    ]
    if strict_snr:
        fields += [
            "strict_angle", "strict_frame_count", "strict_avg_snr",
            "strict_min_snr", "strict_angle_std",
            "strict_cluster_bin", "strict_cluster_span",
            "strict_frames_above_snr",
        ]
    if anchored:
        fields += [
            "ops_expected_bin", "band_tol_mph", "ops_bin_tol",
            "anchored_angle", "anchored_frame_count",
            "anchored_avg_snr", "anchored_max_snr",
            "anchored_angle_std",
            "anchored_bin_offset_mean", "anchored_bin_offset_max",
        ]
    with out.open("w") as fh:
        fh.write(f"# source: {src.name}\n")
        fh.write(",".join(fields) + "\n")
        for s in summary:
            fh.write(",".join(_csv(s.get(k)) for k in fields) + "\n")
    print(f"wrote {out}")


def _print_anchored_summary(
    summary: list[dict], band_tol_mph: float, ops_bin_tol: int,
) -> None:
    n = len(summary)
    hits = [s for s in summary if s.get("anchored_angle") is not None]
    print()
    print("OPS-bin-anchored replay")
    print(f"  band_tol_mph             : +/- {band_tol_mph:.1f} mph "
          f"(narrow ball band around OPS speed)")
    print(f"  ops_bin_tol              : +/- {ops_bin_tol} bins around "
          f"OPS-expected bin")
    print(f"  shots accepted           : {len(hits)} / {n}  "
          f"({len(hits)*100/max(1,n):.1f}%)")

    if hits:
        angles = [s["anchored_angle"] for s in hits]
        snrs = [s["anchored_avg_snr"] for s in hits]
        fcs = [s["anchored_frame_count"] for s in hits]
        in_bounds = sum(1 for a in angles if abs(a) <= 15.0)
        print(f"  anchored angle           : "
              f"mean={np.mean(angles):+.2f}  std={np.std(angles):.2f}  "
              f"min={min(angles):+.1f}  max={max(angles):+.1f}")
        print(f"  shots in [-15, +15] deg  : {in_bounds} / {len(hits)}")
        print(f"  cluster size             : "
              f"mean={np.mean(fcs):.1f}  max={max(fcs)}")
        print(f"  cluster avg SNR          : "
              f"mean={np.mean(snrs):.1f}  min={min(snrs):.1f}")

    def _f(v: Optional[float], fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    print()
    print(f"  {'shot':>4}  {'aim':>4}  {'mph':>5}  "
          f"{'live':>7}  {'l_n':>3}  {'l_snr':>5}  "
          f"{'anch':>7}  {'a_n':>3}  {'a_snr':>5}  "
          f"{'ops_bin':>7}  {'b_off':>5}")
    for s in summary:
        live = s["live_angle"]
        anch = s["anchored_angle"]
        live_snr = s.get("live_snr")
        anch_snr = s.get("anchored_avg_snr")
        ops_bin = s.get("ops_expected_bin")
        b_off = s.get("anchored_bin_offset_mean")
        delta = ""
        if live is not None and anch is not None:
            delta = f"  d={anch-live:+.1f}"
        print(
            f"  {s['shot_idx']:>4}  "
            f"{(s.get('aim_tag') or '-'):>4}  "
            f"{s['ball_mph']:>5.1f}  "
            f"{_f(live, '+.1f'):>7}  "
            f"{s.get('frame_count', 0):>3}  "
            f"{_f(live_snr, '.1f'):>5}  "
            f"{_f(anch, '+.1f'):>7}  "
            f"{s.get('anchored_frame_count', 0):>3}  "
            f"{_f(anch_snr, '.1f'):>5}  "
            f"{(str(ops_bin) if ops_bin is not None else '-'):>7}  "
            f"{(str(b_off) if b_off is not None else '-'):>5}"
            f"{delta}"
        )


def _plot_anchored_vs_live(summary: list[dict], out_dir: Path) -> None:
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    xs = [s["shot_idx"] for s in summary]
    for s in summary:
        x = s["shot_idx"]
        live = s["live_angle"]
        anch = s["anchored_angle"]
        c = "#1565C0"
        if s.get("aim_tag") == "L":
            c = "#2E7D32"
        elif s.get("aim_tag") == "R":
            c = "#C62828"
        if live is not None:
            ax.scatter([x], [live], color=c, marker="o", s=110,
                       edgecolor="k", linewidth=0.8)
        if anch is not None:
            ax.scatter([x], [anch], color=c, marker="s", s=140,
                       edgecolor="k", linewidth=0.8)
        if live is not None and anch is not None:
            ax.plot([x, x], [live, anch], color=c,
                    linestyle=":", alpha=0.6, linewidth=1.2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(15, color="#F44336", linestyle="--", alpha=0.4)
    ax.axhline(-15, color="#F44336", linestyle="--", alpha=0.4)
    ax.set_xticks(xs)
    ax.set_xlabel("shot index")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Live (circle) vs OPS-anchored (square)\n"
                 "L=green, R=red, untagged=blue")
    ax.grid(alpha=0.3)

    ax = axes[1]
    live_vals = [s["live_angle"] for s in summary
                 if s["live_angle"] is not None]
    anch_vals = [s["anchored_angle"] for s in summary
                 if s["anchored_angle"] is not None]
    edges = np.arange(-20, 21, 1)
    if live_vals:
        ax.hist(live_vals, bins=edges, color="#1565C0", alpha=0.45,
                edgecolor="k", label=f"live (n={len(live_vals)})")
    if anch_vals:
        ax.hist(anch_vals, bins=edges, color="#FF9800", alpha=0.55,
                edgecolor="k", label=f"anchored (n={len(anch_vals)})")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("count")
    ax.set_title("Angle distribution: live vs OPS-anchored")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "radc_anchored_vs_live.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def _print_strict_snr_summary(
    summary: list[dict], snr_floor: float, cluster_tol: int,
) -> None:
    n = len(summary)
    strict_hits = [s for s in summary if s.get("strict_angle") is not None]
    print()
    print("Strict-SNR replay")
    print(f"  snr_floor               : {snr_floor:.1f}")
    print(f"  cluster_bin_tol         : +/- {cluster_tol} bins, "
          f">=2 frames required")
    print(f"  shots accepted          : {len(strict_hits)} / {n}  "
          f"({len(strict_hits)*100/max(1,n):.1f}%)")

    if strict_hits:
        angles = [s["strict_angle"] for s in strict_hits]
        snrs = [s["strict_avg_snr"] for s in strict_hits]
        fc = [s["strict_frame_count"] for s in strict_hits]
        spans = [s["strict_cluster_span"] for s in strict_hits]
        print(f"  strict angle            : "
              f"mean={np.mean(angles):+.2f}  std={np.std(angles):.2f}  "
              f"min={min(angles):+.1f}  max={max(angles):+.1f}")
        print(f"  cluster size            : "
              f"mean={np.mean(fc):.1f} frames  max={max(fc)}")
        print(f"  cluster bin span        : "
              f"mean={np.mean(spans):.1f}  max={max(spans)}  "
              f"(0 = single-bin lock)")
        print(f"  cluster avg SNR         : "
              f"mean={np.mean(snrs):.1f}  min={min(snrs):.1f}")

    # Side-by-side per-shot table
    print()
    print(f"  {'shot':>4}  {'aim':>4}  {'mph':>5}  "
          f"{'live':>7}  {'live_n':>6}  {'live_snr':>8}  "
          f"{'strict':>7}  {'s_n':>3}  {'s_snr':>5}  "
          f"{'s_bin':>5}  {'span':>4}")
    def _f(v: Optional[float], fmt: str) -> str:
        if v is None:
            return "-"
        return format(v, fmt)

    for s in summary:
        live = s["live_angle"]
        strict = s["strict_angle"]
        live_snr = s.get("live_snr")
        strict_snr_v = s.get("strict_avg_snr")
        cluster_bin = s.get("strict_cluster_bin")
        cluster_span = s.get("strict_cluster_span")
        delta_str = ""
        if live is not None and strict is not None:
            delta_str = f"  d={strict-live:+.1f}"
        print(
            f"  {s['shot_idx']:>4}  "
            f"{(s.get('aim_tag') or '-'):>4}  "
            f"{s['ball_mph']:>5.1f}  "
            f"{_f(live, '+.1f'):>7}  "
            f"{s.get('frame_count', 0):>6}  "
            f"{_f(live_snr, '.1f'):>8}  "
            f"{_f(strict, '+.1f'):>7}  "
            f"{s.get('strict_frame_count', 0):>3}  "
            f"{_f(strict_snr_v, '.1f'):>5}  "
            f"{(str(cluster_bin) if cluster_bin is not None else '-'):>5}  "
            f"{(str(cluster_span) if cluster_span is not None else '-'):>4}"
            f"{delta_str}"
        )


def _plot_strict_vs_live(summary: list[dict], out_dir: Path) -> None:
    """Two panels:
      - left: per-shot live angle vs strict angle (paired markers + delta).
      - right: histogram of live and strict angles overlaid for the shots
        where both produced a value.
    """
    if not summary:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    xs = []
    for s in summary:
        x = s["shot_idx"]
        live = s["live_angle"]
        strict = s["strict_angle"]
        tag_color = "#1565C0"
        if s.get("aim_tag") == "L":
            tag_color = "#2E7D32"
        elif s.get("aim_tag") == "R":
            tag_color = "#C62828"
        if live is not None:
            ax.scatter([x], [live], color=tag_color, marker="o", s=110,
                       edgecolor="k", linewidth=0.8,
                       label="live" if x == summary[0]["shot_idx"] else None)
        if strict is not None:
            ax.scatter([x], [strict], color=tag_color, marker="^", s=140,
                       edgecolor="k", linewidth=0.8,
                       label="strict" if x == summary[0]["shot_idx"] else None)
        if live is not None and strict is not None:
            ax.plot([x, x], [live, strict], color=tag_color,
                    linestyle=":", alpha=0.6, linewidth=1.2)
        xs.append(x)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(15, color="#F44336", linestyle="--", alpha=0.4)
    ax.axhline(-15, color="#F44336", linestyle="--", alpha=0.4)
    ax.set_xlabel("shot index")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Live (circle) vs Strict-SNR (triangle)\n"
                 "L=green, R=red, untagged=blue")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    live_vals = [s["live_angle"] for s in summary
                 if s["live_angle"] is not None]
    strict_vals = [s["strict_angle"] for s in summary
                   if s["strict_angle"] is not None]
    bins_edges = np.arange(-20, 21, 1)
    if live_vals:
        ax.hist(live_vals, bins=bins_edges, color="#1565C0", alpha=0.45,
                edgecolor="k", label=f"live (n={len(live_vals)})")
    if strict_vals:
        ax.hist(strict_vals, bins=bins_edges, color="#FF9800", alpha=0.55,
                edgecolor="k", label=f"strict (n={len(strict_vals)})")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("count")
    ax.set_title("Angle distribution: live vs strict")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "radc_strict_vs_live.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- main ----------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("logs", nargs="*", type=Path,
                    help="JSONL session log paths (or globs expanded by shell)")
    ap.add_argument("--radc", type=Path, default=None,
                    help="Path to a kld7_radc_*.pkl raw RADC capture; "
                         "enables full FFT replay diagnostics")
    ap.add_argument("--dc-mask", type=int, default=None,
                    help="Override DC_MASK_BINS for this replay only "
                         "(default 8). Useful to test whether near-DC "
                         "leakage is dominating the picked peak.")
    ap.add_argument("--aim", type=str, default=None,
                    help="Comma-separated aim tags, one per shot in OPS243 "
                         "order. Example: --aim L,L,L,R,R. Enables the "
                         "aim-vs-angle plot.")
    ap.add_argument("--strict-snr", action="store_true",
                    help="Run a strict-SNR variant alongside the live "
                         "extract_launch_angle pipeline. Requires per-frame "
                         "SNR >= --strict-snr-floor and >=2 frames whose "
                         "peak bins agree within +/- --strict-cluster-tol. "
                         "Produces side-by-side plots and a per-shot table.")
    ap.add_argument("--strict-snr-floor", type=float, default=8.0,
                    help="Minimum per-frame peak SNR (linear) to be eligible "
                         "for the strict-SNR cluster (default 8.0).")
    ap.add_argument("--strict-cluster-tol", type=int, default=10,
                    help="Maximum bin distance from the cluster anchor for a "
                         "frame to be included (default +/- 10 bins).")
    ap.add_argument("--band-tol-mph", type=float, default=10.0,
                    help="Width of the OPS-anchored ball-velocity band in "
                         "mph (default +/- 10). Tightening this narrows "
                         "the search around the OPS-measured ball speed.")
    ap.add_argument("--ops-bin-tol", type=int, default=None,
                    help="If set, run an OPS-bin-anchored variant that "
                         "rejects frames whose peak bin is more than this "
                         "many bins from the OPS-expected bin. Useful for "
                         "killing persistent clutter stripes that the "
                         "live algorithm latches onto.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("session_logs/h_angle_diag"),
                    help="Where to write plots and CSV")
    args = ap.parse_args()

    if not args.logs and not args.radc:
        ap.error("provide at least one JSONL log path or --radc <pkl>")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.logs:
        paths = sorted({p.resolve() for p in args.logs if p.exists()})
        if paths:
            rows = load_sessions(paths)
            if rows:
                print_summary(rows)
                plot_distribution(rows, args.output_dir)
                plot_per_session(rows, args.output_dir)
                plot_face_vs_path_scatter(rows, args.output_dir)
                write_csv(rows, args.output_dir)
            else:
                print("warning: no shots in supplied JSONL logs")
        else:
            print("warning: no JSONL log paths matched")

    if args.radc:
        if not args.radc.exists():
            raise SystemExit(f"--radc path not found: {args.radc}")
        aim_tags = (
            [t.strip() for t in args.aim.split(",")] if args.aim else None
        )
        replay_capture(
            args.radc, args.output_dir,
            dc_mask_bins=args.dc_mask, aim_tags=aim_tags,
            strict_snr=args.strict_snr,
            strict_snr_floor=args.strict_snr_floor,
            strict_cluster_tol=args.strict_cluster_tol,
            band_tol_mph=args.band_tol_mph,
            ops_bin_tol=args.ops_bin_tol,
        )

    print()
    print(f"all artifacts in: {args.output_dir}")


if __name__ == "__main__":
    main()
