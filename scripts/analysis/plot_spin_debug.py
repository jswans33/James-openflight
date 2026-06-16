"""Plot the OPS radar's view of a single shot for spin debugging.

Produces a four-panel figure for one rolling_buffer_capture entry:

  [1] Speed timeline      — speed (mph) over time from non-overlapping
                            128-sample FFT blocks, with club/ball/trigger
                            markers and net-arrival shaded.
  [2] I/Q time-domain     — raw I and Q traces post-trigger.
  [3] Bandpass envelope   — what detect_spin actually FFTs: I/Q
                            bandpass-filtered around ball Doppler, then
                            envelope, trimmed for transients.
  [4] Spin FFT spectrum   — envelope FFT magnitude vs RPM, with the
                            DC-leakage guard zone, lower/upper rails,
                            and picked peak annotated. The captured
                            spin_rpm / SNR / rejection reason are shown
                            so you can see what the algorithm decided.

Usage::

    .venv/bin/python scripts/analysis/plot_spin_debug.py \\
        --log session_logs/<session>.jsonl --shot 16
    .venv/bin/python scripts/analysis/plot_spin_debug.py \\
        --log session_logs/<session>.jsonl --shot 16 --net-distance-ft 10
    .venv/bin/python scripts/analysis/plot_spin_debug.py \\
        --log session_logs/<session>.jsonl --shot 16 --tm-spin 5500

The script imports openflight modules with a stub for ops243 so it can
run on a dev machine without pyserial / picamera2.
"""
# ruff: noqa: I001
# Import ordering is intentional: matplotlib backend must be set before
# pyplot import, and several imports follow path/stub setup.

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

if "openflight" not in sys.modules:
    pkg = types.ModuleType("openflight")
    pkg.__path__ = [str(REPO_ROOT / "src" / "openflight")]
    sys.modules["openflight"] = pkg

ops243_stub = types.ModuleType("openflight.ops243")


class _Stub: ...


for _name in ("SpeedReading", "OPS243Radar", "SpeedUnit", "Direction"):
    setattr(ops243_stub, _name, _Stub)
sys.modules.setdefault("openflight.ops243", ops243_stub)

sys.modules.setdefault("openflight.rolling_buffer", types.ModuleType("openflight.rolling_buffer"))
sys.modules["openflight.rolling_buffer"].__path__ = [
    str(REPO_ROOT / "src" / "openflight" / "rolling_buffer")
]

spec = importlib.util.spec_from_file_location(
    "openflight.launch_monitor",
    REPO_ROOT / "src" / "openflight" / "launch_monitor.py",
)
lm = importlib.util.module_from_spec(spec)
sys.modules["openflight.launch_monitor"] = lm
spec.loader.exec_module(lm)

from openflight.rolling_buffer.processor import RollingBufferProcessor  # noqa: E402

from scipy.signal import butter, sosfiltfilt  # noqa: E402


def load_capture(log_path: Path, shot_number: int) -> dict:
    with log_path.open() as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                entry.get("type") == "rolling_buffer_capture"
                and entry.get("shot_number") == shot_number
            ):
                return entry
    raise SystemExit(
        f"shot {shot_number} not found in {log_path} " f"(no rolling_buffer_capture entry)"
    )


def speed_timeline(processor, i, q):
    """Reproduce the standard 128-sample non-overlapping speed FFT
    pipeline. Returns (times_s, speeds_mph, magnitudes, directions).
    """
    SR = processor.SAMPLE_RATE
    WIN = processor.WINDOW_SIZE
    FFT_N = processor.FFT_SIZE
    DC_MASK = processor.DC_MASK_BINS

    times = []
    speeds = []
    mags = []
    dirs = []  # +1 outbound, -1 inbound
    for start in range(0, len(i) - WIN + 1, WIN):
        i_b = np.asarray(i[start : start + WIN], dtype=np.float64)
        q_b = np.asarray(q[start : start + WIN], dtype=np.float64)
        i_b -= np.mean(i_b)
        q_b -= np.mean(q_b)
        i_b *= np.hanning(WIN)
        q_b *= np.hanning(WIN)
        sig = i_b + 1j * q_b
        spec = np.fft.fft(sig, FFT_N)
        mag = np.abs(spec)
        mag[:DC_MASK] = 0
        mag[FFT_N - DC_MASK :] = 0
        peak = int(np.argmax(mag))
        bin_signed = peak if peak <= FFT_N // 2 else peak - FFT_N
        freq = bin_signed * SR / FFT_N
        speed = freq * processor.WAVELENGTH_M / 2 * processor.MPS_TO_MPH
        t = (start + WIN / 2) / SR
        times.append(t)
        speeds.append(speed)
        mags.append(mag[peak])
        dirs.append(1 if speed > 0 else -1)
    return (
        np.array(times),
        np.array(speeds),
        np.array(mags),
        np.array(dirs),
    )


def envelope_for_spin(processor, i, q, ball_speed_mph, ball_timestamp_ms):
    """Reproduce detect_spin's envelope construction so we can plot it
    and the FFT spectrum the picker sees.
    """
    SR = processor.SAMPLE_RATE
    BW = processor.SPIN_BANDPASS_BW_HZ

    i = np.asarray(i, dtype=np.float64) - np.mean(i)
    q = np.asarray(q, dtype=np.float64) - np.mean(q)
    iq = i + 1j * q

    ball_doppler_hz = 2 * (ball_speed_mph / processor.MPS_TO_MPH) / processor.WAVELENGTH_M
    nyq = SR / 2
    low = max((ball_doppler_hz - BW) / nyq, 0.001)
    high = min((ball_doppler_hz + BW) / nyq, 0.999)
    sos = butter(processor.SPIN_BANDPASS_ORDER, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, iq)
    envelope = np.abs(filtered)

    start_sample = max(0, int(ball_timestamp_ms * SR / 1000))
    ball_env = envelope[start_sample:]
    transient = int(SR / BW)
    if len(ball_env) > 2 * transient + processor.SPIN_MIN_SAMPLES:
        ball_env = ball_env[transient:-transient]
        ball_env_start_sample = start_sample + transient
    else:
        ball_env_start_sample = start_sample

    return envelope, ball_env, ball_env_start_sample, ball_doppler_hz


def envelope_fft(processor, ball_env):
    """Compute the envelope FFT exactly as detect_spin does."""
    SR = processor.SAMPLE_RATE
    FFT_N = processor.SPIN_ENVELOPE_FFT_SIZE
    if len(ball_env) < processor.SPIN_MIN_SAMPLES:
        return None
    env_mean = float(np.mean(ball_env))
    detrended = ball_env - env_mean
    windowed = detrended * np.hanning(len(detrended))
    fft = np.fft.fft(windowed, FFT_N)
    freqs = np.fft.fftfreq(FFT_N, d=1 / SR)
    half = FFT_N // 2
    mag = np.abs(fft[1:half])
    freqs = freqs[1:half]
    return freqs, mag


def plot_shot(entry: dict, *, net_distance_ft: float | None, tm_spin: float | None, output: Path):
    proc = RollingBufferProcessor()
    SR = proc.SAMPLE_RATE

    i = entry["i_samples"]
    q = entry["q_samples"]
    ball_speed = float(entry.get("ball_speed_mph") or 0.0)
    ball_ts_ms = float(entry.get("ball_timestamp_ms") or 0.0)
    club_speed = entry.get("club_speed_mph")
    club_ts_ms = entry.get("club_timestamp_ms")
    trigger_offset_ms = entry.get("trigger_offset_ms")  # seg * 128/SR * 1000
    if trigger_offset_ms is None:
        # Fall back to half buffer if not stored.
        trigger_offset_ms = (len(i) / SR / 2) * 1000

    captured_spin_rpm = entry.get("spin_rpm")
    captured_spin_snr = entry.get("spin_snr")
    captured_peak_hz = entry.get("spin_peak_freq_hz")
    captured_quality = entry.get("spin_quality") or ""
    rejection = entry.get("spin_rejection_reason")
    modulation_depth = entry.get("spin_modulation_depth")

    # Build figure with four panels.
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_iq = fig.add_subplot(gs[0, 1])
    ax_env = fig.add_subplot(gs[1, 0])
    ax_fft = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"Shot #{entry.get('shot_number')} — "
        f"ball {ball_speed:.1f} mph, "
        f"club {f'{float(club_speed):.1f}' if club_speed else '—'} mph",
        fontsize=14,
        fontweight="bold",
    )

    # ---------- Panel 1: Speed timeline ----------
    times, speeds, mags, _ = speed_timeline(proc, i, q)
    times_ms = times * 1000
    out_mask = speeds >= 0
    ax_speed.scatter(
        times_ms[out_mask], speeds[out_mask], s=14, c="#2196F3", label="outbound", alpha=0.85
    )
    ax_speed.scatter(
        times_ms[~out_mask],
        -speeds[~out_mask],
        s=14,
        c="#FF9800",
        label="inbound (mag)",
        alpha=0.85,
    )

    if club_ts_ms is not None and club_speed is not None:
        # club_timestamp_ms is relative to ball; reconstruct absolute time
        club_abs_ms = ball_ts_ms + float(club_ts_ms)
        ax_speed.axvline(
            club_abs_ms,
            color="#FF9800",
            linestyle="--",
            alpha=0.6,
            label=f"club @ {float(club_speed):.1f} mph",
        )
    ax_speed.axvline(
        ball_ts_ms, color="#2196F3", linestyle="--", alpha=0.7, label=f"ball @ {ball_speed:.1f} mph"
    )
    ax_speed.axvline(
        trigger_offset_ms,
        color="black",
        linestyle=":",
        alpha=0.6,
        label=f"trigger ({trigger_offset_ms:.0f} ms)",
    )

    if net_distance_ft and ball_speed > 0:
        net_distance_m = net_distance_ft * 0.3048
        ball_speed_mps = ball_speed / proc.MPS_TO_MPH
        net_arrival_ms = ball_ts_ms + (net_distance_m / ball_speed_mps) * 1000
        ax_speed.axvspan(
            net_arrival_ms,
            times_ms[-1],
            alpha=0.10,
            color="red",
            label=f"post-net (>{net_distance_ft} ft)",
        )

    ax_speed.set_xlabel("time (ms)")
    ax_speed.set_ylabel("|speed| (mph)")
    ax_speed.set_title("Speed timeline (32 non-overlapping 128-sample blocks)")
    ax_speed.legend(loc="best", fontsize=8)
    ax_speed.grid(True, alpha=0.3)

    # ---------- Panel 2: I/Q time domain ----------
    t_iq = np.arange(len(i)) / SR * 1000
    # Center on zero so I/Q comparable.
    i_c = np.asarray(i, dtype=np.float64) - np.mean(i)
    q_c = np.asarray(q, dtype=np.float64) - np.mean(q)
    ax_iq.plot(t_iq, i_c, color="#2196F3", linewidth=0.5, label="I (centered)", alpha=0.7)
    ax_iq.plot(t_iq, q_c, color="#F44336", linewidth=0.5, label="Q (centered)", alpha=0.7)
    ax_iq.axvline(ball_ts_ms, color="#2196F3", linestyle="--", alpha=0.7)
    ax_iq.axvline(trigger_offset_ms, color="black", linestyle=":", alpha=0.5)
    ax_iq.set_xlabel("time (ms)")
    ax_iq.set_ylabel("ADC counts (mean-removed)")
    ax_iq.set_title(f"Raw I/Q ({len(i)} samples @ {SR // 1000} ksps)")
    ax_iq.legend(loc="best", fontsize=8)
    ax_iq.grid(True, alpha=0.3)

    # ---------- Panel 3: Bandpass envelope ----------
    envelope_full, ball_env, env_start, ball_doppler_hz = envelope_for_spin(
        proc,
        i,
        q,
        ball_speed,
        ball_ts_ms,
    )
    t_env = np.arange(len(envelope_full)) / SR * 1000
    ax_env.plot(
        t_env, envelope_full, color="#9E9E9E", linewidth=0.6, alpha=0.6, label="full envelope"
    )
    if len(ball_env):
        t_ball = (env_start + np.arange(len(ball_env))) / SR * 1000
        ax_env.plot(
            t_ball,
            ball_env,
            color="#F44336",
            linewidth=0.9,
            label=f"FFT input ({len(ball_env)} samples)",
        )
        if modulation_depth is not None:
            mean_v = float(np.mean(ball_env))
            ax_env.axhline(
                mean_v,
                color="black",
                linestyle=":",
                alpha=0.5,
                label=f"mean (mod={modulation_depth:.3f})",
            )
    ax_env.axvline(ball_ts_ms, color="#2196F3", linestyle="--", alpha=0.7, label="ball start")
    ax_env.set_xlabel("time (ms)")
    ax_env.set_ylabel("envelope (linear)")
    ax_env.set_title(
        f"Bandpass envelope around ball Doppler "
        f"({ball_doppler_hz:.0f} Hz ± {proc.SPIN_BANDPASS_BW_HZ} Hz)"
    )
    ax_env.legend(loc="best", fontsize=8)
    ax_env.grid(True, alpha=0.3)

    # ---------- Panel 4: Spin FFT spectrum ----------
    fft_result = envelope_fft(proc, ball_env)
    if fft_result is None:
        ax_fft.text(
            0.5,
            0.5,
            f"Envelope too short ({len(ball_env)} samples,\n" f"need {proc.SPIN_MIN_SAMPLES})",
            ha="center",
            va="center",
            transform=ax_fft.transAxes,
            fontsize=12,
        )
    else:
        freqs, mag = fft_result
        rpm = freqs * 60
        valid_mask = (freqs >= proc.SPIN_MIN_SEAM_HZ) & (freqs <= proc.SPIN_MAX_SEAM_HZ)
        # Plot the entire spectrum dimly so context is visible, then the
        # search range opaque.
        ax_fft.plot(rpm, mag, color="#9E9E9E", linewidth=0.5, alpha=0.4)
        ax_fft.plot(
            rpm[valid_mask],
            mag[valid_mask],
            color="#3F51B5",
            linewidth=1.0,
            label="seam search range",
        )

        # Apply leakage guard like the picker does
        valid_mag = mag.copy()
        valid_mag[~valid_mask] = 0
        valid_idxs = np.where(valid_mask)[0]
        if len(valid_idxs) > 0:
            leakage_idxs = valid_idxs[: proc.SPIN_DC_LEAKAGE_BINS]
            valid_mag[leakage_idxs] = 0
            leakage_lo = float(rpm[valid_idxs[0]])
            leakage_hi = (
                float(rpm[valid_idxs[proc.SPIN_DC_LEAKAGE_BINS - 1]])
                if len(valid_idxs) >= proc.SPIN_DC_LEAKAGE_BINS
                else float(rpm[valid_idxs[-1]])
            )
            ax_fft.axvspan(
                leakage_lo,
                leakage_hi,
                alpha=0.18,
                color="orange",
                label=f"DC leakage guard ({proc.SPIN_DC_LEAKAGE_BINS} bins)",
            )
            # Lower rail extends beyond leakage by SPIN_UPPER_RAIL_BINS
            rail_end_idx = min(
                proc.SPIN_DC_LEAKAGE_BINS + proc.SPIN_UPPER_RAIL_BINS,
                len(valid_idxs),
            )
            rail_hi = float(rpm[valid_idxs[rail_end_idx - 1]])
            ax_fft.axvspan(leakage_hi, rail_hi, alpha=0.10, color="red", label="lower-rail zone")
            # Upper rail
            upper_rail_start_idx = max(
                0,
                len(valid_idxs) - proc.SPIN_UPPER_RAIL_BINS,
            )
            upper_rail_lo = float(rpm[valid_idxs[upper_rail_start_idx]])
            upper_rail_hi = float(rpm[valid_idxs[-1]])
            ax_fft.axvspan(upper_rail_lo, upper_rail_hi, alpha=0.10, color="red")

        # Mark the picked peak using captured fields when present.
        if captured_peak_hz:
            picked_rpm = float(captured_peak_hz) * 60
            color = "#4CAF50" if (captured_spin_rpm or 0) > 0 else "#F44336"
            label = (
                f"picked: {picked_rpm:.0f} RPM"
                + (f" (SNR {captured_spin_snr:.1f})" if captured_spin_snr else "")
                + (" ✓" if (captured_spin_rpm or 0) > 0 else " ✗ rejected")
            )
            ax_fft.axvline(picked_rpm, color=color, linewidth=2, label=label)

        if tm_spin:
            ax_fft.axvline(
                tm_spin,
                color="black",
                linewidth=1.5,
                linestyle="--",
                label=f"Trackman truth: {tm_spin:.0f} RPM",
            )

        ax_fft.set_xlim(0, proc.SPIN_MAX_SEAM_HZ * 60 * 1.05)
        ax_fft.set_xlabel("RPM")
        ax_fft.set_ylabel("FFT magnitude (linear)")
        title = "Spin envelope FFT spectrum"
        if rejection:
            title += f"  —  rejected: {rejection[:60]}"
        elif captured_spin_rpm:
            title += f"  —  reported: {int(captured_spin_rpm)} RPM, quality={captured_quality}"
        ax_fft.set_title(title, fontsize=10)
        ax_fft.legend(loc="best", fontsize=8)
        ax_fft.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path, help="Session JSONL file")
    p.add_argument("--shot", required=True, type=int, help="shot_number to plot")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: ./plots_<session>/shot<N>_spin.png)",
    )
    p.add_argument(
        "--net-distance-ft",
        type=float,
        default=None,
        help="Net distance in feet (shades post-net region in panel 1)",
    )
    p.add_argument(
        "--tm-spin", type=float, default=None, help="Trackman ground-truth spin RPM for panel 4"
    )
    args = p.parse_args()

    if not args.log.exists():
        print(f"Session log not found: {args.log}", file=sys.stderr)
        return 2

    entry = load_capture(args.log, args.shot)

    if args.output is None:
        out_dir = args.log.parent / f"plots_{args.log.stem}"
        args.output = out_dir / f"shot{args.shot}_spin.png"

    plot_shot(
        entry,
        net_distance_ft=args.net_distance_ft,
        tm_spin=args.tm_spin,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
