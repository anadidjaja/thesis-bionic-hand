#!/usr/bin/env python3
"""Plot before/after 50 Hz notch filter for one CSV."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import iirnotch, filtfilt
except Exception as e:  # pragma: no cover
    print("Missing dependency: scipy/numpy/matplotlib. Install with: pip install scipy numpy matplotlib", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"
SAMPLE_RATE_HZ = 1000
NOTCH_HZ = 50.0
NOTCH_Q = 30.0


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_csv(path: Path):
    times = []
    channels = {ch: [] for ch in CHANNELS}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + CHANNELS:
            if col not in reader.fieldnames:
                return None
        for row in reader:
            ts_raw = row.get(TIMESTAMP_COL, "")
            if not ts_raw:
                continue
            try:
                ts = parse_iso(ts_raw)
            except Exception:
                continue
            times.append(ts)
            for ch in CHANNELS:
                try:
                    channels[ch].append(float(row[ch]))
                except Exception:
                    channels[ch].append(float("nan"))
    if not times:
        return None
    t0 = times[0]
    t_sec = [(t - t0).total_seconds() for t in times]
    return t_sec, channels


def apply_notch(arr):
    b, a = iirnotch(w0=NOTCH_HZ, Q=NOTCH_Q, fs=SAMPLE_RATE_HZ)
    return filtfilt(b, a, arr)

def compute_fft(y):
    n = len(y)
    if n == 0:
        return None, None
    yf = np.fft.rfft(y)
    xf = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)
    mag = np.abs(yf) / n
    return xf, mag


def main():
    ap = argparse.ArgumentParser(description="Plot before/after 50 Hz notch filtering")
    ap.add_argument("csv_path", help="Path to CSV file")
    ap.add_argument("--channel", default="CH1", choices=CHANNELS, help="Channel to plot")
    ap.add_argument("--max-seconds", type=float, default=5.0, help="Limit plot to first N seconds")
    ap.add_argument("--max-freq", type=float, default=200.0, help="Max frequency for FFT plot (Hz)")
    ap.add_argument("--zoom-center", type=float, default=50.0, help="Center frequency for zoomed FFT (Hz)")
    ap.add_argument("--zoom-width", type=float, default=20.0, help="Zoom window width (Hz)")
    ap.add_argument("--log-fft", action="store_true", help="Use log scale for FFT magnitude")
    args = ap.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        print(f"Missing file: {path}", file=sys.stderr)
        sys.exit(1)

    loaded = load_csv(path)
    if loaded is None:
        print("Failed to load CSV (missing columns?)", file=sys.stderr)
        sys.exit(1)

    t_sec, channels = loaded
    ch = args.channel
    y = np.asarray(channels[ch], dtype=float)
    y_f = apply_notch(y)

    # Limit to first N seconds for readability
    if args.max_seconds > 0:
        keep = [i for i, t in enumerate(t_sec) if t <= args.max_seconds]
        if keep:
            idx_end = keep[-1] + 1
            t_sec = t_sec[:idx_end]
            y = y[:idx_end]
            y_f = y_f[:idx_end]

    xf_raw, mag_raw = compute_fft(y)
    xf_f, mag_f = compute_fft(y_f)

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    # Time domain
    axes[0, 0].plot(t_sec, y, color="tab:blue", linewidth=0.9)
    axes[0, 0].set_title(f"{path.name} - {ch} (raw)")
    axes[0, 0].set_ylabel("Value")

    axes[1, 0].plot(t_sec, y_f, color="tab:orange", linewidth=0.9)
    axes[1, 0].set_title(f"{path.name} - {ch} (50 Hz notch)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Value")

    # Frequency domain
    if xf_raw is not None and mag_raw is not None:
        mask = xf_raw <= args.max_freq
        axes[0, 1].plot(xf_raw[mask], mag_raw[mask], color="tab:blue", linewidth=0.9)
        if args.log_fft:
            axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("FFT (raw)")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")

    if xf_f is not None and mag_f is not None:
        mask = xf_f <= args.max_freq
        axes[1, 1].plot(xf_f[mask], mag_f[mask], color="tab:orange", linewidth=0.9)
        if args.log_fft:
            axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("FFT (notch)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude")

    # Zoomed FFT around center frequency
    zc = args.zoom_center
    zw = args.zoom_width / 2.0
    zmin, zmax = max(0.0, zc - zw), zc + zw
    if xf_raw is not None and mag_raw is not None:
        mask = (xf_raw >= zmin) & (xf_raw <= zmax)
        axes[0, 2].plot(xf_raw[mask], mag_raw[mask], color="tab:blue", linewidth=0.9)
        if args.log_fft:
            axes[0, 2].set_yscale("log")
    axes[0, 2].set_title(f"FFT zoom (raw) {zmin:.0f}-{zmax:.0f} Hz")
    axes[0, 2].set_xlabel("Frequency (Hz)")
    axes[0, 2].set_ylabel("Magnitude")

    if xf_f is not None and mag_f is not None:
        mask = (xf_f >= zmin) & (xf_f <= zmax)
        axes[1, 2].plot(xf_f[mask], mag_f[mask], color="tab:orange", linewidth=0.9)
        if args.log_fft:
            axes[1, 2].set_yscale("log")
    axes[1, 2].set_title(f"FFT zoom (notch) {zmin:.0f}-{zmax:.0f} Hz")
    axes[1, 2].set_xlabel("Frequency (Hz)")
    axes[1, 2].set_ylabel("Magnitude")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
