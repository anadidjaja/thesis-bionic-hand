import argparse
import csv
import os
from statistics import median

import numpy as np
import matplotlib.pyplot as plt


def load_time_ecg(csv_path, time_col="time_ms", ecg_col="ecg"):
    times_ms = []
    ecg = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if time_col not in row or ecg_col not in row:
                raise ValueError(f"Missing columns. Found: {reader.fieldnames}")
            t = row[time_col].strip()
            v = row[ecg_col].strip()
            if not t or not v:
                continue
            try:
                times_ms.append(float(t))
                ecg.append(float(v))
            except ValueError:
                continue
    if len(times_ms) < 2:
        raise ValueError("Not enough samples to compute FFT.")
    return np.array(times_ms), np.array(ecg)


def estimate_fs(times_ms):
    dt = np.diff(times_ms) / 1000.0
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("All time deltas are zero or negative.")
    return 1.0 / median(dt)


def main():
    parser = argparse.ArgumentParser(description="Simple FFT from ECG CSV.")
    parser.add_argument("csv_path", help="Path to CSV file (with time_ms and ecg columns).")
    parser.add_argument("--time-col", default="time_ms", help="Time column name.")
    parser.add_argument("--ecg-col", default="ecg", help="ECG column name.")
    parser.add_argument("--max-hz", type=float, default=None, help="Limit plot to this max frequency.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise SystemExit(f"File not found: {args.csv_path}")

    times_ms, ecg = load_time_ecg(args.csv_path, args.time_col, args.ecg_col)
    fs = estimate_fs(times_ms)

    ecg = ecg - np.mean(ecg)
    n = ecg.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(ecg)) / n

    if args.max_hz is not None:
        mask = freqs <= args.max_hz
        freqs = freqs[mask]
        spectrum = spectrum[mask]

    print(f"Samples: {n}")
    print(f"Estimated fs: {fs:.2f} Hz")
    print(f"FFT bins: {freqs.size}")

    plt.figure()
    plt.plot(freqs, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("ECG FFT")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
