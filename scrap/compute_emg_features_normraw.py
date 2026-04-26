#!/usr/bin/env python3
"""Compute MAV, RMS, ZCR, WL, SSC per channel after raw normalization."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print("Missing dependency: numpy. Install with: pip install numpy", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"


def mav(vals):
    return sum(abs(v) for v in vals) / len(vals) if vals else float("nan")


def rms(vals):
    return (sum(v * v for v in vals) / len(vals)) ** 0.5 if vals else float("nan")


def zcr(vals):
    if len(vals) < 2:
        return 0
    count = 0
    for i in range(1, len(vals)):
        if vals[i - 1] == 0:
            continue
        if (vals[i - 1] > 0 and vals[i] < 0) or (vals[i - 1] < 0 and vals[i] > 0):
            count += 1
    return count


def wl(vals):
    total = 0.0
    for i in range(1, len(vals)):
        total += abs(vals[i] - vals[i - 1])
    return total


def ssc(vals):
    count = 0
    for i in range(1, len(vals) - 1):
        diff1 = vals[i] - vals[i - 1]
        diff2 = vals[i] - vals[i + 1]
        if diff1 * diff2 > 0:
            count += 1
    return count


def normalize(vals):
    arr = np.asarray(vals, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return arr.tolist()
    return ((arr - mu) / sigma).tolist()


def main():
    ap = argparse.ArgumentParser(description="Compute MAV, RMS, ZCR, WL, SSC per channel (normalized)")
    ap.add_argument("path", help="CSV file or root folder with class subfolders")
    ap.add_argument("--cut-seconds", type=float, default=0.0, help="Trim first N seconds")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Missing path: {path}", file=sys.stderr)
        sys.exit(1)

    def compute_for_csv(csv_path: Path):
        values = {ch: [] for ch in CHANNELS}
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return None
            for col in [TIMESTAMP_COL] + CHANNELS:
                if col not in reader.fieldnames:
                    return None
            t0 = None
            for row in reader:
                ts_raw = row.get(TIMESTAMP_COL, "")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except Exception:
                    continue
                if t0 is None:
                    t0 = ts
                if args.cut_seconds > 0 and ts < t0 + timedelta(seconds=args.cut_seconds):
                    continue
                for ch in CHANNELS:
                    try:
                        values[ch].append(float(row[ch]))
                    except Exception:
                        pass
        return values

    if path.is_file():
        values = compute_for_csv(path)
        if values is None:
            print(f"Skipping {path} (missing columns or empty).", file=sys.stderr)
            sys.exit(1)
        print(path)
        for ch in CHANNELS:
            vals = normalize(values[ch])
            print(
                f"{ch}: MAV={mav(vals):.6f} RMS={rms(vals):.6f} "
                f"ZCR={zcr(vals)} WL={wl(vals):.6f} SSC={ssc(vals)}"
            )
        return

    class_dirs = sorted([p for p in path.iterdir() if p.is_dir()])
    if not class_dirs:
        print("No class folders found.", file=sys.stderr)
        sys.exit(1)

    for class_dir in class_dirs:
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            print(f"Skipping {class_dir} (no CSVs).", file=sys.stderr)
            continue
        csv_path = csv_files[0]
        values = compute_for_csv(csv_path)
        if values is None:
            print(f"Skipping {csv_path} (missing columns or empty).", file=sys.stderr)
            continue
        print(f"\n{class_dir.name} -> {csv_path.name}")
        for ch in CHANNELS:
            vals = normalize(values[ch])
            print(
                f"{ch}: MAV={mav(vals):.6f} RMS={rms(vals):.6f} "
                f"ZCR={zcr(vals)} WL={wl(vals):.6f} SSC={ssc(vals)}"
            )


if __name__ == "__main__":
    main()
