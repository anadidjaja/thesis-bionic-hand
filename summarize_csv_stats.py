#!/usr/bin/env python3
"""Compute min, max, mean, RMS per channel for one CSV or a folder of CSVs."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"
CUT_SECONDS = 1.0


def rms(vals):
    if not vals:
        return float("nan")
    return (sum(v * v for v in vals) / len(vals)) ** 0.5


def summarize_csv(path: Path, cut_seconds: float):
    values = {ch: [] for ch in CHANNELS}
    with path.open("r", newline="") as f:
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
            if cut_seconds > 0 and ts < t0 + timedelta(seconds=cut_seconds):
                continue
            for ch in CHANNELS:
                try:
                    values[ch].append(float(row[ch]))
                except Exception:
                    pass
    if any(len(values[ch]) == 0 for ch in CHANNELS):
        return None
    stats = {}
    for ch in CHANNELS:
        vals = values[ch]
        stats[ch] = {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "rms": rms(vals),
        }
    return stats


def summarize_class(class_dir: Path, cut_seconds: float):
    values = {ch: [] for ch in CHANNELS}
    csv_files = sorted(class_dir.glob("*.csv"))
    if not csv_files:
        return None
    for csv_path in csv_files:
        stats = summarize_csv(csv_path, cut_seconds)
        if stats is None:
            continue
        # Re-read raw values to aggregate; avoid double parsing by collecting per file
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            for col in [TIMESTAMP_COL] + CHANNELS:
                if col not in reader.fieldnames:
                    continue
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
                if cut_seconds > 0 and ts < t0 + timedelta(seconds=cut_seconds):
                    continue
                for ch in CHANNELS:
                    try:
                        values[ch].append(float(row[ch]))
                    except Exception:
                        pass
    if any(len(values[ch]) == 0 for ch in CHANNELS):
        return None
    stats = {}
    for ch in CHANNELS:
        vals = values[ch]
        stats[ch] = {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "rms": rms(vals),
        }
    return stats


def main():
    ap = argparse.ArgumentParser(description="Summarize CSV channel stats")
    ap.add_argument("path", help="CSV file or folder")
    ap.add_argument("--by-class", action="store_true", help="Aggregate per class folder")
    ap.add_argument("--cut-seconds", type=float, default=1.0, help="Trim first N seconds")
    args = ap.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Missing path: {root}", file=sys.stderr)
        sys.exit(1)

    if root.is_file():
        csv_files = [root]
    else:
        csv_files = sorted(root.rglob("*.csv"))

    if root.is_dir() and args.by_class:
        class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not class_dirs:
            print("No class folders found.", file=sys.stderr)
            sys.exit(1)
        for class_dir in class_dirs:
            stats = summarize_class(class_dir, args.cut_seconds)
            if stats is None:
                print(f"Skipping {class_dir} (missing columns or empty).", file=sys.stderr)
                continue
            print(f"\n{class_dir.name}")
            for ch in CHANNELS:
                s = stats[ch]
                print(
                    f"  {ch}: min={s['min']:.6f} max={s['max']:.6f} "
                    f"mean={s['mean']:.6f} rms={s['rms']:.6f}"
                )
        return

    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_files:
        stats = summarize_csv(csv_path, args.cut_seconds)
        if stats is None:
            print(f"Skipping {csv_path} (missing columns or empty).", file=sys.stderr)
            continue
        print(f"\n{csv_path}")
        for ch in CHANNELS:
            s = stats[ch]
            print(f"  {ch}: min={s['min']:.6f} max={s['max']:.6f} mean={s['mean']:.6f} rms={s['rms']:.6f}")


if __name__ == "__main__":
    main()
