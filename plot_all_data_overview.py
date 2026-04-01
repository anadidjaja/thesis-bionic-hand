#!/usr/bin/env python3
"""Overview plot: all CSVs in data/ grouped by class and channel."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("Missing dependency: matplotlib. Install with: pip install matplotlib", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_csv(path: Path, downsample: int, channels):
    times = []
    series = {ch: [] for ch in channels}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + list(channels):
            if col not in reader.fieldnames:
                return None
        idx = 0
        for row in reader:
            if downsample > 1 and (idx % downsample != 0):
                idx += 1
                continue
            idx += 1
            ts_raw = row.get(TIMESTAMP_COL, "")
            if not ts_raw:
                continue
            try:
                ts = parse_iso(ts_raw)
            except Exception:
                continue
            times.append(ts)
            for ch in channels:
                try:
                    series[ch].append(float(row[ch]))
                except Exception:
                    series[ch].append(float("nan"))
    if not times:
        return None
    t0 = times[0]
    t_sec = [(t - t0).total_seconds() for t in times]
    return t_sec, series


def main():
    ap = argparse.ArgumentParser(description="Plot all CSVs in a folder hierarchy")
    ap.add_argument("--root", default="data", help="Root folder containing class subfolders")
    ap.add_argument("--downsample", type=int, default=1, help="Keep every Nth sample")
    ap.add_argument("--alpha", type=float, default=0.25, help="Line transparency")
    # Optional filters
    for ch in CHANNELS:
        ap.add_argument(f"--{ch}", action="store_true", help=f"Show only {ch}")
    ap.add_argument("--fist", action="store_true", help="Show only fist class")
    ap.add_argument("--open", action="store_true", help="Show only open class")
    ap.add_argument("--pen-grip", dest="pen_grip", action="store_true", help="Show only pen-grip class")
    ap.add_argument("--point", action="store_true", help="Show only point class")
    ap.add_argument("--rest", action="store_true", help="Show only rest class")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Missing folder: {root}", file=sys.stderr)
        sys.exit(1)

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        print(f"No class subfolders found in {root}", file=sys.stderr)
        sys.exit(1)

    selected_channels = [ch for ch in CHANNELS if getattr(args, ch)]
    if not selected_channels:
        selected_channels = CHANNELS

    selected_classes = []
    if args.fist:
        selected_classes.append("fist")
    if args.open:
        selected_classes.append("open")
    if args.pen_grip:
        selected_classes.append("pen-grip")
    if args.point:
        selected_classes.append("point")
    if args.rest:
        selected_classes.append("rest")

    if selected_classes:
        class_dirs = [p for p in class_dirs if p.name in selected_classes]
        if not class_dirs:
            print("No class folders matched the requested filters.", file=sys.stderr)
            sys.exit(1)

    n_rows = len(class_dirs)
    n_cols = len(selected_channels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), sharex=False)
    # Normalize axes indexing to 2D list-like access
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for r, class_dir in enumerate(class_dirs):
        csvs = sorted(class_dir.glob("*.csv"))
        if not csvs:
            continue
        for csv_path in csvs:
            loaded = load_csv(csv_path, args.downsample, selected_channels)
            if loaded is None:
                continue
            t_sec, channels = loaded
            for c, ch in enumerate(selected_channels):
                ax = axes[r][c]
                ax.plot(t_sec, channels[ch], alpha=args.alpha, linewidth=0.8)
        for c, ch in enumerate(selected_channels):
            ax = axes[r][c]
            if r == 0:
                ax.set_title(ch)
            if c == 0:
                ax.set_ylabel(class_dir.name)
            if r == n_rows - 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle(f"All CSVs in {root} (per class, per channel)", y=0.995)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
