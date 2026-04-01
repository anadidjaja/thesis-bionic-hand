#!/usr/bin/env python3
"""Simple CSV visualizer for timestamp + CH1..CH4."""
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


def main():
    ap = argparse.ArgumentParser(description="Plot CSV sensor data")
    ap.add_argument("csv_path", help="Path to CSV file")
    args = ap.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        print(f"Missing file: {path}", file=sys.stderr)
        sys.exit(1)

    times = []
    channels = {ch: [] for ch in CHANNELS}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Empty CSV", file=sys.stderr)
            sys.exit(1)
        for col in [TIMESTAMP_COL] + CHANNELS:
            if col not in reader.fieldnames:
                print(f"Missing column: {col}", file=sys.stderr)
                sys.exit(1)

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
        print("No valid rows to plot.", file=sys.stderr)
        sys.exit(1)

    t0 = times[0]
    t_sec = [(t - t0).total_seconds() for t in times]

    plt.figure(figsize=(10, 5))
    for ch in CHANNELS:
        plt.plot(t_sec, channels[ch], label=ch)
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(path.name)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
