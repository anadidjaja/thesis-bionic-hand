#!/usr/bin/env python3
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


def file_creation_dt(path: Path) -> datetime:
    st = path.stat()
    birth = getattr(st, "st_birthtime", None)
    ts = birth if birth is not None else st.st_mtime
    return datetime.fromtimestamp(ts)


def load_csv(path: Path, channels):
    times = []
    series = {ch: [] for ch in channels}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + list(channels):
            if col not in reader.fieldnames:
                return None
        for row in reader:
            ts_raw = row.get(TIMESTAMP_COL, "")
            if not ts_raw:
                continue
            try:
                t = datetime.fromisoformat(ts_raw)
            except Exception:
                continue
            times.append(t)
            for ch in channels:
                try:
                    series[ch].append(float(row[ch]))
                except Exception:
                    series[ch].append(float("nan"))
    if not times:
        return None
    t0 = times[0]
    x = [(t - t0).total_seconds() for t in times]
    return x, series


def pick_files(csvs, num):
    with_dates = [(p, file_creation_dt(p)) for p in csvs]
    with_dates.sort(key=lambda t: t[1])
    earliest = with_dates[:num]
    latest = with_dates[-num:] if num > 0 else []

    selected = {}
    for p, dt in earliest:
        selected[p] = ("earliest", dt)
    for p, dt in latest:
        if p in selected:
            selected[p] = ("both", dt)
        else:
            selected[p] = ("latest", dt)

    return with_dates, selected


def main():
    ap = argparse.ArgumentParser(description="Plot earliest and latest CSVs by file creation time")
    ap.add_argument("--root", default="data", help="Root folder containing CSV files")
    ap.add_argument("--num", type=int, required=True, help="Take N earliest and N latest files")
    for ch in CHANNELS:
        ap.add_argument(f"--{ch}", action="store_true", help=f"Show only {ch}")
    args = ap.parse_args()

    if args.num < 1:
        print("--num must be >= 1", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root)
    if not root.exists():
        print(f"Missing folder: {root}", file=sys.stderr)
        sys.exit(1)

    selected_channels = [ch for ch in CHANNELS if getattr(args, ch)]
    if not selected_channels:
        selected_channels = CHANNELS

    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        print(f"No CSV files found under {root}", file=sys.stderr)
        sys.exit(1)

    _, selected = pick_files(csvs, args.num)
    selected_items = sorted(selected.items(), key=lambda t: file_creation_dt(t[0]))
    if not selected_items:
        print("No files selected.", file=sys.stderr)
        sys.exit(1)

    print("group    | created_at           | file")
    print("---------+----------------------+----------------------------------------------")
    for p, (group, dt) in selected_items:
        print(f"{group:<8} | {dt.isoformat(sep=' ', timespec='seconds')} | {p}")

    n_rows = len(selected_channels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.7 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for p, (group, _) in selected_items:
        loaded = load_csv(p, selected_channels)
        if loaded is None:
            continue
        x, series = loaded
        label = f"{group}: {p.parent.name}/{p.name}"
        for i, ch in enumerate(selected_channels):
            ax = axes[i]
            ax.plot(x, series[ch], linewidth=0.9, alpha=0.8, label=label)
            ax.set_ylabel(ch)

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Earliest and Latest CSVs in {root} (num={args.num})", y=0.995)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
