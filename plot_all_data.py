#!/usr/bin/env python3
"""Overview plot of per-CSV means, optionally computed over fixed time windows."""
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


def print_plot_table(rows):
    headers = ["class", "channel", "x", "y"]
    if not rows:
        print("No plottable values.")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    def fmt_line(values):
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_line(headers))
    print(sep)
    for row in rows:
        print(fmt_line(row))


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


def mean_ignore_nan(values):
    finite = [v for v in values if v == v]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def feature_value(values, feature_name):
    finite = [v for v in values if v == v]
    if not finite:
        return float("nan")

    if feature_name == "mean":
        return mean_ignore_nan(finite)
    if feature_name == "mav":
        return mean_ignore_nan([abs(v) for v in finite])
    if feature_name == "rms":
        return (sum(v * v for v in finite) / len(finite)) ** 0.5
    if feature_name == "zcr":
        if len(finite) < 2:
            return 0.0
        count = 0
        for i in range(1, len(finite)):
            if finite[i - 1] == 0:
                continue
            if (finite[i - 1] > 0 and finite[i] < 0) or (finite[i - 1] < 0 and finite[i] > 0):
                count += 1
        return float(count)
    if feature_name == "wl":
        if len(finite) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(finite)):
            total += abs(finite[i] - finite[i - 1])
        return total
    if feature_name == "ssc":
        if len(finite) < 3:
            return 0.0
        count = 0
        for i in range(1, len(finite) - 1):
            diff1 = finite[i] - finite[i - 1]
            diff2 = finite[i] - finite[i + 1]
            if diff1 * diff2 > 0:
                count += 1
        return float(count)
    raise ValueError(f"Unknown feature: {feature_name}")


def combine_series(series_list, window_seconds, feature_name):
    if not series_list:
        return None

    if window_seconds is None:
        combined = {ch: [] for ch in CHANNELS}
        for _, series in series_list:
            for ch in CHANNELS:
                combined[ch].extend(series[ch])
        for ch in CHANNELS:
            combined[ch] = [feature_value(combined[ch], feature_name)]
        return [0.0], combined

    window_map = {}
    for times, series in series_list:
        start = times[0]
        end = times[-1]
        cur = start
        while cur <= end:
            nxt = cur + window_seconds
            key = round(cur - start, 6)
            window_idx = [i for i, t in enumerate(times) if cur <= t < nxt]
            if window_idx:
                bucket = window_map.setdefault(key, {ch: [] for ch in CHANNELS})
                for ch in CHANNELS:
                    vals = [series[ch][i] for i in window_idx]
                    bucket[ch].append(feature_value(vals, feature_name))
            cur = nxt

    if not window_map:
        return [0.0], {ch: [float("nan")] for ch in CHANNELS}

    x_vals = sorted(window_map.keys())
    combined = {ch: [] for ch in CHANNELS}
    for x in x_vals:
        for ch in CHANNELS:
            combined[ch].append(mean_ignore_nan(window_map[x][ch]))
    return x_vals, combined


def main():
    ap = argparse.ArgumentParser(description="Plot per-CSV mean values in a folder hierarchy")
    ap.add_argument("--root", default="data", help="Root folder containing class subfolders")
    ap.add_argument("--downsample", type=int, default=1, help="Keep every Nth sample")
    ap.add_argument("--alpha", type=float, default=0.25, help="Line transparency")
    ap.add_argument("--seconds", type=float, default=None, help="Mean over fixed windows of N seconds")
    feature_group = ap.add_mutually_exclusive_group()
    feature_group.add_argument("--mav", action="store_true", help="Plot mean absolute value")
    feature_group.add_argument("--rms", action="store_true", help="Plot root mean square")
    feature_group.add_argument("--zcr", action="store_true", help="Plot zero crossing rate")
    feature_group.add_argument("--wl", action="store_true", help="Plot waveform length")
    feature_group.add_argument("--ssc", action="store_true", help="Plot slope sign changes")
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

    if args.mav:
        selected_feature = "mav"
    elif args.rms:
        selected_feature = "rms"
    elif args.zcr:
        selected_feature = "zcr"
    elif args.wl:
        selected_feature = "wl"
    elif args.ssc:
        selected_feature = "ssc"
    else:
        selected_feature = "mean"

    combined_mode = args.seconds is not None or any([args.mav, args.rms, args.zcr, args.wl, args.ssc])
    table_rows = []

    n_rows = len(class_dirs)
    n_cols = len(selected_channels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), sharex=False)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    global_y_max = None
    if combined_mode:
        for r, class_dir in enumerate(class_dirs):
            csvs = sorted(class_dir.glob("*.csv"))
            if not csvs:
                continue
            series_list = []
            for csv_path in csvs:
                loaded = load_csv(csv_path, args.downsample, selected_channels)
                if loaded is None:
                    continue
                series_list.append(loaded)
            combined = combine_series(series_list, args.seconds, selected_feature)
            if combined is None:
                continue
            x_vals, windowed = combined
            for c, ch in enumerate(selected_channels):
                ax = axes[r][c]
                y = windowed[ch]
                finite_vals = [v for v in y if v == v]
                if finite_vals:
                    local_max = max(finite_vals)
                    if global_y_max is None or local_max > global_y_max:
                        global_y_max = local_max
                if args.seconds is None:
                    ax.scatter([0.0] * len(y), y, alpha=args.alpha, s=24)
                else:
                    ax.plot(x_vals, y, alpha=args.alpha, linewidth=1.2)
                for x, yv in zip(([0.0] * len(y)) if args.seconds is None else x_vals, y):
                    table_rows.append([
                        class_dir.name,
                        ch,
                        f"{x:.6f}",
                        "nan" if yv != yv else f"{yv:.6f}",
                    ])
            for c, ch in enumerate(selected_channels):
                ax = axes[r][c]
                if r == 0:
                    ax.set_title(ch)
                if c == 0:
                    ax.set_ylabel(class_dir.name)
                if r == n_rows - 1:
                    ax.set_xlabel("Time (s)" if args.seconds is not None else "CSV mean")
                if args.seconds is None:
                    ax.set_xlim(-0.5, 0.5)
    else:
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
                    y = channels[ch]
                    finite_vals = [v for v in y if v == v]
                    if finite_vals:
                        local_max = max(finite_vals)
                        if global_y_max is None or local_max > global_y_max:
                            global_y_max = local_max
                    ax.plot(t_sec, y, alpha=args.alpha, linewidth=0.8)
                    for x, yv in zip(t_sec, y):
                        table_rows.append([
                            class_dir.name,
                            ch,
                            f"{x:.6f}",
                            "nan" if yv != yv else f"{yv:.6f}",
                        ])
            for c, ch in enumerate(selected_channels):
                ax = axes[r][c]
                if r == 0:
                    ax.set_title(ch)
                if c == 0:
                    ax.set_ylabel(class_dir.name)
                if r == n_rows - 1:
                    ax.set_xlabel("Time (s)")

    if global_y_max is not None:
        for row in axes:
            for ax in row:
                ymin, _ = ax.get_ylim()
                ax.set_ylim(ymin, global_y_max)

    if combined_mode:
        if args.seconds is None:
            fig.suptitle(
                f"Combined {selected_feature} in {root} (per class, per channel)",
                y=0.995,
            )
        else:
            fig.suptitle(
                f"Combined {selected_feature} every {args.seconds:g} s in {root} (per class, per channel)",
                y=0.995,
            )
    else:
        fig.suptitle(f"All CSVs in {root} (per class, per channel)", y=0.995)
    print_plot_table(table_rows)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
