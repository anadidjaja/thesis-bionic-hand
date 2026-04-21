#!/usr/bin/env python3
"""Plot feature correlation heatmaps for the four input pipelines used by the models.

This covers both NN and SVM variants because the correlation is computed on the
shared input features, not on the classifier itself.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, pstdev

try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("Missing dependency: numpy/matplotlib. Install with: pip install numpy matplotlib", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.signal import iirnotch, filtfilt
except Exception as e:  # pragma: no cover
    print("Missing dependency: scipy. Install with: pip install scipy", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"
CUT_SECONDS = 1.0
SAMPLE_RATE_HZ = 1000
WINDOW_MS = 200
STEP_MS = 100
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
STEP_SAMPLES = max(1, int(SAMPLE_RATE_HZ * STEP_MS / 1000))
NOTCH_HZ = 50.0
NOTCH_Q = 30.0


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def normalize_raw(ch_vals):
    arr = np.asarray(ch_vals, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return arr.tolist()
    return ((arr - mu) / sigma).tolist()


def apply_notch(signal_vals):
    if len(signal_vals) < 3:
        return signal_vals
    b, a = iirnotch(w0=NOTCH_HZ, Q=NOTCH_Q, fs=SAMPLE_RATE_HZ)
    return filtfilt(b, a, signal_vals)


def window_features(vals):
    n = len(vals)
    if n == 0:
        return None
    mav = sum(abs(v) for v in vals) / n
    rms = (sum(v * v for v in vals) / n) ** 0.5
    zcr = 0
    for i in range(1, n):
        if vals[i - 1] == 0:
            continue
        if (vals[i - 1] > 0 and vals[i] < 0) or (vals[i - 1] < 0 and vals[i] > 0):
            zcr += 1
    wl = 0.0
    for i in range(1, n):
        wl += abs(vals[i] - vals[i - 1])
    ssc = 0
    for i in range(1, n - 1):
        diff1 = vals[i] - vals[i - 1]
        diff2 = vals[i] - vals[i + 1]
        if diff1 * diff2 > 0:
            ssc += 1
    return [mav, rms, zcr, wl, ssc]


def extract_simple_windows(csv_path: Path, normalize: bool):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + CHANNELS:
            if col not in reader.fieldnames:
                return None

        timestamps = []
        channel_values = [[] for _ in CHANNELS]
        t0 = None
        for row in reader:
            ts_raw = row.get(TIMESTAMP_COL, "")
            if not ts_raw:
                continue
            try:
                ts = parse_iso(ts_raw)
            except Exception:
                continue
            if t0 is None:
                t0 = ts
            if ts < t0 + timedelta(seconds=CUT_SECONDS):
                continue
            vals = []
            bad = False
            for ch in CHANNELS:
                try:
                    vals.append(float(row[ch]))
                except Exception:
                    bad = True
                    break
            if bad:
                continue
            timestamps.append(ts)
            for idx, v in enumerate(vals):
                channel_values[idx].append(v)

    if not timestamps:
        return None

    if normalize:
        channel_values = [normalize_raw(ch) for ch in channel_values]

    if any(len(ch) < WINDOW_SAMPLES for ch in channel_values):
        return None

    windows = []
    for start in range(0, len(channel_values[0]) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        feats = []
        for ch_vals in channel_values:
            window = ch_vals[start:start + WINDOW_SAMPLES]
            feats.extend(window)
        windows.append(feats)

    return windows if windows else None


def extract_complete_features(csv_path: Path, normalize: bool, use_notch: bool):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + CHANNELS:
            if col not in reader.fieldnames:
                return None

        timestamps = []
        channel_values = [[] for _ in CHANNELS]
        t0 = None
        for row in reader:
            ts_raw = row.get(TIMESTAMP_COL, "")
            if not ts_raw:
                continue
            try:
                ts = parse_iso(ts_raw)
            except Exception:
                continue
            if t0 is None:
                t0 = ts
            if ts < t0 + timedelta(seconds=CUT_SECONDS):
                continue
            vals = []
            bad = False
            for ch in CHANNELS:
                try:
                    vals.append(float(row[ch]))
                except Exception:
                    bad = True
                    break
            if bad:
                continue
            timestamps.append(ts)
            for idx, v in enumerate(vals):
                channel_values[idx].append(v)

    if not timestamps:
        return None

    if normalize:
        channel_values = [normalize_raw(ch) for ch in channel_values]

    if use_notch:
        filtered = []
        for ch_vals in channel_values:
            arr = np.asarray(ch_vals, dtype=float)
            filtered.append(apply_notch(arr).tolist())
        channel_values = filtered

    features = []
    for ch_vals in channel_values:
        if len(ch_vals) < WINDOW_SAMPLES:
            return None
        win_feats = []
        for start in range(0, len(ch_vals) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = ch_vals[start:start + WINDOW_SAMPLES]
            wf = window_features(window)
            if wf is not None:
                win_feats.append(wf)
        if not win_feats:
            return None
        for k in range(len(win_feats[0])):
            col = [w[k] for w in win_feats]
            features.append(mean(col))
            features.append(pstdev(col))

    features.append(len(timestamps))
    features.append((timestamps[-1] - timestamps[0]).total_seconds())
    return features


def load_pipeline_features(root: Path, pipeline: str, selected_class: str | None, csv_name: str | None, csv_index: int | None):
    X = []
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if selected_class is not None:
        class_dirs = [p for p in class_dirs if p.name == selected_class]
    for class_dir in class_dirs:
        csv_files = sorted(class_dir.glob("*.csv"))
        if csv_name is not None:
            csv_files = [p for p in csv_files if p.name == csv_name]
        elif csv_index is not None:
            if 0 <= csv_index < len(csv_files):
                csv_files = [csv_files[csv_index]]
            else:
                csv_files = []
        for csv_path in csv_files:
            if pipeline == "simple_raw":
                feats = extract_simple_windows(csv_path, normalize=False)
            elif pipeline == "simple_norm":
                feats = extract_simple_windows(csv_path, normalize=True)
            elif pipeline == "complete_raw_notch":
                feats = extract_complete_features(csv_path, normalize=False, use_notch=True)
            elif pipeline == "complete_norm_notch":
                feats = extract_complete_features(csv_path, normalize=True, use_notch=True)
            else:
                raise ValueError(f"Unknown pipeline: {pipeline}")
            if feats is None:
                continue
            X.extend(feats if isinstance(feats[0], list) else [feats])
    return X


def maybe_reduce_features(X, max_features):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        return None, None
    n_features = X.shape[1]
    if max_features and n_features > max_features:
        idx = np.linspace(0, n_features - 1, max_features, dtype=int)
        X = X[:, idx]
        labels = [f"f{i + 1}" for i in idx]
    else:
        labels = [f"f{i + 1}" for i in range(n_features)]
    return X, labels


def plot_corr(ax, X, labels, title, annotate=False):
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    if annotate and len(labels) <= 20:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    return im


def main():
    ap = argparse.ArgumentParser(description="Plot feature correlation heatmaps for model input pipelines")
    ap.add_argument("--root", default="data", help="Training root folder")
    ap.add_argument(
        "--mode",
        choices=["all", "simple_raw", "simple_norm", "complete_raw_notch", "complete_norm_notch"],
        default="all",
        help="Which pipeline to plot",
    )
    ap.add_argument("--class-name", default=None, help="Only use CSVs from this class folder")
    ap.add_argument("--csv-name", default=None, help="Only use this CSV file name inside the chosen class")
    ap.add_argument("--csv-index", type=int, default=0, help="CSV index inside the chosen class when --csv-name is not set")
    ap.add_argument("--max-features", type=int, default=60, help="Downsample wide feature vectors to this many features")
    ap.add_argument("--annotate", action="store_true", help="Annotate cells when the matrix is small enough")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Missing folder: {root}", file=sys.stderr)
        sys.exit(1)

    pipelines = [
        ("simple_raw", "Simple raw windows"),
        ("simple_norm", "Simple normalized raw windows"),
        ("complete_raw_notch", "Complete features + notch"),
        ("complete_norm_notch", "Complete features + normalized raw + notch"),
    ]
    if args.mode != "all":
        pipelines = [p for p in pipelines if p[0] == args.mode]

    if not pipelines:
        print("No pipelines selected.", file=sys.stderr)
        sys.exit(1)

    if args.class_name is not None:
        available_classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
        if args.class_name not in available_classes:
            print(f"Class not found: {args.class_name}", file=sys.stderr)
            print(f"Available classes: {available_classes}", file=sys.stderr)
            sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = np.asarray(axes).reshape(-1)

    plotted = 0
    last_im = None
    for ax, (pipeline, title) in zip(axes, pipelines):
        csv_index = args.csv_index if args.class_name is not None and args.csv_name is None else None
        csv_name = args.csv_name if args.class_name is not None else None
        X = load_pipeline_features(root, pipeline, args.class_name, csv_name, csv_index)
        reduced, labels = maybe_reduce_features(X, args.max_features)
        if reduced is None:
            ax.axis("off")
            if args.class_name is None:
                ax.set_title(f"{title} (not enough feature dimensions)")
            else:
                ax.set_title(f"{title} ({args.class_name}) (not enough feature dimensions)")
            continue
        if args.class_name is not None:
            title = f"{title} [{args.class_name}]"
            if args.csv_name is not None:
                title += f" / {args.csv_name}"
            else:
                title += f" / index {args.csv_index}"
        last_im = plot_corr(ax, reduced, labels, title, annotate=args.annotate)
        plotted += 1

    for ax in axes[plotted:]:
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), shrink=0.75, location="right")
    fig.suptitle("Feature correlation heatmaps", y=0.995)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
